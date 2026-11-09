"""Single-GPU trainer for the RTX PRO 6000 (sm_120).

bf16 autocast over fp32 master weights, gradient accumulation to a token budget, Muon + fused AdamW, WSD,
GPU tail-EMA, periodic held-out loss / bits-per-byte, checkpoint/resume (including each DataLoader worker's
stream position), optional per-block `torch.compile`, and tokens/s + MFU logging.

MNTP (dense only) is applied here, in the main process, to every `mntp_every`-th micro-batch, so its cadence is
exact and independent of the number of DataLoader workers.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from saul.data.loader import mntp_mask
from saul.train.ema import TailEMA
from saul.train.optim import build_optimizers, optimizer_step
from saul.train.schedule import in_decay, wsd_multiplier
from saul.utils.io import MetricsLogger, ensure_dir
from saul.utils.params import format_breakdown
from saul.utils.seed import rng_state, seed_everything, set_rng_state

BF16_PEAK_TFLOPS = 504.0  # RTX PRO 6000 dense BF16 tensor peak


def estimate_flops_per_token(n_matmul_params: int, n_attn_layers: int, attn_dim: int, seq_len: int) -> float:
    """6N for the weight matmuls plus causal attention scores/values (PaLM appendix B, halved for causality)."""
    return 6.0 * n_matmul_params + 6.0 * n_attn_layers * attn_dim * seq_len


class Trainer:
    def __init__(self, model: nn.Module, loader, cfg: dict, run_dir: str, val_batches: list | None = None,
                 token_bytes: np.ndarray | None = None):
        self.model = model
        self.loader = loader
        self.cfg = cfg
        self.run_dir = ensure_dir(run_dir)
        tcfg = cfg["train"]
        ocfg = cfg["optim"]
        self.device = torch.device(tcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        seed_everything(int(cfg.get("seed", 781)))
        self.model.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.use_bf16 = bool(tcfg.get("bf16", self.device.type == "cuda")) and self.device.type == "cuda"
        self.muon, self.adamw = build_optimizers(
            model, muon_lr=float(ocfg["muon_lr"]), adamw_lr=float(ocfg["adamw_lr"]),
            muon_wd=float(ocfg.get("weight_decay", 0.1)), ns_steps=int(ocfg.get("ns_steps", 5)),
            embed_lr=float(ocfg.get("embed_lr", ocfg["adamw_lr"])),
            engram_lr_mult=float(ocfg.get("engram_lr_mult", 5.0)),
            use_muon=str(ocfg.get("optimizer", "muon")).lower() == "muon",
        )
        self.total_steps = int(tcfg["total_steps"])
        self.accum = int(tcfg.get("grad_accum", 1))
        self.clip = float(tcfg.get("grad_clip", 1.0))
        self.seq_len = int(tcfg["seq_len"])
        self.batch_size = int(tcfg["batch_size"])
        self.log_every = int(tcfg.get("log_every", 20))
        self.ckpt_every = int(tcfg.get("ckpt_every", 500))
        self.eval_every = int(tcfg.get("eval_every", 1000))
        self.warmup_frac = float(ocfg.get("warmup_frac", 0.02))
        self.decay_frac = float(ocfg.get("decay_frac", 0.20))
        self.ema = TailEMA(model, decay=float(tcfg.get("ema_decay", 0.999)))
        mcfg = model.config
        self.mntp = bool(tcfg.get("mntp", False)) and mcfg.arch == "dense"
        self.mntp_every = int(tcfg.get("mntp_every", 8))
        self.mntp_rng = np.random.default_rng(int(cfg.get("seed", 781)) + 17)
        self.val_batches = val_batches or []
        self.token_bytes = token_bytes
        self.step = 0
        self.data_workers = 1
        self.data_state: dict[int, dict] = {}
        self.metrics = MetricsLogger(run_dir, cfg.get("wandb"), cfg)
        self.spike_halved = False
        self._loss_window: list[float] = []
        n_attn = sum(1 for b in model.blocks if b.kind == "attn")
        n_params = sum(p.numel() for p in model.parameters())
        n_lookup = sum(p.numel() for n, p in model.named_parameters() if ".tables." in n)
        self.flops_per_token = estimate_flops_per_token(n_params - n_lookup, n_attn,
                                                        mcfg.n_heads * mcfg.head_dim, self.seq_len)
        print(format_breakdown(model))
        print(f"params={n_params:,} device={self.device} bf16={self.use_bf16} mntp={self.mntp} "
              f"flops/token={self.flops_per_token:.3e}")
        if bool(tcfg.get("compile", False)) and self.device.type == "cuda":
            mode = os.environ.get("SAUL_COMPILE_MODE") or None
            for b in model.blocks:
                b.compile(dynamic=False, mode=mode) if mode else b.compile(dynamic=False)
            print(f"[compile] per-block torch.compile enabled mode={mode or 'default'}")

    # ------------------------------------------------------------------------------------------------ batches
    def _prepare(self, batch: dict, global_micro: int) -> dict:
        ws = batch.pop("_wstate", None)
        if ws is not None:
            w, W, st = ws
            self.data_workers = int(W)
            self.data_state[int(w)] = st
        batch["mode"] = "causal"
        if self.mntp and self.mntp_every > 0 and global_micro % self.mntp_every == self.mntp_every - 1:
            c = self.model.config
            ids, labels, lm = mntp_mask(batch["input_ids"], batch["doc_ids"], c.pad_token_id, c.mask_token_id,
                                        c.vocab_size, rng=self.mntp_rng, random_low=64)
            batch.update(input_ids=ids, labels=labels, loss_mask=lm, mode="bidirectional")
        if "cu_seqlens" in batch:
            batch["cu_seqlens_cpu"] = batch["cu_seqlens"]  # lets the fla kernels skip a host sync per layer
        nb = self.device.type == "cuda"
        return {k: (v.to(self.device, non_blocking=nb) if torch.is_tensor(v) and not k.endswith("_cpu") else v)
                for k, v in batch.items()}

    def _forward(self, batch: dict):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.model(
                input_ids=batch["input_ids"], labels=batch["labels"], doc_ids=batch["doc_ids"],
                position_ids=batch["position_ids"], loss_mask=batch.get("loss_mask"),
                mode=batch.get("mode", "causal"), cu_seqlens=batch.get("cu_seqlens"),
                cu_seqlens_cpu=batch.get("cu_seqlens_cpu"),
            )
        return out.loss

    # ------------------------------------------------------------------------------------------ checkpointing
    def _save(self, tag: str) -> str:
        path = ensure_dir(os.path.join(self.run_dir, f"ckpt_{tag}"))
        self.model.save_pretrained(path)
        torch.save({
            "step": self.step,
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict(),
            "ema": self.ema.state_dict(),
            "loader": {"num_workers": self.data_workers, "workers": self.data_state},
            "rng": rng_state(),
            "mntp_rng": self.mntp_rng.bit_generator.state,
            "spike_halved": self.spike_halved,
        }, os.path.join(path, "trainer.pt"))
        if tag.isdigit():
            self._prune_trainer_states()
        return path

    def _prune_trainer_states(self) -> None:
        """Keep optimizer/EMA state (~1.4 GB each) only for the newest `keep_trainer_states` numbered checkpoints.

        Model weights stay for every checkpoint; `stable`/`final` are never pruned. Without this, a 9,000-step run
        with a checkpoint every 500 steps filled a 200 GB disk and lost its final save.
        """
        keep = int(self.cfg["train"].get("keep_trainer_states", 2))
        if keep <= 0:
            return
        numbered = sorted((int(d[5:]), d) for d in os.listdir(self.run_dir)
                          if d.startswith("ckpt_") and d[5:].isdigit())
        for _, d in numbered[:-keep]:
            p = os.path.join(self.run_dir, d, "trainer.pt")
            if os.path.exists(p):
                os.remove(p)

    def _load_weights(self, path: str) -> None:
        from saul.model.modeling_saul import SaulForCausalLM
        fresh = SaulForCausalLM.from_pretrained(path)
        missing, unexpected = self.model.load_state_dict(fresh.state_dict(), strict=False)
        missing = [m for m in missing if not m.endswith("lm_head.weight")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")

    def init_from(self, path: str) -> None:
        """Weights only: fresh optimizer state, step 0, fresh data stream (used for context extension)."""
        self._load_weights(path)

    def load(self, path: str) -> None:
        st = torch.load(os.path.join(path, "trainer.pt"), map_location="cpu", weights_only=False)
        self._load_weights(path)
        if self.muon is not None and st.get("muon") is not None:
            self.muon.load_state_dict(st["muon"])
        self.adamw.load_state_dict(st["adamw"])
        self.ema.load_state_dict(st["ema"], device=self.device)
        if st.get("loader") and hasattr(self.loader, "dataset"):
            self.loader.dataset.load_state_dict(st["loader"])
            self.data_state = {int(k): v for k, v in st["loader"].get("workers", {}).items()}
            self.data_workers = int(st["loader"].get("num_workers", 1))
        if st.get("rng"):
            set_rng_state(st["rng"])
        if st.get("mntp_rng"):
            self.mntp_rng.bit_generator.state = st["mntp_rng"]
        self.step = int(st["step"])
        self.spike_halved = bool(st.get("spike_halved", False))

    @torch.no_grad()
    def _group_gnorms(self) -> dict:
        """Pre-clip gradient norm per optimizer group (muon, embed, norm, engram, other, matrix)."""
        groups = [("muon", self.muon.param_groups[0]["params"])] if self.muon is not None else []
        groups += [(g.get("name", "group"), g["params"]) for g in self.adamw.param_groups]
        out = {}
        for name, params in groups:
            norms = [torch.linalg.vector_norm(p.grad.float()) for p in params if p.grad is not None]
            if norms:
                out[f"gnorm_{name}"] = float(torch.linalg.vector_norm(torch.stack(norms)))
        return out

    # ------------------------------------------------------------------------------------------------- eval
    def evaluate(self) -> dict:
        if not self.val_batches:
            return {}
        from saul.eval.val import evaluate_lm
        res = evaluate_lm(self.model, self.val_batches, self.token_bytes, self.device, self.use_bf16)
        if self.ema.updates:
            backup = self.ema.copy_to(self.model)
            ema_res = evaluate_lm(self.model, self.val_batches, self.token_bytes, self.device, self.use_bf16)
            self.ema.restore(self.model, backup)
            res.update({f"ema_{k}": v for k, v in ema_res.items()})
        self.metrics.log(res, self.step)
        print(f"[eval] step {self.step} " + " ".join(f"{k} {v:.4f}" for k, v in res.items()))
        return res

    # ------------------------------------------------------------------------------------------------ train
    def train(self, max_steps: int | None = None) -> dict:
        self.model.train()
        steps = min(max_steps or self.total_steps, self.total_steps)
        stable_step = int(self.total_steps * (1 - self.decay_frac))
        data = iter(self.loader)
        t0 = time.perf_counter()
        tokens = 0
        last_log = time.perf_counter()
        log_tokens = 0
        data_wait = 0.0
        saved_at = -1
        while self.step < steps:
            if self.muon is not None:
                self.muon.zero_grad(set_to_none=True)
            self.adamw.zero_grad(set_to_none=True)
            loss_acc = torch.zeros((), device=self.device)
            for micro in range(self.accum):
                tw = time.perf_counter()
                batch = self._prepare(next(data), self.step * self.accum + micro)
                data_wait += time.perf_counter() - tw
                loss = self._forward(batch)
                (loss / self.accum).backward()
                loss_acc += loss.detach()
                n = int(batch["input_ids"].numel())
                tokens += n
                log_tokens += n
            loss_val = float(loss_acc) / self.accum
            if not math.isfinite(loss_val):
                self._save(f"nan_{self.step}")
                raise FloatingPointError(f"non-finite loss at step {self.step}")
            will_log = (self.step + 1) % self.log_every == 0 or self.step == 0
            group_gnorms = self._group_gnorms() if will_log else {}
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            mult = wsd_multiplier(self.step, self.total_steps, self.warmup_frac, self.decay_frac)
            optimizer_step(self.muon, self.adamw, mult)
            if in_decay(self.step, self.total_steps, self.decay_frac):
                self.ema.update(self.model)
            self._loss_window = (self._loss_window + [loss_val])[-50:]
            med = sorted(self._loss_window)[len(self._loss_window) // 2]
            if len(self._loss_window) > 20 and loss_val > 1.3 * med and not self.spike_halved:
                for g in self.adamw.param_groups:
                    if g.get("name") == "embed":
                        g["_base_lr"] = g.get("_base_lr", g["lr"]) * 0.5
                self.spike_halved = True
                print(f"[stability] loss spike at step {self.step}; embedding LR halved")
            self.step += 1
            if self.step % self.log_every == 0 or self.step == 1:
                dt = time.perf_counter() - last_log
                tps = log_tokens / max(dt, 1e-9)
                mfu = (tps * self.flops_per_token) / (BF16_PEAK_TFLOPS * 1e12)
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                rec = {"loss": loss_val, "lr_mult": mult, "gnorm": float(gnorm),
                       "tokens_per_s": tps, "mfu": mfu, "data_wait_s": data_wait, "gpu_gb": mem, **group_gnorms}
                self.metrics.log(rec, self.step)
                print(f"step {self.step}/{steps} loss {loss_val:.4f} gnorm {rec['gnorm']:.3f} tok/s {tps:,.0f} "
                      f"mfu {mfu:.2%} mem {mem:.1f}G wait {data_wait:.2f}s")
                last_log = time.perf_counter()
                log_tokens = 0
                data_wait = 0.0
            if self.eval_every and self.step % self.eval_every == 0:
                self.evaluate()
            if self.ckpt_every and self.step % self.ckpt_every == 0:
                self._save(str(self.step))
                saved_at = self.step
            if self.step == stable_step:
                self._save("stable")
                saved_at = self.step
        if self.step >= self.total_steps:
            self.evaluate()
            self._save("final")
            if self.ema.updates:
                backup = self.ema.copy_to(self.model)
                self.model.save_pretrained(os.path.join(self.run_dir, "ckpt_final_ema"))
                self.ema.restore(self.model, backup)
        elif saved_at != self.step and max_steps is None:
            self._save(str(self.step))
        self.metrics.finish()
        elapsed = time.perf_counter() - t0
        return {"steps": self.step, "tokens": tokens, "seconds": elapsed,
                "tokens_per_s": tokens / max(elapsed, 1e-9)}
