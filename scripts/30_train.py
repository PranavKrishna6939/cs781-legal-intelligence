#!/usr/bin/env python
"""Stage 30: pretrain one Saul model.

    python scripts/30_train.py --config configs/train/base.yaml [--bench 50] [--resume runs/.../ckpt_500]
    python scripts/30_train.py ... --stop_at_stable            # stop at the start of the decay phase
    python scripts/30_train.py ... --init_from runs/.../ckpt   # weights only, new schedule (context extension)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from saul.data.loader import PackedLMDataset, make_dataloader  # noqa: E402
from saul.data.stats import anneal_weights, mixture_weights  # noqa: E402
from saul.data.tokens import open_split  # noqa: E402
from saul.model.factory import config_from_yaml  # noqa: E402
from saul.model.modeling_saul import build_model  # noqa: E402
from saul.train.trainer import Trainer  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import init_run_dir, read_json  # noqa: E402
from saul.utils.seed import seed_everything  # noqa: E402


def _weights(cfg):
    dcfg = load_config(cfg["data_config"])
    shares = read_json(os.path.join(cfg["tokens"], "token_shares.json"))["train"]
    counts = {s: shares[s]["tokens"] for s in shares}
    if cfg["train"].get("mixture", "stable") == "anneal":
        test_mix = None
        mix_path = os.path.join(dcfg.paths.stats, "test_source_mix.json")
        if dcfg.mixture.anneal_use_test_mix and os.path.exists(mix_path):
            test_mix = read_json(mix_path)
        return anneal_weights(dict(dcfg.mixture.anneal), test_mix, dcfg.mixture.anneal_min_share)
    return mixture_weights(counts, dict(dcfg.mixture.stable))


def _compression_map(cfg, scfg):
    from transformers import PreTrainedTokenizerFast

    from saul.tokenizer.compress import build_compression_map
    tok = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer)
    return build_compression_map(tok, scfg.vocab_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/base.yaml")
    ap.add_argument("--bench", type=int, default=0, help="run this many steps and exit (no checkpoint)")
    ap.add_argument("--resume", default=None, help="full resume: weights, optimizers, step, data stream")
    ap.add_argument("--init_from", default=None, help="weights only; optimizer, step and schedule start fresh")
    ap.add_argument("--stop_at_stable", action="store_true", help="stop at the first decay step (ckpt_stable)")
    ap.add_argument("--enforce_min", action="store_true", default=False)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    if a.resume and a.init_from:
        raise SystemExit("--resume and --init_from are exclusive")
    cfg = load_config(a.config, a.overrides)
    init_run_dir(cfg.run_dir, cfg)
    seed_everything(int(cfg.seed))
    # Train-config keys under `model:` override the model YAML (used by the context-extension stage).
    model_overrides = [f"model.{k}={v}" for k, v in dict(cfg.get("model") or {}).items()]
    scfg = config_from_yaml(cfg.model_config, model_overrides or None)
    # Proxy ablations may sit below the 50M competition minimum; the 300M cap is always enforced.
    enforce_min = a.enforce_min or bool(cfg.get("enforce_min_params", scfg.vocab_size >= 30000))
    model = build_model(scfg, enforce_budget=True, enforce_min=enforce_min)
    if model.engram is not None and not (a.resume or a.init_from):
        model.engram.set_compression(_compression_map(cfg, scfg))

    data_cfg = load_config(cfg.data_config)
    sources = open_split(cfg.tokens, "train", list(data_cfg.sources))
    weights = _weights(cfg)
    tr = cfg.train
    ds = PackedLMDataset(
        sources, weights, seq_len=tr.seq_len,
        bos_id=scfg.bos_token_id, eos_id=scfg.eos_token_id, pad_id=scfg.pad_token_id,
        seed=int(cfg.seed), long_doc_prob=float(tr.get("long_doc_prob", 0.0)),
        long_doc_min=int(tr.get("long_doc_min", tr.seq_len // 2)),
    )
    loader = make_dataloader(ds, batch_size=tr.batch_size, num_workers=int(tr.num_workers),
                             pad_id=scfg.pad_token_id)

    val_batches, token_bytes = [], None
    val_sources = open_split(cfg.tokens, "val", list(data_cfg.sources))
    if val_sources and int(tr.get("eval_sequences", 0)) > 0:
        from saul.eval.val import fixed_val_batches
        vw = {s: float(weights.get(s, 0.0)) or 1.0 for s in val_sources}
        val_batches = fixed_val_batches(val_sources, vw, tr.seq_len, tr.batch_size, int(tr.eval_sequences),
                                        scfg.bos_token_id, scfg.eos_token_id, scfg.pad_token_id)
    tb_path = os.path.join(cfg.tokens, "token_bytes.npy")
    if os.path.exists(tb_path):
        token_bytes = np.load(tb_path)

    trainer = Trainer(model, loader, cfg, cfg.run_dir, val_batches=val_batches, token_bytes=token_bytes)
    if a.resume:
        trainer.load(a.resume)
    elif a.init_from:
        trainer.init_from(a.init_from)
    stop = None
    if a.bench:
        stop = trainer.step + a.bench
    elif a.stop_at_stable:
        stop = int(trainer.total_steps * (1 - trainer.decay_frac))
    stats = trainer.train(max_steps=stop)
    print(stats)


if __name__ == "__main__":
    main()
