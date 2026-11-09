"""Packed prefill and the decode step. CUDA graphs are keyed by (slots, prompt bucket, suffix bucket)."""
from __future__ import annotations

import os

import torch

from saul.serve.packing import pack_prompts, segment_spans


class ModelRunner:
    def __init__(self, model, pools, cfg):
        self.model = model
        self.pools = pools
        self.cfg = cfg
        self.device = pools.device
        self.pad_id = int(getattr(model.config, "pad_token_id", 0) or 0)
        v = model.config.vocab_size
        self.logits = torch.empty(pools.S, v, dtype=torch.float32, device=self.device)
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.mempool = None
        self.use_graphs = bool(cfg.cuda_graphs and self.device.type == "cuda" and os.environ.get("SAUL_SERVE_EAGER") != "1")
        self.captures = 0
        self.hits = 0
        self.decode_fn = model.serve_decode
        self.compiled = False
        need = cfg.max_prompt_len + cfg.max_new_tokens + 1
        model._rope(self.device, need)
        if self.device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            # Keep Inductor fusion, but explicitly disable its CUDA graphs. Cache mutation skips
            # many graph partitions, not all: a later partition can attempt capture on the HTTP
            # worker thread and fail the cudagraph-tree thread-local assertion after main-thread
            # warmup. Our outer graphs also stay off to avoid nested allocator/capture issues.
            if os.environ.get("SAUL_SERVE_COMPILE", "1") == "1":
                self.decode_fn = torch.compile(model.serve_decode, dynamic=False,
                                               options={"triton.cudagraphs": False})
                self.use_graphs = False
                self.compiled = True
                self._warmup_compile()
            elif self.use_graphs:
                self._warmup_cublas()

    def _warmup_compile(self) -> None:
        """Compile the full-batch shape once. Smaller batches reuse it by padding inactive slots."""
        sb = self.pools.S
        b = self.pools.B
        self.pools.needs_decode[:sb] = True
        self.pools.slen[:sb] = 1
        self.pools.plen[: sb // b] = 1
        self.pools.plen_slot[:sb] = 1
        self.pools.pos[:sb] = 1
        self.pools.token_ids[:sb] = 4
        self._eager(sb, self.cfg.prompt_buckets[0], self.cfg.suffix_buckets[0])
        self.pools.zero_()
        self.logits.zero_()
        torch.cuda.synchronize()

    def _warmup_cublas(self) -> None:
        """The first GEMM creates the cuBLAS handle. That call is illegal during CUDA graph capture."""
        b = self.pools.B
        self.pools.needs_decode[:b] = True
        self.pools.slen[:b] = 1
        self.pools.plen[0] = 1
        self.pools.plen_slot[:b] = 1
        self.pools.pos[:b] = 1
        self.pools.token_ids[:b] = 4
        self.use_graphs = False
        self._eager(b, self.cfg.prompt_buckets[0], self.cfg.suffix_buckets[0])
        self.use_graphs = True
        self.pools.zero_()
        self.logits.zero_()
        torch.cuda.synchronize()

    def prefill(self, jobs: list[dict]) -> torch.Tensor:
        """jobs: {g, n, ids}. Installs prompt state. Returns last-token logits [J, V] in job order."""
        # CED decoder layers attend over a local window plus encoder memory. The dense
        # right-padded flash prefill has no mask for that, so CED stays on the packed path.
        dense = (self.device.type == "cuda" and self.model.engram is None
                 and not getattr(self.model, "is_ced", False)
                 and all(block.kind == "attn" for block in self.model.blocks))
        if dense:
            return self._prefill_rows(jobs)
        return self._prefill_packed(jobs)

    def _prefill_rows(self, jobs: list[dict]) -> torch.Tensor:
        rows = [job["ids"] for job in jobs]
        width = max(len(row) for row in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long, device=self.device)
        lengths = torch.empty(len(rows), dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            if row:
                ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
            lengths[i] = max(len(row), 1)
        logits, exports = self.model.serve_prefill_rows(ids, lengths)
        for i, job in enumerate(jobs):
            plen = int(lengths[i])
            pieces = []
            for exp in exports:
                pieces.append({name: val[i, :plen] for name, val in exp.items()})
            self.pools.install_prompt(job["g"], job["n"], plen, pieces)
        return logits

    def _prefill_packed(self, jobs: list[dict]) -> torch.Tensor:
        packed = pack_prompts([j["ids"] for j in jobs], self.pad_id, multiple=128)
        device = self.device
        ids = torch.tensor([packed["ids"]], dtype=torch.long, device=device)
        docs = torch.tensor([packed["docs"]], dtype=torch.long, device=device)
        pos = torch.tensor([packed["pos"]], dtype=torch.long, device=device)
        spans = segment_spans(packed["docs"])
        cu_list = [0] + [end for _, end, _ in spans]
        cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
        cu_cpu = torch.tensor(cu_list, dtype=torch.long)
        logits, exports, hist = self.model.serve_prefill_packed(ids, pos, docs, cu, cu_cpu)
        real = [sp for sp in spans if sp[2] >= 0]
        if len(real) != len(jobs):
            raise RuntimeError(f"packed {len(real)} segments for {len(jobs)} prompts")
        first = torch.empty(len(jobs), logits.shape[-1], dtype=torch.float32, device=device)
        for i, ((start, end, _), job) in enumerate(zip(real, jobs)):
            pieces = []
            for li, exp in enumerate(exports):
                piece = {}
                for name, val in exp.items():
                    kind = self.pools.kinds[li][name]
                    if kind == "slot_ctx" and val.dim() == 4:
                        val = val[0]
                    piece[name] = val[start:end] if kind in ("group_seq", "slot_ctx") else val[i]
                pieces.append(piece)
            self.pools.install_prompt(job["g"], job["n"], end - start, pieces)
            if hist is not None:
                s0 = job["g"] * self.pools.B
                h = hist[end - 1]
                self.pools.engram[s0:s0 + job["n"]].copy_(h.unsqueeze(0).expand(job["n"], -1))
            first[i] = logits[end - 1].float()
        return first

    def _views(self, sb: int, lp: int, ls: int) -> dict:
        pools = self.pools
        b = pools.B
        gb = sb // b
        layers = []
        idx = torch.arange(sb, device=self.device)
        kv_index = (pools.slen[:sb] - 1).clamp(min=0)
        needs = pools.needs_decode[:sb]
        for i, layer in enumerate(pools.layers):
            view = {"idx": idx, "kv_index": kv_index, "needs": needs, "block": b,
                    "plen": pools.plen[:gb], "plen_slot": pools.plen_slot[:sb], "pos": pools.pos[:sb]}
            for name, tensor in layer.items():
                kind = pools.kinds[i][name]
                if kind == "group_seq":
                    view[name] = tensor[:gb, :lp]
                elif kind == "slot_seq":
                    view[name] = tensor[:sb, :ls]
                elif kind == "slot_ctx":
                    # Prefix of the batch dim stays contiguous, so flash can read it with no copy.
                    view[name] = tensor[:sb]
                else:
                    view[name] = tensor[:sb]
            layers.append(view)
        eng = None if pools.engram is None else pools.engram[:sb]
        return {"layers": layers, "engram": eng, "block": b}

    def _eager(self, sb: int, lp: int, ls: int) -> None:
        st = self._views(sb, lp, ls)
        out = self.decode_fn(self.pools.token_ids[:sb], self.pools.pos[:sb], st)
        self.logits[:sb].copy_(out.float())

    def decode(self, sb: int, lp: int, ls: int) -> torch.Tensor:
        if sb == 0:
            return self.logits[:0]
        # One compiled shape. Inactive slots are masked by needs_decode and ignored by the sampler.
        if self.compiled:
            sb = self.pools.S
        if not self.use_graphs:
            self._eager(sb, lp, ls)
            return self.logits[:sb]
        key = (sb,)
        if key not in self.graphs:
            self._capture(key, sb, lp, ls)
            return self.logits[:sb]
        self.graphs[key].replay()
        self.hits += 1
        return self.logits[:sb]

    def _capture(self, key, sb, lp, ls) -> None:
        if self.mempool is None:
            self.mempool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        # On this Torch build the capture records the step without writing its outputs. Replay is the
        # step that actually fills logits and KV, so the capturing call replays once before returning.
        torch.cuda.synchronize()
        with torch.cuda.graph(g, pool=self.mempool):
            self._eager(sb, lp, ls)
        self.graphs[key] = g
        self.captures += 1
        g.replay()
