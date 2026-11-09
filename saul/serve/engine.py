"""Continuous batching. One iteration admits a prefill batch, decodes every armed slot once, then retires."""
from __future__ import annotations

import time

import torch

from saul.serve.config import EngineConfig, memory_ok
from saul.serve.finish import REASON, Finisher, specials_of
from saul.serve.metrics import Metrics
from saul.serve.runner import ModelRunner
from saul.serve.sampling import TRUNC, sample_from
from saul.serve.sampling import update_freq, store_ngram_keys
from saul.serve.scheduler import Request, RequestOutput, Scheduler
from saul.serve.state import StatePools
from saul.serve.words import WordCounter, stop_flags, suppressed_ids


class LLMEngine:
    def __init__(self, model, tokenizer, cfg: EngineConfig | None = None):
        self.model = model.eval()
        self.tok = tokenizer
        self.cfg = cfg or EngineConfig()
        self.device = torch.device(self.cfg.device if (self.cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
        if self.device.type == "cpu":
            self.cfg.cuda_graphs = False
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.cfg.dtype]
        if self.device.type == "cpu" and dtype != torch.float32:
            dtype = torch.float32
        specs, eng = model.serve_spec()
        free = None
        if self.device.type == "cuda":
            free = torch.cuda.mem_get_info(self.device)[0]
            n_kv = model.config.n_kv_heads
            n_gdn = sum(1 for b in model.blocks if b.kind == "gdn")
            needed = self.cfg.estimate_bytes(len(model.blocks), n_kv, model.config.head_dim, model.config.vocab_size,
                                             n_gdn, model.config.gdn_heads, model.config.gdn_head_dim)
            memory_ok(self.cfg, needed, free)
        self.pools = StatePools(specs, eng, self.cfg.n_blocks, self.cfg.block_size, self.cfg.max_prompt_len,
                                self.cfg.max_new_tokens, model.config.vocab_size, self.device, dtype)
        self.runner = ModelRunner(model, self.pools, self.cfg)
        self.sched = Scheduler(self.cfg.n_blocks, self.cfg.block_size, self.cfg.prefill_token_budget,
                               self.cfg.admit_batch_blocks)
        self.counter = WordCounter(tokenizer, model.config.vocab_size, self.device)
        ids = suppressed_ids(tokenizer, model.config.vocab_size)
        self.suppress = torch.zeros(model.config.vocab_size, dtype=torch.bool, device=self.device)
        if ids:
            self.suppress[torch.tensor(ids, device=self.device)] = True
        self.finisher = Finisher(model, tokenizer, specials_of(tokenizer), self.cfg.rouge_workers)
        self.metrics = Metrics()
        self.pending: list[int] = []
        self.finish_q: list[tuple] = []
        self.owner: list[Request | None] = [None] * self.pools.S
        self.local = [0] * self.pools.S
        self.h_slen = [0] * self.pools.S
        self.h_done = [False] * self.pools.S
        self.h_reason = [0] * self.pools.S
        self.h_plen = [0] * self.pools.G
        self._rid = 0
        self._closed = False
        self._recorders = 0

    # --------------------------------------------------------------------------------------------- requests
    def tokenize(self, prompts: list[str], prepend_bos: bool = True) -> list[list[int]]:
        try:
            rows = self.tok(prompts, add_special_tokens=False)["input_ids"]
        except Exception:
            rows = [self.tok.encode(p, add_special_tokens=False) for p in prompts]
        bos = getattr(self.tok, "bos_token_id", None)
        out = []
        for ids in rows:
            ids = [int(i) for i in ids]
            if prepend_bos and bos is not None and (not ids or ids[0] != bos):
                ids = [int(bos)] + ids
            if not ids:
                ids = [int(bos if bos is not None else 0)]
            if len(ids) > self.cfg.max_prompt_len:
                if self.cfg.truncate_left:
                    ids = ids[-self.cfg.max_prompt_len:]
                else:
                    raise ValueError(f"prompt has {len(ids)} tokens, max_prompt_len is {self.cfg.max_prompt_len}")
            out.append(ids)
        return out

    def submit_ids(self, ids: list[int], params, text: str = "") -> Request:
        if self._closed:
            raise RuntimeError("engine is closed")
        if len(ids) > self.cfg.max_prompt_len and not self.cfg.truncate_left:
            raise ValueError(f"prompt has {len(ids)} tokens, max_prompt_len is {self.cfg.max_prompt_len}")
        req = Request(self._rid, text, list(ids), params)
        self._rid += 1
        try:
            self.sched.submit(req)
        except BufferError:
            req.status = "FAILED"
            req.error = BufferError("request queue is full")
            raise
        return req

    def submit_text(self, text: str, params, prepend_bos: bool = True) -> Request:
        return self.submit_ids(self.tokenize([text], prepend_bos)[0], params, text)

    # --------------------------------------------------------------------------------------------- loop
    def step(self) -> bool:
        if self._closed:
            return False
        self.metrics.iterations += 1
        touched: list[int] = []
        if self.pending:
            slots = list(self.pending)
            self.pending = []
            self._arm(slots)
            sb, lp, ls = self._buckets(slots)
            t0 = time.perf_counter()
            self.runner.decode(sb, lp, ls)
            self.metrics.add("decode", time.perf_counter() - t0)
            idx = torch.tensor(slots, dtype=torch.long, device=self.device)
            t0 = time.perf_counter()
            self._sample(self.runner.logits[idx], idx)
            self.metrics.add("sample", time.perf_counter() - t0)
            touched.extend(slots)
        t0 = time.perf_counter()
        admitted = self.sched.admit(self.cfg.max_prompt_len)
        if admitted:
            for req in admitted:
                self._bind(req)
            jobs = [{"g": req.block, "n": req.params.n, "ids": req.ids} for req in admitted]
            first = self.runner.prefill(jobs)
            self.metrics.prefill_tokens += sum(len(r.ids) for r in admitted)
            self.metrics.admits += len(admitted)
            rows, idxs = [], []
            for j, req in enumerate(admitted):
                for s in range(req.params.n):
                    rows.append(first[j])
                    idxs.append(req.s0 + s)
            idx = torch.tensor(idxs, dtype=torch.long, device=self.device)
            self._sample(torch.stack(rows), idx)
            touched.extend(idxs)
        self.metrics.add("prefill", time.perf_counter() - t0)
        if touched:
            codes = self.pools.finish[torch.tensor(touched, dtype=torch.long, device=self.device)].tolist()
            self._absorb(touched, codes)
        self._maybe_retire()
        t0 = time.perf_counter()
        self._maybe_finish()
        self.metrics.add("finish", time.perf_counter() - t0)
        active = sum(self.owner[s] is not None and not self.h_done[s] for s in range(self.pools.S))
        self.metrics.occupancy.append(active)
        self.metrics.graph_hits = self.runner.hits
        self.metrics.graph_captures = self.runner.captures
        return bool(touched or self.finish_q or self.sched.busy or self.pending)

    def run_until_complete(self) -> None:
        guard = 0
        limit = 100000
        while self.sched.busy or self.pending or self.finish_q:
            progressed = self.step()
            guard += 1
            if not progressed or guard > limit:
                if self.sched.busy or self.pending or self.finish_q:
                    raise RuntimeError("engine made no progress; refusing to spin")
                break

    def generate(self, prompts: list[str], params, prepend_bos: bool = True) -> list[RequestOutput]:
        if isinstance(params, list):
            if len(params) != len(prompts):
                raise ValueError("one SamplingParams per prompt, or a single params for all")
            plist = params
        else:
            plist = [params] * len(prompts)
        ids = self.tokenize(prompts, prepend_bos) if prompts else []
        reqs = [self.submit_ids(row, p, text) for row, p, text in zip(ids, plist, prompts)]
        try:
            self.run_until_complete()
        except BaseException:
            self.close()
            raise
        outs = []
        for req in reqs:
            if req.error is not None:
                raise req.error
            if req.output is None:
                raise RuntimeError(f"request {req.rid} produced no output")
            outs.append(req.output)
        return outs

    def close(self) -> None:
        self._closed = True
        err = RuntimeError("engine shut down")
        for req in list(self.sched.queue):
            req.error = err
            req.status = "FAILED"
        self.sched.queue.clear()
        for req in list(self.sched.groups.values()):
            req.error = err
            req.status = "FAILED"
        self.finish_q.clear()

    # --------------------------------------------------------------------------------------------- internals
    def _bind(self, req: Request) -> None:
        p = req.params
        s0, n = req.s0, p.n
        sl = slice(s0, s0 + n)
        pools = self.pools
        dev = self.device
        pools.temp[sl] = p.temperature if p.temperature > 0 else 1.0
        pools.greedy[sl] = p.temperature <= 0
        pools.min_p[sl] = p.min_p
        pools.top_p[sl] = p.top_p
        pools.eta[sl] = p.eta
        pools.trunc[sl] = 0 if p.temperature <= 0 else TRUNC[p.truncation]
        pools.ngram_n[sl] = p.no_repeat_ngram
        pools.ngram_max[sl] = p.max_ngram_occurrences
        pools.freq_pen[sl] = p.frequency_penalty
        pools.freq_win[sl] = max(1, p.frequency_window)
        pools.target_words[sl] = -1 if p.target_words is None else int(p.target_words)
        pools.max_new[sl] = min(int(p.max_new_tokens), self.cfg.max_new_tokens)
        pools.forced[sl] = -1
        if p.forced_tokens:
            ft = torch.tensor(list(p.forced_tokens), dtype=torch.long, device=dev)
            pools.forced[s0:s0 + n, :ft.numel()] = ft
        pools.stop_ids[sl] = -1
        if p.stop_token_ids:
            st = [int(x) for x in p.stop_token_ids][:4]
            pools.stop_ids[s0:s0 + n, :len(st)] = torch.tensor(st, dtype=torch.long, device=dev)
        for i in range(n):
            self.owner[s0 + i] = req
            self.local[s0 + i] = i
            self.h_slen[s0 + i] = 0
            self.h_done[s0 + i] = False
            self.h_reason[s0 + i] = 0
        self.h_plen[req.block] = len(req.ids)
        if p.record_logits:
            self._recorders += 1

    def _hints(self):
        do_top = do_eta = False
        all_g = True
        levels = set()
        live = False
        for req in self.sched.groups.values():
            live = True
            p = req.params
            if p.temperature > 0:
                all_g = False
                do_top = do_top or p.truncation == "top_p"
                do_eta = do_eta or p.truncation == "eta"
            if p.no_repeat_ngram >= 2:
                levels.add(int(p.no_repeat_ngram))
            if p.seed is not None:
                all_g = False
        return tuple(sorted(levels)), do_top, do_eta, all_g and live

    def _record(self, logits: torch.Tensor, idx: torch.Tensor) -> None:
        if not any(self.owner[s] is not None and self.owner[s].traces is not None for s in idx.tolist()):
            return
        cpu = logits.detach().float().cpu()
        for i, s in enumerate(idx.tolist()):
            req = self.owner[int(s)]
            if req is not None and req.traces is not None:
                req.traces[self.local[int(s)]].append(cpu[i].clone())

    def _sample(self, logits: torch.Tensor, idx: torch.Tensor) -> None:
        if not idx.numel():
            return
        if self._recorders:
            self._record(logits, idx)
        levels, do_top, do_eta, all_g = self._hints()
        nxt = sample_from(logits, self.pools, idx, self.suppress, None, levels, do_top, do_eta, all_g)
        sl = self.pools.slen[idx]
        forced = self.pools.forced[idx, sl]
        nxt = torch.where(forced >= 0, forced, nxt)
        stops = self.pools.stop_ids[idx]
        is_stop = ((nxt[:, None] == stops) & (stops >= 0)).any(dim=-1)
        self._accept(idx, nxt, is_stop)

    def _accept(self, idx: torch.Tensor, nxt: torch.Tensor, is_stop: torch.Tensor) -> None:
        pools = self.pools
        keep = ~is_stop
        sl = pools.slen[idx]
        old_tok = pools.out_tokens[idx, sl]
        pools.out_tokens[idx, sl] = torch.where(keep, nxt, old_tok)
        new_slen = sl + keep.long()
        pools.slen[idx] = new_slen
        pools.token_ids[idx] = torch.where(keep, nxt, pools.token_ids[idx])
        pools.pos[idx] = torch.where(keep, pools.plen_slot[idx] + sl, pools.pos[idx])
        words, prev = self.counter.update(nxt, pools.words[idx], pools.prev_ws[idx])
        pools.words[idx] = torch.where(keep, words, pools.words[idx])
        pools.prev_ws[idx] = torch.where(keep, prev, pools.prev_ws[idx])
        store_ngram_keys(pools, idx, self._hints()[0])
        update_freq(pools, idx, nxt, keep)
        empty = nxt.new_empty(0, dtype=torch.long)
        done, reason = stop_flags(pools.words[idx], pools.prev_ws[idx], new_slen, nxt,
                                  pools.target_words[idx], pools.max_new[idx], empty)
        done = done | is_stop
        reason = torch.where(is_stop, torch.full_like(reason, 3), reason)
        pools.done[idx] = done
        pools.finish[idx] = torch.where(done, reason, torch.zeros_like(reason))
        pools.needs_decode[idx] = False

    def _arm(self, slots: list[int]) -> None:
        if slots:
            self.pools.needs_decode[torch.tensor(slots, dtype=torch.long, device=self.device)] = True

    def _buckets(self, slots: list[int]) -> tuple[int, int, int]:
        cfg = self.cfg
        last = max(slots) + 1
        max_sl = max(self.h_slen[s] for s in slots)
        max_pl = max(self.h_plen[s // self.pools.B] for s in slots)
        # the write index is slen-1, so the suffix bucket must cover `slen` positions
        return (cfg.bucket(last, cfg.slot_buckets),
                cfg.bucket(max(1, max_pl), cfg.prompt_buckets),
                cfg.bucket(max(1, max_sl), cfg.suffix_buckets))

    def _absorb(self, slots: list[int], codes: list[int]) -> None:
        for s, code in zip(slots, codes):
            if code != 3:
                self.h_slen[s] += 1
                self.metrics.generated_tokens += 1
            self.h_reason[s] = int(code)
            if code == 1 and self._short_of_target(s):
                self.pools.done[s] = False
                self.pools.finish[s] = 0
                self.h_done[s] = False
                self.h_reason[s] = 0
                self.pending.append(s)
            elif code == 0:
                self.h_done[s] = False
                self.pending.append(s)
            else:
                self.h_done[s] = True

    def _short_of_target(self, s: int) -> bool:
        req = self.owner[s]
        target = None if req is None else req.params.target_words
        if not target:
            return False
        text = self._decode_slot(s)
        n = len(text.split())
        if n >= target:
            return False
        self.pools.words[s] = n
        self.pools.prev_ws[s] = (not text) or text[-1].isspace()
        return True

    def _decode_slot(self, s: int) -> str:
        n = self.h_slen[s]
        if n <= 0:
            return ""
        ids = self.pools.out_tokens[s, :n].tolist()
        return self._decode_ids(ids)

    def _decode_ids(self, ids: list[int]) -> str:
        if not ids:
            return ""
        try:
            return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except TypeError:
            return self.tok.decode(ids)

    def _maybe_retire(self) -> None:
        for g, req in list(self.sched.groups.items()):
            slots = range(req.s0, req.s0 + req.params.n)
            if not all(self.h_done[s] for s in slots):
                continue
            texts, reasons, gens = [], [], []
            for s in slots:
                ids = self.pools.out_tokens[s, :self.h_slen[s]].tolist()
                texts.append(self._decode_ids(ids))
                reasons.append(REASON.get(self.h_reason[s], "length"))
                gens.append(len(ids))
            self.finish_q.append((req, texts, reasons, gens))
            if req.params.record_logits:
                self._recorders -= 1
            self.sched.free_block(g)
            self.pools.reset_block(g)
            for s in range(g * self.pools.B, (g + 1) * self.pools.B):
                self.owner[s] = None
                self.h_slen[s] = 0
                self.h_done[s] = False
                self.h_reason[s] = 0
            self.pending = [s for s in self.pending if self.owner[s] is not None]

    def _maybe_finish(self) -> None:
        if not self.finish_q:
            return
        if len(self.finish_q) < self.cfg.mbr_batch_groups and (self.sched.busy or self.pending):
            return
        batch = self.finish_q
        self.finish_q = []
        groups = []
        for req, texts, reasons, gens in batch:
            p = req.params
            groups.append({
                "texts": texts, "reasons": reasons, "target": p.target_words,
                "utility": p.mbr.utility, "alpha": p.mbr.alpha,
                "mbr": bool(p.mbr.enabled and p.n > 1),
            })
        chosen = self.finisher.choose(groups)
        now = time.perf_counter()
        for (req, texts, reasons, gens), text in zip(batch, chosen):
            req.output = RequestOutput(
                text=text, candidates=list(texts), finish_reasons=list(reasons),
                prompt_tokens=len(req.ids), gen_tokens=list(gens), latency_s=now - req.arrival,
                logits=req.traces,
            )
            req.status = "DONE"
            self.metrics.finished += 1

    def stats(self) -> dict:
        return self.metrics.snapshot({"pool_bytes": self.pools.bytes(), "slots": self.pools.S,
                                      "graphs": len(self.runner.graphs)})
