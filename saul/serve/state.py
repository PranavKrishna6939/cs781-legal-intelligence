"""Fixed decode pools. A request owns one block of `block_size` slots; its prompt KV is stored once per block.

Per-layer tensors come from each mixer's `serve_spec`:
  group_seq  [G, Lp, *shape]   prompt K/V, written once at prefill
  slot_seq   [S, Ls, *shape]   suffix K/V
  slot_fixed [S, *shape]       GDN state, conv tails, Engram history
"""
from __future__ import annotations

import torch


class StatePools:
    def __init__(self, layer_specs: list[dict], engram_spec: dict | None, n_blocks: int, block_size: int,
                 max_prompt: int, max_new: int, vocab: int, device, dtype):
        self.G = n_blocks
        self.B = block_size
        self.S = n_blocks * block_size
        self.Lp = max_prompt
        self.Ls = max_new
        self.device = device
        self.dtype = dtype
        self.layers: list[dict[str, torch.Tensor]] = []
        self.kinds: list[dict[str, str]] = []
        for spec in layer_specs:
            tensors, kinds = {}, {}
            for name, (kind, shape, dt) in spec.items():
                kinds[name] = kind
                dt = dtype if dt is None else dt
                if kind == "group_seq":
                    tensors[name] = torch.zeros((self.G, self.Lp, *shape), device=device, dtype=dt)
                elif kind == "slot_seq":
                    tensors[name] = torch.zeros((self.S, self.Ls, *shape), device=device, dtype=dt)
                elif kind == "slot_ctx":
                    # Full context cache: prompt is copied in once, decode appends. Flash reads it with seqused.
                    tensors[name] = torch.zeros((self.S, self.Lp + self.Ls, *shape), device=device, dtype=dt)
                elif kind == "slot_fixed":
                    tensors[name] = torch.zeros((self.S, *shape), device=device, dtype=dt)
                else:
                    raise ValueError(kind)
            self.layers.append(tensors)
            self.kinds.append(kinds)
        self.engram = None
        if engram_spec:
            self.engram = torch.zeros((self.S, *engram_spec["hist"][1]), device=device, dtype=torch.long)

        dev = device
        self.plen = torch.zeros(self.G, dtype=torch.long, device=dev)
        self.plen_slot = torch.zeros(self.S, dtype=torch.long, device=dev)
        self.slen = torch.zeros(self.S, dtype=torch.long, device=dev)
        self.pos = torch.zeros(self.S, dtype=torch.long, device=dev)
        self.active = torch.zeros(self.S, dtype=torch.bool, device=dev)
        self.needs_decode = torch.zeros(self.S, dtype=torch.bool, device=dev)
        self.done = torch.zeros(self.S, dtype=torch.bool, device=dev)
        self.token_ids = torch.zeros(self.S, dtype=torch.long, device=dev)
        self.out_tokens = torch.zeros(self.S, self.Ls, dtype=torch.long, device=dev)
        self.words = torch.zeros(self.S, dtype=torch.long, device=dev)
        self.prev_ws = torch.ones(self.S, dtype=torch.bool, device=dev)
        self.ng_keys = torch.full((self.S, self.Ls), -1, dtype=torch.long, device=dev)
        self.freq = torch.zeros(self.S, vocab, dtype=torch.int16, device=dev)
        self.finish = torch.zeros(self.S, dtype=torch.int8, device=dev)  # 0 none, 1 words, 2 length, 3 stop
        # sampling parameters (filled on admit)
        self.temp = torch.ones(self.S, device=dev)
        self.min_p = torch.full((self.S,), 0.05, device=dev)
        self.top_p = torch.ones(self.S, device=dev)
        self.eta = torch.full((self.S,), 3e-3, device=dev)
        self.trunc = torch.ones(self.S, dtype=torch.int8, device=dev)  # 0 none, 1 min_p, 2 top_p, 3 eta
        self.ngram_n = torch.zeros(self.S, dtype=torch.int8, device=dev)
        self.ngram_max = torch.ones(self.S, dtype=torch.int16, device=dev)
        self.freq_pen = torch.zeros(self.S, device=dev)
        self.freq_win = torch.full((self.S,), 64, dtype=torch.long, device=dev)
        self.target_words = torch.full((self.S,), -1, dtype=torch.long, device=dev)
        self.max_new = torch.full((self.S,), max_new, dtype=torch.long, device=dev)
        self.greedy = torch.zeros(self.S, dtype=torch.bool, device=dev)
        self.forced = torch.full((self.S, self.Ls), -1, dtype=torch.long, device=dev)
        self.stop_ids = torch.full((self.S, 4), -1, dtype=torch.long, device=dev)

    def zero_(self) -> None:
        for layer in self.layers:
            for t in layer.values():
                t.zero_()
        if self.engram is not None:
            self.engram.zero_()
        for t in (self.plen, self.plen_slot, self.slen, self.pos, self.words, self.out_tokens, self.freq,
                  self.finish, self.token_ids):
            t.zero_()
        self.active.zero_()
        self.needs_decode.zero_()
        self.done.zero_()
        self.prev_ws.fill_(True)
        self.ng_keys.fill_(-1)
        self.forced.fill_(-1)
        self.stop_ids.fill_(-1)

    def reset_block(self, g: int) -> None:
        s0, s1 = g * self.B, (g + 1) * self.B
        self.slen[s0:s1] = 0
        self.words[s0:s1] = 0
        self.pos[s0:s1] = 0
        self.plen_slot[s0:s1] = 0
        self.token_ids[s0:s1] = 0
        self.out_tokens[s0:s1] = 0
        self.freq[s0:s1] = 0
        self.finish[s0:s1] = 0
        self.ng_keys[s0:s1] = -1
        self.forced[s0:s1] = -1
        self.stop_ids[s0:s1] = -1
        self.prev_ws[s0:s1] = True
        self.done[s0:s1] = False
        self.active[s0:s1] = False
        self.needs_decode[s0:s1] = False
        if self.engram is not None:
            self.engram[s0:s1] = 0

    def install_prompt(self, g: int, n: int, plen: int, exports: list[dict]) -> None:
        """Copy one prompt's exported tensors into block `g` and its first `n` slots."""
        self.plen[g] = plen
        s0 = g * self.B
        self.plen_slot[s0:s0 + self.B] = plen
        for i, exp in enumerate(exports):
            layer, kinds = self.layers[i], self.kinds[i]
            for name, value in exp.items():
                if name not in layer:
                    continue
                kind = kinds[name]
                if kind == "group_seq":
                    layer[name][g, :plen].copy_(value[:plen])
                elif kind == "slot_ctx":
                    layer[name][s0:s0 + n, :plen].copy_(value[:plen].unsqueeze(0).expand(n, *value[:plen].shape))
                elif kind == "slot_fixed":
                    slot = layer[name][s0]
                    if value.shape == slot.shape:
                        src = value.unsqueeze(0).expand(n, *value.shape)
                    elif value.shape == layer[name][s0:s0 + n].shape:
                        src = value
                    else:
                        raise RuntimeError(f"{name} export shape {tuple(value.shape)} does not match {tuple(slot.shape)}")
                    layer[name][s0:s0 + n].copy_(src)
        self.pos[s0:s0 + n] = plen
        self.active[s0:s0 + n] = True

    def bytes(self) -> int:
        n = 0
        for layer in self.layers:
            n += sum(t.numel() * t.element_size() for t in layer.values())
        if self.engram is not None:
            n += self.engram.numel() * self.engram.element_size()
        n += self.freq.numel() * self.freq.element_size()
        n += self.out_tokens.numel() * self.out_tokens.element_size()
        return n
