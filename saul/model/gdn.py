"""Gated DeltaNet: flash-linear-attention chunk kernel on GPU (`impl='fla'`), pure-PyTorch recurrence otherwise.

Both paths compute  S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T,  y_t = S_t^T (q_t / sqrt(d_k))
with L2-normalised q, k and the state reset at every document boundary. The fla kernel scales q by 1/sqrt(d_k)
by default, so the reference applies the same scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.model.conv_block import conv_step, conv_tail, doc_causal_conv
from saul.model.norm import ZeroCenteredRMSNorm


def gdn_recurrent(q, k, v, alpha, beta, doc_ids: torch.Tensor | None, state: torch.Tensor | None = None,
                  snapshots: list | None = None):
    """Reference recurrence. q,k: [B,T,H,Dk] (normalised, q pre-scaled), v: [B,T,H,Dv], alpha/beta: [B,T,H].

    If `snapshots` is a list, append `(doc_id, S[0])` after the last token of each segment (packed batch 1).
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype
    q, k, v, alpha, beta = (t.float() for t in (q, k, v, alpha, beta))
    S = q.new_zeros(B, H, Dk, Dv) if state is None else state.float()
    ys = []
    for t in range(T):
        if doc_ids is not None and t > 0:
            reset = (doc_ids[:, t] != doc_ids[:, t - 1])[:, None, None, None]
            S = torch.where(reset, torch.zeros_like(S), S)
        S, y = gdn_step(q[:, t], k[:, t], v[:, t], alpha[:, t], beta[:, t], S)
        ys.append(y)
        if snapshots is not None:
            end = t + 1 == T or (doc_ids is not None and bool((doc_ids[:, t + 1] != doc_ids[:, t]).any()))
            if end:
                doc = int(doc_ids[0, t]) if doc_ids is not None else 0
                snapshots.append((doc, S[0].detach().clone()))
    return torch.stack(ys, dim=1).to(dtype), S


def gdn_step(q, k, v, alpha, beta, S):
    """One token. q,k: [B,H,Dk], v: [B,H,Dv], alpha/beta: [B,H], S: [B,H,Dk,Dv] (float32)."""
    q, k, v = q.float(), k.float(), v.float()
    a = alpha.float()[:, :, None, None]
    b = beta.float()[:, :, None]
    S = a * S
    v_new = b * (v - torch.einsum("bhde,bhd->bhe", S, k))
    S = S + k[:, :, :, None] * v_new[:, :, None, :]
    return S, torch.einsum("bhde,bhd->bhe", S, q)


class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, head_dim: int = 128, conv_kernel: int = 4, impl: str = "torch"):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.conv_kernel = conv_kernel
        self.impl = impl
        inner = n_heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.z_proj = nn.Linear(d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, d_model, bias=False)
        self.W_a = nn.Linear(d_model, n_heads, bias=True)
        self.W_b = nn.Linear(d_model, n_heads, bias=False)
        self.A = nn.Parameter(torch.empty(n_heads).uniform_(-4.0, -1.0))
        self.q_conv = nn.Conv1d(inner, inner, conv_kernel, groups=inner, padding=0, bias=False)
        self.k_conv = nn.Conv1d(inner, inner, conv_kernel, groups=inner, padding=0, bias=False)
        self.v_conv = nn.Conv1d(inner, inner, conv_kernel, groups=inner, padding=0, bias=False)
        self.out_norm = ZeroCenteredRMSNorm(head_dim)
        nn.init.zeros_(self.o_proj.weight)

    def _convs(self, x, doc_ids, cache, phase):
        outs = []
        for name, proj, conv in (("q", self.q_proj, self.q_conv), ("k", self.k_proj, self.k_conv),
                                 ("v", self.v_proj, self.v_conv)):
            p = proj(x)
            if phase == "step":
                y, cache[f"{name}_tail"] = conv_step(p, cache[f"{name}_tail"], conv.weight)
            else:
                y = doc_causal_conv(p, conv.weight, doc_ids)
                if phase == "prefill":
                    cache[f"{name}_tail"] = conv_tail(p, doc_ids, self.conv_kernel)
            outs.append(F.silu(y))
        return outs

    def forward(self, x: torch.Tensor, doc_ids: torch.Tensor | None, cu_seqlens: torch.Tensor | None = None,
                cache: dict | None = None, phase: str = "train") -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q, k, v = self._convs(x, doc_ids, cache, phase)
        v = v.view(B, T, H, D)
        # autocast runs normalize in fp32; the fla kernels need q, k and v in one dtype
        q = F.normalize(q.view(B, T, H, D), dim=-1, eps=1e-6).to(v.dtype)
        k = F.normalize(k.view(B, T, H, D), dim=-1, eps=1e-6).to(v.dtype)
        beta = torch.sigmoid(self.W_b(x).float())
        g = -torch.exp(self.A.float())[None, None, :] * F.softplus(self.W_a(x).float())  # log decay
        scale = D ** -0.5
        if phase == "step":
            S, y = gdn_step((q * scale)[:, 0], k[:, 0], v[:, 0], torch.exp(g[:, 0]), beta[:, 0], cache["S"])
            cache["S"] = S
            y = y[:, None].to(x.dtype)
        elif self.impl == "fla" and x.is_cuda:
            y, final = _fla_forward(q, k, v, g, beta, doc_ids, scale, phase == "prefill")
            if phase == "prefill":
                cache["S"] = final
        else:
            y, S = gdn_recurrent(q * scale, k, v, torch.exp(g), beta, doc_ids)
            if phase == "prefill":
                cache["S"] = S
        y = self.out_norm(y)
        gate = torch.sigmoid(self.z_proj(x)).view(B, T, H, D)
        return self.o_proj((y * gate).reshape(B, T, H * D))

    def serve_spec(self) -> dict:
        inner = self.n_heads * self.head_dim
        k = self.conv_kernel - 1
        return {
            "S": ("slot_fixed", (self.n_heads, self.head_dim, self.head_dim), torch.float32),
            "q_tail": ("slot_fixed", (k, inner), None),
            "k_tail": ("slot_fixed", (k, inner), None),
            "v_tail": ("slot_fixed", (k, inner), None),
        }

    def _segment_tails(self, projected: dict[str, torch.Tensor], doc_ids: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Conv tails at each real segment end. projected values are [1, T, C] (pre-conv)."""
        from saul.serve.packing import segment_spans

        spans = segment_spans(doc_ids[0] if doc_ids is not None else torch.zeros(projected["q"].shape[1], dtype=torch.long))
        out = {name: [] for name in projected}
        for start, end, doc in spans:
            if doc < 0:
                continue
            for name, full in projected.items():
                out[name].append(conv_tail(full[:, start:end], None, self.conv_kernel)[0])
        return {name: torch.stack(parts) for name, parts in out.items()}

    def serve_export(self, x, doc_ids):
        """Train-equivalent output plus per-segment state and conv tails (real documents only)."""
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        projected = {}
        activated = []
        for name, proj, conv in (("q", self.q_proj, self.q_conv), ("k", self.k_proj, self.k_conv),
                                 ("v", self.v_proj, self.v_conv)):
            p = proj(x)
            projected[name] = p
            activated.append(F.silu(doc_causal_conv(p, conv.weight, doc_ids)))
        q, k, v = activated
        v = v.view(B, T, H, D)
        q = F.normalize(q.view(B, T, H, D), dim=-1, eps=1e-6).to(v.dtype)
        k = F.normalize(k.view(B, T, H, D), dim=-1, eps=1e-6).to(v.dtype)
        beta = torch.sigmoid(self.W_b(x).float())
        g = -torch.exp(self.A.float())[None, None, :] * F.softplus(self.W_a(x).float())
        scale = D ** -0.5
        tails = self._segment_tails(projected, doc_ids)
        if self.impl == "fla" and x.is_cuda:
            y, final = _fla_forward(q, k, v, g, beta, doc_ids, scale, True, all_segments=True)
            states = _real_segment_states(final, doc_ids)
        else:
            snaps: list = []
            y, _ = gdn_recurrent(q * scale, k, v, torch.exp(g), beta, doc_ids, snapshots=snaps)
            states = torch.stack([s for doc, s in snaps if doc >= 0])
        y = self.out_norm(y)
        gate = torch.sigmoid(self.z_proj(x)).view(B, T, H, D)
        exp = {"S": states, "q_tail": tails["q"], "k_tail": tails["k"], "v_tail": tails["v"]}
        return self.o_proj((y * gate).reshape(B, T, H * D)), exp

    def serve_decode(self, x, st) -> torch.Tensor:
        """One token. Updates conv tails and S in place. x: [Sb, 1, d]."""
        sb, _, _ = x.shape
        H, D = self.n_heads, self.head_dim
        activated = []
        needs = st["needs"].view(sb, 1, 1)
        for name, proj, conv in (("q", self.q_proj, self.q_conv), ("k", self.k_proj, self.k_conv),
                                 ("v", self.v_proj, self.v_conv)):
            y, new_tail = conv_step(proj(x), st[f"{name}_tail"], conv.weight)
            st[f"{name}_tail"].copy_(torch.where(needs, new_tail, st[f"{name}_tail"]))
            activated.append(F.silu(y))
        q, k, v = activated
        v = v.view(sb, 1, H, D)
        q = F.normalize(q.view(sb, 1, H, D), dim=-1, eps=1e-6).to(v.dtype)
        k = F.normalize(k.view(sb, 1, H, D), dim=-1, eps=1e-6).to(v.dtype)
        beta = torch.sigmoid(self.W_b(x).float())
        g = -torch.exp(self.A.float())[None, None, :] * F.softplus(self.W_a(x).float())
        scale = D ** -0.5
        S, y = gdn_step((q * scale)[:, 0], k[:, 0], v[:, 0], torch.exp(g[:, 0]), beta[:, 0], st["S"])
        st["S"].copy_(torch.where(st["needs"].view(sb, 1, 1, 1), S, st["S"]))
        y = self.out_norm(y[:, None].to(x.dtype))
        gate = torch.sigmoid(self.z_proj(x)).view(sb, 1, H, D)
        return self.o_proj((y * gate).reshape(sb, 1, H * D))


class _VarlenContext:
    """Segment boundaries for the current forward, set by the model outside the compiled blocks.

    `cu_seqlens` changes length every batch (the number of documents varies). Passing it through a compiled block
    would recompile per shape until torch.compile gives up and runs the block eagerly, so only the non-compiled
    `_fla_forward` reads it from here. fla caches its chunk indices by tensor identity, so every layer gets the same
    objects; the CPU copy lets fla build those indices without a device-to-host sync.
    """
    gpu: torch.Tensor | None = None
    cpu: torch.Tensor | None = None


VARLEN = _VarlenContext()


def set_varlen_context(cu_gpu: torch.Tensor | None, cu_cpu: torch.Tensor | None = None) -> None:
    VARLEN.gpu = None if cu_gpu is None else cu_gpu.long()
    VARLEN.cpu = None if cu_cpu is None else cu_cpu.long()


@torch.compiler.disable
def _real_segment_states(final: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """fla returns one state per segment, including a pad segment. Keep real documents, in order."""
    from saul.serve.packing import segment_spans

    docs = [doc for _, _, doc in segment_spans(doc_ids[0])]
    state = final.float()
    if state.shape[0] != len(docs) and state.dim() > 1 and state.shape[1] == len(docs):
        state = state[0]
    if state.shape[0] != len(docs):
        raise RuntimeError(f"fla returned {tuple(final.shape)} states for {len(docs)} segments")
    keep = [i for i, doc in enumerate(docs) if doc >= 0]
    return state[keep]


def _fla_forward(q, k, v, g, beta, doc_ids, scale: float, final_state: bool, all_segments: bool = False):
    """Packed varlen call: [B, T] is flattened to one row of B*T tokens split at the segment boundaries.

    Boundaries come from `VARLEN` (set by the model) or, when absent, from `doc_ids`. With `final_state`, returns
    each row's decoding state: the final state of the segment that ends the row.
    """
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except Exception as exc:
        raise RuntimeError(
            "gdn_impl='fla' needs flash-linear-attention (built for sm_120). Set gdn_impl: torch only for tests; "
            "the reference recurrence is far too slow for pretraining."
        ) from exc
    B, T, H, D = q.shape
    cu, cu_cpu = VARLEN.gpu, VARLEN.cpu
    if cu is None:
        from saul.model.masks import cu_seqlens_from_doc_ids
        cu = (cu_seqlens_from_doc_ids(doc_ids) if doc_ids is not None
              else torch.arange(0, B * T + 1, T, device=q.device, dtype=torch.int32)).long()
        cu_cpu = None
    out, final = chunk_gated_delta_rule(
        q.reshape(1, B * T, H, D), k.reshape(1, B * T, H, D), v.reshape(1, B * T, H, D),
        g.reshape(1, B * T, H), beta.to(q.dtype).reshape(1, B * T, H),
        scale=scale, output_final_state=final_state, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu,
    )
    if final_state and not all_segments:
        ends = torch.arange(1, B + 1, device=q.device) * T
        final = final[torch.searchsorted(cu[1:], ends)].float()
    elif final is not None:
        final = final.float()
    return out.reshape(B, T, H, D), final
