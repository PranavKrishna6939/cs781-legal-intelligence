"""BIOES CRF and a banded biaffine span head.

Both heads read the body slice of the backbone hidden state. The CRF's illegal
BIOES transitions are masked. The span head scores spans of length 1..64 only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.ner.labels import IGNORE, N_LABELS, N_SPAN, N_TAGS, NONE, legal_transition_masks

MAX_SPAN = 64
SPAN_RANK = 64
NEG = -1.0e4


class CRF(nn.Module):
    def __init__(self, n_tags: int = N_TAGS):
        super().__init__()
        ok, start, end = legal_transition_masks()
        self.register_buffer("ok", ok)
        self.register_buffer("start_ok", start)
        self.register_buffer("end_ok", end)
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))

    def _scores(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trans = self.trans.masked_fill(~self.ok, NEG)
        start = self.start.masked_fill(~self.start_ok, NEG)
        end = self.end.masked_fill(~self.end_ok, NEG)
        return trans, start, end

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """emissions [B, T, K], tags [B, T], mask [B, T] (True = real token)."""
        log_z = self._log_z(emissions, mask)
        gold = self._gold(emissions, tags, mask)
        # Per token, not per window. A 1,800-token window otherwise dominates the span loss
        # and the clipped gradient wrecks the pretrained backbone.
        lengths = mask.sum(1).clamp(min=1).to(log_z.dtype)
        return ((log_z - gold) / lengths).mean()

    def _log_z(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        trans, start, end = self._scores()
        # alpha[b, k] over the real prefix. mask[:, 0] is always True for a packed body.
        alpha = start + emissions[:, 0]
        for t in range(1, emissions.shape[1]):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0)  # [B, prev, next]
            nxt = torch.logsumexp(scores, dim=1) + emissions[:, t]
            take = mask[:, t].unsqueeze(1)
            alpha = torch.where(take, nxt, alpha)
        last = mask.long().sum(1) - 1
        # end transition from the last real tag. alpha already sits on that position.
        return torch.logsumexp(alpha + end, dim=-1)

    def _gold(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        trans, start, end = self._scores()
        b, tlen, _ = emissions.shape
        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, tlen):
            prev = tags[:, t - 1]
            cur = tags[:, t]
            step = trans[prev, cur] + emissions[:, t].gather(1, cur.unsqueeze(1)).squeeze(1)
            score = score + step * mask[:, t].to(score.dtype)
        last = mask.long().sum(1) - 1
        last_tag = tags.gather(1, last.unsqueeze(1)).squeeze(1)
        return score + end[last_tag]

    def viterbi(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        trans, start, end = self._scores()
        b, tlen, k = emissions.shape
        back = emissions.new_zeros(b, tlen, k, dtype=torch.long)
        score = start + emissions[:, 0]
        for t in range(1, tlen):
            cand = score.unsqueeze(2) + trans.unsqueeze(0)
            best, idx = cand.max(dim=1)
            nxt = best + emissions[:, t]
            take = mask[:, t].unsqueeze(1)
            score = torch.where(take, nxt, score)
            back[:, t] = torch.where(take, idx, back[:, t])
        score = score + end
        last = (mask.long().sum(1) - 1).tolist()
        paths = []
        best_last = score.argmax(-1).tolist()
        back_cpu = back.tolist()
        for i in range(b):
            t = last[i]
            tag = best_last[i]
            path = [tag]
            while t > 0:
                tag = back_cpu[i][t][tag]
                t -= 1
                path.append(tag)
            path.reverse()
            paths.append(path)
        return paths

    def marginals(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Token marginals [B, T, K], 0 on padding."""
        trans, start, end = self._scores()
        b, tlen, k = emissions.shape
        alpha = emissions.new_zeros(b, tlen, k)
        alpha[:, 0] = start + emissions[:, 0]
        for t in range(1, tlen):
            scores = alpha[:, t - 1].unsqueeze(2) + trans.unsqueeze(0)
            nxt = torch.logsumexp(scores, dim=1) + emissions[:, t]
            take = mask[:, t].unsqueeze(1)
            alpha[:, t] = torch.where(take, nxt, alpha[:, t - 1])
        beta = emissions.new_zeros(b, tlen, k)
        last = mask.long().sum(1) - 1
        for i in range(b):
            beta[i, last[i]] = end
        for t in range(tlen - 2, -1, -1):
            # beta_t(i) = logsumexp_j trans[i, j] + emit_{t+1}(j) + beta_{t+1}(j)
            # Only real positions whose next token is also real. Padding must not overwrite beta[last].
            nxt_emit = emissions[:, t + 1] + beta[:, t + 1]
            scores = trans.unsqueeze(0) + nxt_emit.unsqueeze(1)  # [B, i, j]
            nxt = torch.logsumexp(scores, dim=2)
            use = (mask[:, t] & mask[:, t + 1]).unsqueeze(1)
            beta[:, t] = torch.where(use, nxt, beta[:, t])
        # For positions after the true end, alpha was copied forward and beta copied backward,
        # so restrict to the mask.
        log_z = torch.logsumexp(alpha[torch.arange(b, device=emissions.device), last] + end, dim=-1)
        log_m = alpha + beta - log_z.view(b, 1, 1)
        probs = log_m.exp() * mask.unsqueeze(-1).to(log_m.dtype)
        return probs


class SpanHead(nn.Module):
    def __init__(self, d_model: int, rank: int = SPAN_RANK, max_span: int = MAX_SPAN, n_classes: int = N_SPAN):
        super().__init__()
        self.max_span = max_span
        self.start = nn.Linear(d_model, rank)
        self.end = nn.Linear(d_model, rank)
        self.U = nn.Parameter(torch.empty(n_classes, rank, rank))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        nn.init.xavier_uniform_(self.U)

    def loss(self, hidden: torch.Tensor, targets: torch.Tensor, class_weight: torch.Tensor) -> torch.Tensor:
        """hidden [B, T, D] packed body. targets [B, T, max_span], IGNORE where the span does not fit."""
        s = self.start(hidden)
        e = self.end(hidden)
        total = hidden.new_zeros((), dtype=torch.float32)
        count = hidden.new_zeros((), dtype=torch.float32)
        weight = class_weight.float()
        for off in range(self.max_span):
            n = hidden.shape[1] - off
            if n <= 0:
                break
            tgt = targets[:, :n, off]
            valid = tgt != IGNORE
            if not bool(valid.any()):
                continue
            su = torch.einsum("bnr,crx->bncx", s[:, :n].float(), self.U.float())
            logits = torch.einsum("bncx,bnx->bnc", su, e[:, off:off + n].float()) + self.bias.float()
            total = total + F.cross_entropy(logits[valid], tgt[valid], weight=weight, reduction="sum")
            count = count + valid.sum()
        return total / count.clamp(min=1)

    @torch.no_grad()
    def candidates(self, hidden: torch.Tensor, body_mask_len: torch.Tensor, limit: int = 512
                   ) -> list[list[tuple[int, int, list[float]]]]:
        """Per row, up to `limit` spans with P(none) < 0.5. Each is (start, end exclusive, 13 probs)."""
        s = self.start(hidden).float()
        e = self.end(hidden).float()
        out: list[list[tuple[int, int, list[float]]]] = [[] for _ in range(hidden.shape[0])]
        lengths = body_mask_len.tolist()
        for off in range(self.max_span):
            n = hidden.shape[1] - off
            if n <= 0:
                break
            su = torch.einsum("bnr,crx->bncx", s[:, :n], self.U.float())
            logits = torch.einsum("bncx,bnx->bnc", su, e[:, off:off + n]) + self.bias.float()
            probs = torch.softmax(logits, dim=-1)
            keep = probs[:, :, NONE] < 0.5
            for i in range(hidden.shape[0]):
                last = int(lengths[i]) - off
                if last <= 0:
                    continue
                idx = torch.nonzero(keep[i, :last], as_tuple=False).flatten()
                for j in idx.tolist():
                    out[i].append((j, j + off + 1, probs[i, j].tolist()))
        for i, row in enumerate(out):
            if len(row) > limit:
                row.sort(key=lambda sp: sp[2][NONE])
                out[i] = row[:limit]
        return out


class NERHeads(nn.Module):
    def __init__(self, d_model: int, probe: bool = False):
        super().__init__()
        self.probe = probe
        self.emit = nn.Linear(d_model, N_TAGS)
        self.crf = CRF(N_TAGS)
        self.span = None if probe else SpanHead(d_model)
        if probe:
            for p in self.crf.parameters():
                p.requires_grad_(False)
        self.register_buffer("tag_weight", torch.ones(N_TAGS))
        self.register_buffer("span_weight", torch.ones(N_SPAN))

    def set_weights(self, tag_weight: torch.Tensor, span_weight: torch.Tensor | None) -> None:
        self.tag_weight.copy_(tag_weight)
        if span_weight is not None and self.span is not None:
            self.span_weight.copy_(span_weight)

    def _pack(self, hidden: torch.Tensor, body_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = body_mask.long().sum(1)
        width = int(lengths.max())
        packed = hidden.new_zeros(hidden.shape[0], width, hidden.shape[-1])
        mask = torch.zeros(hidden.shape[0], width, dtype=torch.bool, device=hidden.device)
        for i in range(hidden.shape[0]):
            n = int(lengths[i])
            packed[i, :n] = hidden[i, body_mask[i]]
            mask[i, :n] = True
        return packed, mask, lengths

    def forward(self, hidden: torch.Tensor, batch: dict) -> dict[str, torch.Tensor]:
        packed, mask, lengths = self._pack(hidden, batch["body_mask"])
        emissions = self.emit(packed).float()
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            if self.probe:
                loss = F.cross_entropy(
                    emissions[mask], batch_tags(batch, mask), weight=self.tag_weight.float(),
                )
                zero = loss * 0.0
                return {"loss": loss, "nll": loss, "ce": zero, "span": zero,
                        "emissions": emissions, "mask": mask, "lengths": lengths}
            tags = _pack_tags(batch["tags"], batch["body_mask"], mask)
            span_tgt = batch["span_targets"][:, :packed.shape[1]]
            # span targets are indexed in body order, which matches the packed order.
            span_loss = self.span.loss(packed, span_tgt, self.span_weight)
            labelled = mask & (tags != IGNORE)
            if bool(labelled.any()):
                ce = F.cross_entropy(
                    emissions.reshape(-1, emissions.shape[-1]), tags.reshape(-1),
                    weight=self.tag_weight.float(), ignore_index=IGNORE,
                )
            else:
                ce = emissions.sum() * 0.0
            complete = bool(torch.equal(labelled, mask))
            if complete:
                nll = self.crf.neg_log_likelihood(emissions, tags, mask)
            else:
                nll = emissions.sum() * 0.0
            loss = nll + ce + 0.5 * span_loss
            if "teacher_probs" in batch:
                loss = loss + _token_kl(emissions, batch["teacher_probs"], mask)
        return {
            "loss": loss, "nll": nll, "ce": ce, "span": span_loss,
            "emissions": emissions, "mask": mask, "lengths": lengths, "packed": packed,
        }


def _token_kl(emissions: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    tgt = teacher[:, :emissions.shape[1]].float().clamp(min=1e-8)
    tgt = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-8)
    log_p = F.log_softmax(emissions.float() / temperature, dim=-1)
    kl = F.kl_div(log_p, tgt, reduction="none").sum(-1)
    denom = mask.sum().clamp(min=1).to(kl.dtype)
    return (kl * mask.to(kl.dtype)).sum() / denom * (temperature ** 2)


def batch_tags(batch: dict, mask: torch.Tensor) -> torch.Tensor:
    return _pack_tags(batch["tags"], batch["body_mask"], mask)[mask]


def _pack_tags(tags: torch.Tensor, body_mask: torch.Tensor, packed_mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(packed_mask.shape, dtype=torch.long, device=tags.device)
    for i in range(tags.shape[0]):
        n = int(packed_mask[i].sum())
        out[i, :n] = tags[i, body_mask[i]]
    return out
