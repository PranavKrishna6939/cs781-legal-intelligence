"""Sentence head: marker pooling, a BiGRU, two domain CRFs, and a label-shift loss.

The NER CRF is not used here. Its transitions are masked to legal BIOES tags, and
rhetorical roles are a plain 13-way chain.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.rr.data import N_LABELS


class ChainCRF(nn.Module):
    def __init__(self, n_tags: int = N_LABELS):
        super().__init__()
        self.n_tags = n_tags
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))

    def nll_per_example(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-sentence NLL, shape [B]. emissions [B, S, K]."""
        emissions = emissions.float()
        log_z = self._log_z(emissions, mask)
        gold = self._gold(emissions, tags, mask)
        lengths = mask.sum(1).clamp(min=1).to(log_z.dtype)
        return (log_z - gold) / lengths

    def marginals(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emissions = emissions.detach().float()
        trans = self.trans.detach()
        start = self.start.detach()
        end = self.end.detach()
        b, slen, k = emissions.shape
        alpha = emissions.new_zeros(b, slen, k)
        alpha[:, 0] = start + emissions[:, 0]
        for t in range(1, slen):
            scores = alpha[:, t - 1].unsqueeze(2) + trans.unsqueeze(0)
            nxt = torch.logsumexp(scores, dim=1) + emissions[:, t]
            take = mask[:, t].unsqueeze(1)
            alpha[:, t] = torch.where(take, nxt, alpha[:, t - 1])
        beta = emissions.new_zeros(b, slen, k)
        last = mask.long().sum(1) - 1
        for i in range(b):
            if int(last[i]) >= 0 and bool(mask[i].any()):
                beta[i, last[i]] = end
        for t in range(slen - 2, -1, -1):
            nxt_emit = emissions[:, t + 1] + beta[:, t + 1]
            scores = trans.unsqueeze(0) + nxt_emit.unsqueeze(1)
            nxt = torch.logsumexp(scores, dim=2)
            use = (mask[:, t] & mask[:, t + 1]).unsqueeze(1)
            beta[:, t] = torch.where(use, nxt, beta[:, t])
        index = torch.arange(b, device=emissions.device)
        log_z = torch.logsumexp(alpha[index, last.clamp(min=0)] + end, dim=-1)
        log_m = alpha + beta - log_z.view(b, 1, 1)
        return log_m.exp() * mask.unsqueeze(-1).to(log_m.dtype)

    def _log_z(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        alpha = self.start + emissions[:, 0]
        for t in range(1, emissions.shape[1]):
            scores = alpha.unsqueeze(2) + self.trans.unsqueeze(0)
            nxt = torch.logsumexp(scores, dim=1) + emissions[:, t]
            take = mask[:, t].unsqueeze(1)
            alpha = torch.where(take, nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=-1)

    def _gold(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe = tags.clamp(min=0)
        score = self.start[safe[:, 0]] + emissions[:, 0].gather(1, safe[:, 0:1]).squeeze(1)
        for t in range(1, emissions.shape[1]):
            prev = safe[:, t - 1]
            cur = safe[:, t]
            step = self.trans[prev, cur] + emissions[:, t].gather(1, cur.unsqueeze(1)).squeeze(1)
            score = score + step * mask[:, t].to(score.dtype)
        last = mask.long().sum(1) - 1
        last_tag = safe.gather(1, last.clamp(min=0).unsqueeze(1)).squeeze(1)
        return score + self.end[last_tag]


def fourier(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    freqs = torch.arange(n_freq, device=x.device, dtype=torch.float32)
    ang = x.float().unsqueeze(-1) * (2.0 ** freqs) * math.pi
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class RRHeads(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, 512)
        self.domain = nn.Embedding(2, 512)
        self.freq_doc = nn.Linear(16, 512)
        self.freq_win = nn.Linear(8, 512)
        self.gru = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.emit = nn.Linear(512, N_LABELS)
        self.crf = nn.ModuleList([ChainCRF(N_LABELS), ChainCRF(N_LABELS)])
        self.lsp = nn.Linear(1024, 1)
        self.register_buffer("ce_weight", torch.ones(N_LABELS))

    def set_weights(self, weights: torch.Tensor) -> None:
        self.ce_weight.copy_(weights.to(self.ce_weight.device))

    def pool(self, hidden: torch.Tensor, batch: dict) -> torch.Tensor:
        marker = batch["marker_pos"].clamp(min=0)
        b, sent = marker.shape
        d = hidden.shape[-1]
        gather = marker.unsqueeze(-1).expand(b, sent, d)
        marker_h = hidden.gather(1, gather)
        mean = hidden.new_zeros(b, sent, d)
        starts = batch["tok_start"]
        ends = batch["tok_end"]
        mask = batch["sent_mask"]
        for i in range(b):
            for j in range(sent):
                if not bool(mask[i, j]):
                    continue
                a = int(starts[i, j])
                c = int(ends[i, j])
                if c <= a:
                    mean[i, j] = marker_h[i, j]
                else:
                    mean[i, j] = hidden[i, a:c].mean(0)
        return torch.cat([marker_h, mean], dim=-1)

    def encode(self, hidden: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.pool(hidden, batch)
        z = self.proj(s.float())
        z = z + self.domain(batch["domain"]).unsqueeze(1)
        z = z + self.freq_doc(fourier(batch["rel_pos"], 8))
        z = z + self.freq_win(fourier(batch["window_pos"], 4))
        lengths = batch["sent_mask"].sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        packed_u, _ = self.gru(packed)
        u, _ = nn.utils.rnn.pad_packed_sequence(packed_u, batch_first=True, total_length=z.shape[1])
        emissions = self.emit(self.drop(u))
        return u, emissions

    def forward(self, hidden: torch.Tensor, batch: dict, kd_weight: float = 0.0) -> dict:
        u, emissions = self.encode(hidden, batch)
        mask = batch["sent_mask"]
        labels = batch["labels"]
        domain = batch["domain"]
        labeled = mask & (labels >= 0)
        # Keep the CRF term in the autograd graph. Writing into a fresh zeros
        # tensor drops that graph, so the transitions never train.
        parts = []
        marginals = emissions.new_zeros(emissions.shape[0], emissions.shape[1], N_LABELS, dtype=torch.float32)
        for d in (0, 1):
            sel = domain == d
            if not bool(sel.any()):
                continue
            rows = sel.nonzero(as_tuple=False).squeeze(1)
            use = labeled[rows]
            if bool(use.any()):
                # CRF needs a complete chain. Drop a row whose real sentences are not all labelled.
                complete = use.sum(1) == mask[rows].sum(1)
                if bool(complete.any()):
                    keep = rows[complete]
                    parts.append(self.crf[d].nll_per_example(emissions[keep], labels[keep], mask[keep]))
            marginals[rows] = self.crf[d].marginals(emissions[rows], mask[rows]).detach()
        nll_loss = torch.cat(parts).mean() if parts else emissions.new_zeros(())
        ce = F.cross_entropy(
            emissions.float().transpose(1, 2),
            labels.clamp(min=0),
            weight=self.ce_weight.float(),
            reduction="none",
        )
        ce_loss = (ce * labeled.to(ce.dtype)).sum() / labeled.to(ce.dtype).sum().clamp(min=1)
        lsp_loss = self._lsp(u, labels, mask)
        loss = nll_loss + 0.5 * ce_loss + 0.2 * lsp_loss
        if kd_weight > 0 and bool(batch["has_teacher"].any()):
            kl = _kl(emissions, batch["teacher"], batch["has_teacher"] & mask, temperature=2.0)
            gold_rows = labeled.any(1)
            # Gold rows mix the task loss with KL. Pool rows (no gold) are KL only,
            # which the caller applies by zeroing the task loss when kd_only is set.
            if bool(batch.get("kd_only", torch.tensor(False))):
                loss = kl
            else:
                loss = (1.0 - kd_weight) * loss + kd_weight * kl
            _ = gold_rows
        else:
            kl = emissions.new_zeros(())
        return {
            "loss": loss,
            "nll": nll_loss.detach(),
            "ce": ce_loss.detach(),
            "lsp": lsp_loss.detach(),
            "kl": kl.detach() if torch.is_tensor(kl) else kl,
            "emissions": emissions,
            "marginals": marginals,
            "mask": mask,
        }

    def _lsp(self, u: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if u.shape[1] < 2:
            return u.new_zeros(())
        pair = torch.cat([u[:, :-1], u[:, 1:]], dim=-1)
        logit = self.lsp(pair.float()).squeeze(-1)
        gold = (labels[:, 1:] != labels[:, :-1]).to(logit.dtype)
        valid = mask[:, 1:] & mask[:, :-1] & (labels[:, 1:] >= 0) & (labels[:, :-1] >= 0)
        if not bool(valid.any()):
            return logit.new_zeros(())
        bce = F.binary_cross_entropy_with_logits(logit, gold, reduction="none")
        return (bce * valid.to(bce.dtype)).sum() / valid.to(bce.dtype).sum().clamp(min=1)


def _kl(emissions: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p = torch.log(teacher.float().clamp(min=1e-8))
    soft_t = torch.softmax(log_p / temperature, dim=-1)
    log_s = torch.log_softmax(emissions.float() / temperature, dim=-1)
    log_soft = torch.log(soft_t.clamp(min=1e-8))
    kl = (soft_t * (log_soft - log_s)).sum(-1) * (temperature ** 2)
    denom = mask.to(kl.dtype).sum().clamp(min=1)
    return (kl * mask.to(kl.dtype)).sum() / denom
