"""Online logit distillation. The teacher is a Phase-1 Saul checkpoint, never an external model.

loss = ce_weight * CE(student, next token) + (1 - ce_weight) * KL(teacher top-k || student) at temperature 1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int = 64) -> torch.Tensor:
    """KL on the teacher's top-k support, renormalised on both sides. Both logits are [N, V]."""
    topv, topi = teacher_logits.detach().topk(k, dim=-1)
    log_p = F.log_softmax(student_logits.gather(-1, topi).float(), dim=-1)
    q = F.softmax(topv.float(), dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


def distillation_loss(student_logits, teacher_logits, targets, k: int = 64, ce_weight: float = 0.5) -> torch.Tensor:
    """student/teacher logits [B, T, V]; targets [B, T] already aligned with the logits (`next_token_targets`)."""
    flat_y = targets.reshape(-1)
    valid = flat_y != -100
    flat_s = student_logits.reshape(-1, student_logits.shape[-1])[valid]
    flat_t = teacher_logits.reshape(-1, teacher_logits.shape[-1])[valid]
    if flat_s.shape[0] == 0:
        return student_logits.sum() * 0.0
    ce = F.cross_entropy(flat_s.float(), flat_y[valid])
    return ce_weight * ce + (1 - ce_weight) * topk_kl(flat_s, flat_t, k=k)
