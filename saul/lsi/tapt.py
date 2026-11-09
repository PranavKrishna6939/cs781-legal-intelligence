"""Masked next-token prediction on the normalised LSI train and dev text."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from saul.cjpe.model import enable_activation_checkpointing
from saul.data.loader import mntp_mask
from saul.lsi import p5_dir
from saul.lsi.const import CHUNK, SEED
from saul.lsi.data import load_token_split
from saul.lsi.model import lr_mult
from saul.model.modeling_saul import SaulForCausalLM
from saul.ner.registry import resolve_checkpoint
from saul.utils.io import ensure_dir


def _documents() -> list[np.ndarray]:
    """Train and dev documents. Test text is not included."""
    docs = []
    for name in ("train", "dev"):
        split = load_token_split(name, os.path.join(p5_dir(), "tok"))
        offsets = split["offsets"]
        flat = split["flat"]
        for index in range(len(offsets) - 1):
            docs.append(flat[int(offsets[index]):int(offsets[index + 1])].astype(np.int64, copy=False))
    return docs


def _pack(docs: list[np.ndarray], seq_len: int, pad_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One pass of packed sequences. Document boundaries stay in doc_ids and positions restart."""
    sequences = []
    cur_i: list[int] = []
    cur_d: list[int] = []
    cur_p: list[int] = []
    filled = 0
    seg = 0
    for doc in docs:
        if len(doc) == 0:
            continue
        offset = 0
        while offset < len(doc):
            take = min(seq_len - filled, len(doc) - offset)
            piece = doc[offset:offset + take]
            cur_i.extend(int(x) for x in piece)
            cur_d.extend([seg] * take)
            cur_p.extend(range(take))
            filled += take
            offset += take
            seg += 1
            if filled == seq_len:
                sequences.append((cur_i, cur_d, cur_p))
                cur_i, cur_d, cur_p = [], [], []
                filled = 0
                seg = 0
    if cur_i:
        pad = seq_len - filled
        cur_i.extend([pad_id] * pad)
        cur_d.extend([-1] * pad)
        cur_p.extend([0] * pad)
        sequences.append((cur_i, cur_d, cur_p))
    if not sequences:
        raise RuntimeError("TAPT corpus is empty")
    ids = np.asarray([row[0] for row in sequences], dtype=np.int64)
    doc_ids = np.asarray([row[1] for row in sequences], dtype=np.int64)
    pos = np.asarray([row[2] for row in sequences], dtype=np.int64)
    return ids, doc_ids, pos


def run_tapt(seq_len: int = CHUNK, lr: float = 1e-4, batch_size: int = 4, passes: int = 1,
             model_name: str = "dense_v1", context: str = "bidirectional") -> str:
    out = os.path.join(p5_dir(), "adapt", model_name)
    marker = os.path.join(out, "done.json")
    if os.path.exists(marker):
        print(f"[tapt] reuse {out}", flush=True)
        return out
    ensure_dir(out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SaulForCausalLM.from_pretrained(resolve_checkpoint(model_name)).to(device)
    enable_activation_checkpointing(model, 2)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    ids, doc_ids, positions = _pack(_documents(), seq_len, int(model.config.pad_token_id))
    if passes > 1:
        ids = np.concatenate([ids] * passes)
        doc_ids = np.concatenate([doc_ids] * passes)
        positions = np.concatenate([positions] * passes)
    use_cuda = device == "cuda"
    total = max(1, int(np.ceil(len(ids) / batch_size)))
    seen = 0
    step = 0
    order = rng.permutation(len(ids))
    for start in range(0, len(order), batch_size):
        pick = order[start:start + batch_size]
        batch = torch.tensor(ids[pick], dtype=torch.long)
        docs = torch.tensor(doc_ids[pick], dtype=torch.long)
        pos = torch.tensor(positions[pick], dtype=torch.long)
        masked, labels, loss_mask = mntp_mask(
            batch, docs, int(model.config.pad_token_id), int(model.config.mask_token_id),
            int(model.config.vocab_size), rng=rng, random_low=64,
        )
        masked, labels, loss_mask, docs, pos = (t.to(device) for t in (masked, labels, loss_mask, docs, pos))
        mult = lr_mult(step, total, 0.05)
        for group in opt.param_groups:
            group["lr"] = lr * mult
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            loss = model(
                masked, labels=labels, doc_ids=docs, position_ids=pos, loss_mask=loss_mask, mode=context,
            ).loss
        if not torch.isfinite(loss):
            raise SystemExit("[tapt] non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += int((docs >= 0).sum())
        step += 1
        if step % 20 == 0:
            print(f"[tapt] step {step}/{total} tokens {seen} loss {float(loss):.3f}", flush=True)
    model.save_pretrained(out)
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump({"tokens": seen, "steps": step, "passes": passes}, handle)
    print(f"[tapt] saved {out}", flush=True)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dense_v1")
    parser.add_argument("--context", default="bidirectional")
    args = parser.parse_args()
    run_tapt(model_name=args.model, context=args.context)


if __name__ == "__main__":
    main()
