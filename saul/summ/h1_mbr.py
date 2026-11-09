"""MBR decode of the hybrid LoRMA run that scored 0.8507.

Same checkpoint, same first-2048/last-2048 prompt, same 1,200-word stop.
The only change from that submission is minimum-Bayes-risk selection over 16 samples.
"""
from __future__ import annotations

import os

import torch

from saul.ner.peft import LoRMAPlus, wrap_lorma
from saul.ner.registry import resolve_checkpoint
from saul.summ.data import load_rows
from saul.summ.run import _body, _load
from saul.summ.submit import write_zip, check_rows
from saul.tokenizer.train import load_hf_tokenizer


def _prompt_ids(tok, document: str, sep_id: int, summary_id: int) -> list[int]:
    return _body(tok, document, "ends", sep_id) + [summary_id]


def main() -> None:
    from saul.model.modeling_saul import SaulForCausalLM
    from saul.serve.api import LLM
    from saul.serve.sampling import MBRParams, SamplingParams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_hf_tokenizer(os.path.expanduser("~/cs781/artifacts/tokenizers/legal32k"))
    bos = int(tok.bos_token_id)
    sep_id = int(tok.convert_tokens_to_ids("<|sep|>"))
    summary_id = int(tok.convert_tokens_to_ids("<|summary|>"))
    overlay = os.environ.get("H1_OVERLAY", "/data/cs781-p6/exp/H1/overlay.pt")
    target = int(os.environ.get("H1_WORDS", "1200"))
    n = int(os.environ.get("H1_N", "16"))
    slots = int(os.environ.get("H1_SLOTS", "32"))

    model = SaulForCausalLM.from_pretrained(resolve_checkpoint("hybrid_b"))
    wrap_lorma(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRMAPlus):
            for param in list(module.A.parameters()) + list(module.B.parameters()):
                param.requires_grad_(True)
    _load(model, overlay)
    model.to(device).eval()
    if device == "cuda":
        model.to(dtype=torch.bfloat16)
        torch.set_float32_matmul_precision("high")

    rows = load_rows("test")
    prompts = [_prompt_ids(tok, row["document"], sep_id, summary_id) for row in rows]
    max_new = target * 2
    block = n
    slots = max(block, slots - (slots % block))
    max_prompt = max(len(p) for p in prompts) + 1
    llm = LLM(
        model, tok,
        max_slots=slots, block_size=block, max_prompt_len=max_prompt, max_new_tokens=max_new,
        prefill_token_budget=max(max_prompt, 8192), cuda_graphs=device == "cuda",
        device=device, dtype="bfloat16" if device == "cuda" else "float32", rouge_workers=4,
    )
    params = SamplingParams(
        n=n, temperature=0.7, truncation="min_p", min_p=0.05,
        no_repeat_ngram=4, max_ngram_occurrences=1, frequency_penalty=0.3, frequency_window=64,
        target_words=target, max_new_tokens=max_new,
        mbr=MBRParams(enabled=True, utility="mixed", alpha=0.6),
    )
    texts = []
    try:
        for start in range(0, len(prompts), 32):
            batch = prompts[start:start + 32]
            ids = [[bos, *row] for row in batch]
            reqs = [llm.engine.submit_ids(row, params) for row in ids]
            llm.engine.run_until_complete()
            for req in reqs:
                if req.error is not None:
                    raise req.error
                if req.output is None:
                    raise RuntimeError("no output")
                texts.append(req.output.text if req.output.text.strip() else " ")
            print(f"[h1] decoded {len(texts)}/{len(prompts)}", flush=True)
    finally:
        llm.close()

    name = f"H1_mbr{n}"
    pred = [{"id": row["id"], "summary": text} for row, text in zip(rows, texts)]
    report = check_rows(pred, [row["id"] for row in rows])
    path = write_zip(name, pred, report)
    print(f"[h1] {name} {report} {path}", flush=True)


if __name__ == "__main__":
    main()
