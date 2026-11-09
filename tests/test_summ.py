"""Phase 6 data, packing, and the forced length token."""
from __future__ import annotations

import torch

from saul.model.modeling_saul import build_model, tiny_config
from saul.serve.config import EngineConfig
from saul.serve.engine import LLMEngine
from saul.serve.sampling import SamplingParams
from saul.summ.data import (
    bucket_of, build_sequence, caption_cut, caption_is_safe, collate, fit_document,
    gram_index, matching_query_ids, pack_sequences, strip_caption,
)
from saul.summ.decode import render_prompt
from saul.summ.length import bucket_midpoints, fit_ridge, words_from_probs


class Tok:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 1, 2]
    all_special_tokens = ["<pad>", "<bos>", "<eos>"]
    added_tokens_decoder = {}

    def encode(self, text, add_special_tokens=False):
        return [int(part[1:]) for part in text.split()] if text.strip() else []

    def decode(self, ids, **kwargs):
        return "".join(f"w{int(i)} " for i in ids)


def test_length_buckets_cover_sixteen():
    assert bucket_of(0) == 0
    assert bucket_of(149) == 0
    assert bucket_of(150) == 1
    assert bucket_of(3300) == 15
    assert bucket_of(9000) == 15
    assert len(bucket_midpoints()) == 16


def test_caption_stops_at_the_judgment_line():
    text = "Civil Appeal No. 1. Counsel for the Appellant. The Judgment of the Court was delivered by X. The facts are these."
    assert "facts" in strip_caption(text)
    assert "Counsel" not in strip_caption(text)
    assert caption_cut(text) < 20


def test_caption_is_unsafe_when_the_summary_copies_it():
    doc = "The Judgment of the Court was delivered by X. Alpha beta gamma delta epsilon zeta."
    summary = "The Judgment of the Court was delivered by X and the bench agreed completely here."
    assert caption_is_safe([{"document": doc, "summary": summary}], n=1) is False
    safe = "The appellant was a tenant and the custodian refused possession of the house."
    assert caption_is_safe([{"document": doc, "summary": safe}], n=1) is True


def test_loss_mask_covers_length_summary_and_eos_only():
    seq, mask = build_sequence([10, 11, 12, 13], [20, 21], bos=1, summary_id=7, len_id=30, eos=2, lmax=32)
    assert seq[:5] == [1, 10, 11, 12, 13]
    assert seq[5:8] == [7, 30, 20]
    assert seq[-1] == 2
    assert mask[:6] == [0, 0, 0, 0, 0, 0]
    assert mask[6:] == [1, 1, 1, 1]
    assert len(seq) == len(mask)


def test_over_budget_keeps_head_and_tail():
    body = fit_document(list(range(100)), 10)
    assert body[:6] == list(range(6))
    assert body[6:] == list(range(96, 100))
    seq, mask = build_sequence(list(range(100)), [7, 8], 1, 3, 4, 2, lmax=20)
    assert len(seq) <= 20
    assert sum(mask) == 4  # length, two summary tokens, eos
    long_sum, _ = build_sequence(list(range(100)), list(range(100)), 1, 3, 4, 2, lmax=20)
    assert len(long_sum) <= 20


def test_packed_rows_do_not_share_a_document_id():
    a = build_sequence([10, 11], [20], 1, 7, 30, 2, 32)
    b = build_sequence([12, 13, 14], [21, 22], 1, 7, 31, 2, 32)
    packed = pack_sequences([a, b], token_budget=64)
    assert len(packed) == 1
    docs = packed[0]["doc_ids"]
    assert docs[: len(a[0])] == [0] * len(a[0])
    assert docs[len(a[0]) :] == [1] * len(b[0])
    batch = collate(packed, pad_id=0)
    # A new segment at 0 and at the second document. Nothing predicts across the boundary.
    assert list(batch["cu_seqlens"]) == [0, len(a[0]), len(a[0]) + len(b[0])]
    assert batch["position_ids"][0, len(a[0])] == 0


def test_prompt_round_trip():
    class Plain:
        def encode(self, text, add_special_tokens=False):
            return [int(x) for x in text.split()] if text else []

        def decode(self, ids, **kwargs):
            return " ".join(str(int(i)) for i in ids)

    assert render_prompt(Plain(), [4, 5, 7]) == "4 5 7"


def test_overlap_drops_a_shared_document():
    shared = " ".join(f"w{i}" for i in range(40))
    index = gram_index([shared + " tail"])
    queries = [
        {"id": "hit", "document": shared + " other"},
        {"id": "miss", "document": " ".join(f"z{i}" for i in range(40))},
    ]
    found = matching_query_ids(queries, index, min_hits=2)
    assert found == ["hit"]


def test_ridge_predicts_a_positive_length():
    rows = []
    for i in range(30):
        rows.append({
            "document": "Civil Appeal No. " + ("section " * (20 + i)) + "the facts",
            "summary": " ".join(["held"] * (40 + i)),
        })
    weights = fit_ridge(rows)
    pred = float(__import__("math").exp(__import__("numpy").concatenate([
        __import__("saul.summ.length", fromlist=["features"]).features(rows[0]["document"]),
        [1.0],
    ]) @ weights))
    assert pred > 10


def test_forced_length_token_is_first():
    torch.manual_seed(0)
    model = build_model(tiny_config("dense"), enforce_min=False).eval()
    cfg = EngineConfig(
        max_slots=2, block_size=1, max_prompt_len=16, max_new_tokens=4,
        prefill_token_budget=16, cuda_graphs=False, device="cpu", dtype="float32",
        suffix_buckets=(4,), prompt_buckets=(16,), rouge_workers=1, seed=0,
    )
    eng = LLMEngine(model, Tok(), cfg)
    params = SamplingParams(
        n=1, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
        target_words=None, max_new_tokens=3, forced_tokens=(8,),
    )
    req = eng.submit_ids([4, 5, 6], params)
    eng.run_until_complete()
    assert req.output.candidates[0].startswith("w8 ")
    probs = words_from_probs(torch.tensor([0.0, 0.0, 1.0] + [0.0] * 13))
    assert probs > 100
