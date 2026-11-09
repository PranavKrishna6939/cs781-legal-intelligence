#!/usr/bin/env python
"""Stage 10b: run the release gates for a tokenizer (and GPT-2 as a compression reference).

Writes artifacts/tokenizers/{name}/gates.json; exit code 1 if a hard gate fails (round-trip, Phase-2 span
alignment > 99.5%, digits, special ids/atomic tags, no UNK).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.tokenizer.gates import (gate_compression, gate_continuation_length, gate_digits, gate_no_unk,  # noqa: E402
                                  gate_roundtrip, gate_span_alignment, gate_specials)
from saul.tokenizer.specials import N_SPECIAL  # noqa: E402
from saul.tokenizer.train import load_raw_tokenizer  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import iter_shards, read_json, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--gpt2_reference", action="store_true", help="also report GPT-2 compression (downloads gpt2 tokenizer)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dcfg = load_config(cfg.data_config)
    name = a.name or cfg.default
    tdir = os.path.join(cfg.out_root, name)
    tok = load_raw_tokenizer(tdir)
    meta = read_json(os.path.join(tdir, "saul_tokenizer_meta.json"))
    g = cfg.gates
    res: dict = {"name": name}

    test = read_json(os.path.join(dcfg.paths.raw, dcfg.hf.test_file))
    test_texts = [v["text"] for v in test.values()]
    val_docs = [r["text"] for s in dcfg.sources
                for r in itertools.islice(iter_shards(os.path.join(dcfg.paths.final, "val", s)), g.roundtrip_docs // 4)]
    res["roundtrip"] = gate_roundtrip(tok, val_docs + test_texts)

    p2 = os.path.join(dcfg.paths.raw, "Phase_2_LNER/Train")
    docs, gold = {}, {}
    for fold in ("fold1", "fold2"):
        tp, gp = os.path.join(p2, f"{fold}-text.json"), os.path.join(p2, f"{fold}-gold.json")
        if os.path.exists(tp) and os.path.exists(gp):
            docs.update(read_json(tp))
            gold.update(read_json(gp))
    res["span_alignment"] = (gate_span_alignment(tok, docs, gold, g.span_alignment_threshold) if gold
                             else {"pass": None, "note": "Phase-2 files missing; run 00_download.py"})
    if docs:
        res["roundtrip_phase2"] = gate_roundtrip(tok, list(docs.values()))

    by_src = {s: [r["text"] for r in itertools.islice(iter_shards(os.path.join(dcfg.paths.final, "val", s)),
                                                      g.compression_docs_per_source)] for s in dcfg.sources}
    by_src["test_prompts"] = test_texts
    res["compression"] = gate_compression(tok, by_src)
    if a.gpt2_reference:
        from transformers import AutoTokenizer

        gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")._tokenizer
        res["compression_gpt2"] = gate_compression(gpt2, by_src)

    truth_p = os.path.join(dcfg.paths.final, "dev_truth.json")
    if os.path.exists(truth_p):
        res["continuation_length"] = gate_continuation_length(tok, [v["text"] for v in read_json(truth_p).values()])
    res["digits"] = gate_digits(tok, N_SPECIAL)
    res["specials"] = gate_specials(tok, meta["specials"], meta.get("anon_tags", []))
    extra = []
    for rel in ("Phase_3_RR/Train/CL_train.json", "Phase_7_PCR/Dev/dev.json"):
        p = os.path.join(dcfg.paths.raw, rel)
        if os.path.exists(p):
            data = read_json(p)
            items = data.values() if isinstance(data, dict) else data
            for it in itertools.islice(items, 50):
                txt = it.get("text") if isinstance(it, dict) else None
                if isinstance(txt, list):
                    extra.extend(txt[:50])
    res["no_unk"] = gate_no_unk(tok, extra + test_texts)

    hard = ["roundtrip", "digits", "specials", "no_unk"]
    failed = [k for k in hard if not res[k]["pass"]]
    if res["span_alignment"].get("pass") is False:
        failed.append("span_alignment")
    res["failed"] = failed
    write_json(res, os.path.join(tdir, "gates.json"))
    summary = {k: (v.get("pass") if isinstance(v, dict) else v) for k, v in res.items()}
    print(summary)
    if "span_alignment" in res and "rate" in res["span_alignment"]:
        print(f"span alignment rate {res['span_alignment']['rate']:.4%}  per-label {res['span_alignment']['per_label']}")
    print("compression:", {k: round(v["tokens_per_word"], 3) for k, v in res["compression"].items()})
    if failed:
        print(f"GATES FAILED: {failed}")
        sys.exit(1)
    print("all hard gates passed")


if __name__ == "__main__":
    main()
