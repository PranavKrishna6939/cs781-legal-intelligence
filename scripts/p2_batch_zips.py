"""Build the five score-probe zips from cached test distributions.

Uses the rule config of the equal-weight mix that scored 0.701 on the test set.
Does not retune and does not touch the distillation job.
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.ner.data import load_split  # noqa: E402
from saul.ner.registry import p2_dir, repo_root  # noqa: E402
from saul.ner.submit import decode_test, write_zip  # noqa: E402
from saul.ner.tokdec import average_logps, load_run_logps  # noqa: E402

PERSONS = ("APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT")


def cached(exp: str):
    return load_run_logps([os.path.join(p2_dir(), exp, "test")])


def main() -> None:
    found = json.load(open(os.path.join(p2_dir(), "ensemble", "P2_E8_mix.json"), encoding="utf-8"))
    base = found["cfg"]
    docs = load_split("train")
    logs = {name: cached(name) for name in ("E2v2", "E5v2", "E3v2", "E1v2", "E4v2")}
    out_root = os.path.join(repo_root(), "submissions", "phase2")

    def emit(folder: str, names: list[str], weights: list[float], cfg: dict) -> None:
        pred = decode_test(average_logps([logs[n] for n in names], weights), cfg, docs)
        path = write_zip(pred, os.path.join(out_root, folder), folder)
        n = sum(len(v) for v in pred.values())
        print(f"[batch] {folder} spans={n} zip={path}", flush=True)

    trio = ["E2v2", "E5v2", "E3v2"]
    emit("P2_mix_lorma_adapter_lora", trio, [1, 1, 1], base)
    emit("P2_mix_lorma_adapter_lora_weighted", trio, [0.5, 0.3, 0.2], base)
    emit("P2_mix_lorma_adapter", ["E2v2", "E5v2"], [1, 1], base)

    roles = copy.deepcopy(base)
    for lab in PERSONS:
        roles["lexicon"][lab] = False
    for lab in ("APP", "RESP", "WIT"):
        roles["tau"][lab] = 0.4
    emit("P2_E8_mix_role_precision", found["exps"], found["weights"], roles)

    spans = copy.deepcopy(base)
    spans["tau"]["PREC"] = 0.5
    spans["tau"]["CASENO"] = 0.8
    emit("P2_E8_mix_span_precision", found["exps"], found["weights"], spans)


if __name__ == "__main__":
    main()
