"""Fill reports/phase4_results.md from whatever experiments have finished."""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from saul.cjpe import exp_dir, p4_dir
from saul.cjpe.calibrate import apply_prior
from saul.cjpe.score import macro_scores, predict_logits
from saul.ner.registry import repo_root

CUE = re.compile(
    r"appeal is allowed|appeals are dismissed|we dismiss the appeal|petition is allowed|"
    r"is hereby dismissed|appeal is dismissed|appeals are allowed",
    re.I,
)
MULTI = re.compile(r"\bappeals are\b|\bboth the appeals\b|\bthese appeals\b", re.I)


def _done(name: str) -> dict | None:
    path = os.path.join(exp_dir(name), "done.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def _line(name: str) -> str:
    blob = _done(name)
    if not blob:
        return f"| {name} | pending | | | | | |"
    return (
        f"| {name} | {blob.get('model', '')} | {blob.get('technique', '')} | "
        f"{blob.get('dev_f1', blob.get('dev_f1', float('nan')))} | {blob.get('trainable', '')} | "
        f"{blob.get('peak_gb', '')} | {blob.get('hours', '')} |"
    )


def _errors(limit: int = 30) -> list[str]:
    mix = os.path.join(p4_dir(), "mix", "all", "logits_dsel.npz")
    path = mix if os.path.exists(mix) else os.path.join(exp_dir("A1"), "logits_dsel.npz")
    text_path = os.path.join(p4_dir(), "text", "dsel.jsonl")
    if not os.path.exists(path) or not os.path.exists(text_path):
        return ["pending"]
    blob = np.load(path, allow_pickle=True)
    ids = [str(i) for i in blob["ids"].tolist()]
    logits = np.asarray(blob["logits"], dtype=np.float64)
    if not os.path.exists(os.path.join(os.path.dirname(path), "calib.json")):
        done_path = os.path.join(os.path.dirname(path), "done.json")
        if os.path.exists(done_path):
            done = json.load(open(done_path, encoding="utf-8"))
            logits = apply_prior(logits, done.get("loss") == "ce")
    labels = np.asarray(blob["labels"]) if "labels" in blob.files else None
    texts = {json.loads(line)["id"]: json.loads(line)["text"] for line in open(text_path, encoding="utf-8") if line.strip()}
    if labels is None:
        return ["logits have no labels"]
    pred = predict_logits(logits)
    rows = []
    for i, case_id in enumerate(ids):
        if pred[i] == labels[i]:
            continue
        text = texts.get(case_id, "")
        lo = float(logits[i, 1] - logits[i, 0])
        kind = "FP" if pred[i] == 1 else "FN"
        year = case_id.split("_", 1)[0]
        rows.append((abs(lo), kind, case_id, int(labels[i]), int(pred[i]), len(text.split()), year[:3] + "0" if year.isdigit() else "",
                     bool(CUE.search(text[-1500:])), bool(MULTI.search(text))))
    rows.sort(reverse=True)
    lines = ["id\tgold\tpred\tkind\twords\tdecade\tcue\tmulti"]
    kept = {"FP": 0, "FN": 0}
    for _score, kind, case_id, gold, got, words, decade, cue, multi in rows:
        if kept[kind] >= limit:
            continue
        kept[kind] += 1
        lines.append(f"{case_id}\t{gold}\t{got}\t{kind}\t{words}\t{decade}\t{int(cue)}\t{int(multi)}")
        if kept["FP"] >= limit and kept["FN"] >= limit:
            break
    return lines


def main() -> None:
    names = ["B0", "A1", "A1_noovl", "A1_ce", "A1probe", "A3", "A4", "A5", "A6", "A7", "A9", "A9gold", "A11", "A12"]
    table = [
        "# Phase 4 results",
        "",
        "Scores are macro-F1 on D-sel unless a cell says otherwise. Pending rows have not finished.",
        "",
        "## Models",
        "",
        "| run | model | technique | D-sel F1 | trainable | peak GB | hours |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    table.extend(_line(name) for name in names)
    sections = os.path.join(exp_dir("A3"), "sections.json")
    table.extend(["", "## Sections", ""])
    if os.path.exists(sections):
        blob = json.load(open(sections, encoding="utf-8"))
        table.append("```")
        table.append(json.dumps(blob.get("scores", {}), indent=2)[:4000])
        table.append("```")
    else:
        table.append("pending")
    table.extend(["", "## Errors", "", "Thirty false positives and thirty false negatives, highest confidence first.", ""])
    table.extend(_errors())
    path = os.path.join(repo_root(), "reports", "phase4_results.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(table) + "\n")
    err = os.path.join(repo_root(), "reports", "phase4_errors.tsv")
    lines = _errors()
    with open(err, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[analyze] {path}", flush=True)


if __name__ == "__main__":
    main()
