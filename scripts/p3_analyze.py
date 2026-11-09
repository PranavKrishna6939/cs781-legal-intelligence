"""Fill reports/phase3_results.md from done.json files and cached OOF marginals."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from saul.ner.registry import repo_root
from saul.rr.data import LABEL_ID, LABELS, labeled_docs, load_docs, p3_dir
from saul.rr.decode import load_split_probs
from saul.rr.score import confusion_pair, macro_scores

PAIRS = [
    ("ArgumentPetitioner", "ArgumentRespondent"),
    ("PrecedentReliedUpon", "PrecedentNotReliedUpon"),
    ("RatioOfTheDecision", "RulingByPresentCourt"),
    ("Fact", "RulingByLowerCourt"),
    ("Issue", "Fact"),
]


def _rows() -> list[str]:
    root = p3_dir()
    lines = []
    if not os.path.isdir(root):
        return lines
    for exp in sorted(os.listdir(root)):
        scores, peaks, minutes, rates, trainable = [], [], [], [], []
        for k in range(5):
            path = os.path.join(root, exp, f"fold{k}", "done.json")
            if not os.path.exists(path):
                path = os.path.join(root, exp, "full", "done.json") if k == 0 else ""
            if not path or not os.path.exists(path):
                continue
            blob = json.load(open(path, encoding="utf-8"))
            if blob.get("dev_f1") is not None:
                scores.append(blob["dev_f1"])
            peaks.append(blob.get("peak_gb") or 0)
            minutes.append(blob.get("minutes") or 0)
            rates.append(blob.get("sentences_per_s") or 0)
            trainable.append(blob.get("trainable") or 0)
        if not scores and not peaks:
            continue
        oof = ""
        table = load_split_probs(exp, "oof")
        if table:
            docs = {d.doc_id: d for d in labeled_docs(load_docs())}
            gold, pred = [], []
            for doc_id, probs in table.items():
                doc = docs.get(doc_id)
                if doc is None or doc.labels is None:
                    continue
                gold.extend(doc.label_ids)
                pred.extend(probs.argmax(1).tolist())
            if gold:
                oof = f"{macro_scores(gold, pred)['macro_f1']:.4f}"
        mean = f"{float(np.mean(scores)):.4f}" if scores else ""
        lines.append(
            f"| {exp} | {oof} | {mean} | {int(np.mean(trainable)) if trainable else ''} | "
            f"{float(np.mean(peaks)):.1f} | {float(np.sum(minutes)):.1f} | {float(np.mean(rates)):.1f} |"
        )
    return lines


def _confusion() -> list[str]:
    lines = []
    root = p3_dir()
    if not os.path.isdir(root):
        return lines
    docs = {d.doc_id: d for d in labeled_docs(load_docs())}
    for exp in ("E1", "E0", "E2", "E3", "E6"):
        table = load_split_probs(exp, "oof")
        if not table:
            continue
        gold, pred = [], []
        dom_g = {"CL": [], "IT": []}
        dom_p = {"CL": [], "IT": []}
        for doc_id, probs in table.items():
            doc = docs.get(doc_id)
            if doc is None or doc.labels is None:
                continue
            ids = probs.argmax(1).tolist()
            gold.extend(doc.label_ids)
            pred.extend(ids)
            dom_g[doc.domain].extend(doc.label_ids)
            dom_p[doc.domain].extend(ids)
        if not gold:
            continue
        lines.append(f"### {exp}")
        for dom in ("CL", "IT"):
            if dom_g[dom]:
                lines.append(f"- {dom} macro-F1 {macro_scores(dom_g[dom], dom_p[dom])['macro_f1']:.4f}")
        per = macro_scores(gold, pred)["per_label"]
        lines.append("- per-label: " + ", ".join(f"{LABELS[i]} {per[i]:.3f}" for i in range(len(LABELS))))
        for a, b in PAIRS:
            stats = confusion_pair(gold, pred, LABEL_ID[a], LABEL_ID[b])
            lines.append(f"- {a} vs {b}: gold {stats['gold_a']}, swapped {stats['a_as_b']}/{stats['b_as_a']}")
    return lines


def main() -> None:
    path = os.path.join(repo_root(), "reports", "phase3_results.md")
    text = open(path, encoding="utf-8").read()
    marker = "<!-- generated -->"
    head = text.split(marker)[0].rstrip() + "\n\n" + marker + "\n\n"
    body = ["| Experiment | OOF argmax | Mean fold dev | Trainable | Peak GB | Minutes | Sent/s |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    body.extend(_rows() or ["| — | | | | | | |"])
    body.append("")
    extra = _confusion()
    if extra:
        body.append("## Confusion and domains")
        body.append("")
        body.extend(extra)
        body.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(body) + "\n")
    print(f"[analyze] {path}")


if __name__ == "__main__":
    main()
