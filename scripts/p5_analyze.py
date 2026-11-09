"""Fill reports/phase5_results.md from whatever /data/cs781-p5 already contains."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import exp_dir, p5_dir
from saul.ner.registry import repo_root


def _read(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _runs() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(p5_dir(), "exp", "*", "done.json"))):
        blob = _read(path) or {}
        blob["exp"] = os.path.basename(os.path.dirname(path))
        rows.append(blob)
    return rows


def _fmt(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _run_map(runs: list[dict]) -> dict[str, dict]:
    return {row.get("exp"): row for row in runs}


def _rare_line(exp: str, freq: list) -> str:
    path = os.path.join(exp_dir(exp), "dev_logits.npz")
    if not os.path.exists(path) or not freq:
        return ""
    from saul.lsi.decide import load_logits, quick_macro
    from saul.lsi.data import dev_half_array
    from saul.lsi.score import rare_macro
    blob = load_logits(exp, "dev")
    if "labels" not in blob:
        return ""
    half = dev_half_array(blob["ids"])
    pred = quick_macro(blob["logits"], blob["labels"], half)["pred"]
    return f"{exp} rare-20 macro-F1 {rare_macro(blob['labels'], pred, freq, 20):.4f}"


def _piece(report: dict | None, name: str) -> dict | None:
    if not report:
        return None
    for piece in report.get("pieces") or []:
        if piece.get("piece") == name:
            return piece
    return None


def main() -> None:
    g0 = _read(os.path.join(p5_dir(), "g0.json"))
    stats = _read(os.path.join(p5_dir(), "stats.json"))
    labels = _read(os.path.join(p5_dir(), "label_stats.json"))
    runs = _runs()
    lines = [
        "# Phase 5 results",
        "",
        "Filled by `scripts/p5_analyze.py`. Empty cells mean that run has not finished.",
        "",
        "## Data",
        "",
    ]
    if g0:
        lines.append(f"G0 pass={g0.get('pass')} max label gap={g0.get('max_gap')} "
                     f"A={g0.get('n_A')} B={g0.get('n_B')} groups_intact={g0.get('groups_intact')}.")
    else:
        lines.append("G0 has not been run.")
    if stats:
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(stats, indent=2)[:2000])
        lines.append("```")
    if labels:
        freq = labels.get("freq") or []
        if freq:
            order = sorted(range(len(freq)), key=lambda i: freq[i])
            lines.append("")
            lines.append(f"Rarest labels: {[(i, int(freq[i])) for i in order[:10]]}.")
            lines.append(f"Top pairs: {labels.get('top_pairs', [])[:8]}.")
    by_exp = _run_map(runs)
    lines += ["", "## Runs", "",
              "| Exp | Model | Technique | Head | Loss | Dev macro-F1 | Micro | Subset | Hours | tok/s | Peak GB | Trainable |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        lines.append(
            f"| {run.get('exp')} | {run.get('model', '')} | {run.get('technique', '')} | {run.get('head', '')} "
            f"| {run.get('loss', '')} | {_fmt(run.get('dev_f1'))} | {_fmt(run.get('micro_f1'))} "
            f"| {_fmt(run.get('subset_accuracy'))} | {_fmt(run.get('hours'))} | {_fmt(run.get('tok_s'))} "
            f"| {_fmt(run.get('peak_gb'))} | {run.get('trainable', '')} |"
        )
    if not runs:
        lines.append("| — | | | | | | | | | | | |")
    decide = _read(os.path.join(exp_dir("decide"), "decide.json"))
    sections = _read(os.path.join(exp_dir("A4"), "sections.json")) or _read(os.path.join(exp_dir("A1"), "sections.json"))
    freq = (labels or {}).get("freq") or []
    lines += ["", "## Report sections (plan §11)", ""]

    def cell(name: str, key: str = "dev_f1") -> str:
        row = by_exp.get(name) or {}
        value = row.get(key)
        return "not run" if value is None else f"{name} {key}={_fmt(value)}"

    lines.append("1. Techniques: " + "; ".join(cell(name) for name in ("A1", "A5", "A6", "A7", "A9", "B1")) + ".")
    lines.append(f"2. PEFT vs full FT: {cell('A1')} vs {cell('A7')}.")
    p1, p2 = by_exp.get("P1"), by_exp.get("P2")
    long_doc = "P1/P2 not both finished"
    if p1 and p2 and p1.get("dev_f1") is not None and p2.get("dev_f1") is not None:
        long_doc = f"P1 LAAT {_fmt(p1['dev_f1'])} vs P2 pooled {_fmt(p2['dev_f1'])}"
    if sections:
        long_doc += (
            f"; first-chunk {_fmt(sections.get('first_chunk_macro'))} vs all chunks {_fmt(sections.get('full_macro'))}"
        )
    lines.append(f"3. Long documents: {long_doc}.")
    rare_bits = [text for text in (_rare_line(name, freq) for name in ("B1", "A4", "mix")) if text]
    lines.append("4. Rare statutes: " + ("; ".join(rare_bits) if rare_bits else "logits not ready") + ".")
    lines.append("5. Loss and sampling: " + "; ".join(cell(name) for name in ("P1", "P3", "P4", "P5", "P6")) + ".")
    d1, d2 = _piece(decide, "D1"), _piece(decide, "D2")
    if d1:
        thresh = (
            f"D1 kappa={d1.get('kappa')} cross={_fmt(d1.get('cross'))} in-sample={_fmt(d1.get('insample'))} "
            f"gain={_fmt(d1.get('gain'))}"
        )
    else:
        thresh = "decide.json not ready"
    if d2:
        thresh += f"; D2 mode={d2.get('mode')} gain={_fmt(d2.get('gain'))}"
    lines.append(f"6. Thresholds: {thresh}.")
    d3 = _piece(decide, "D3")
    lines.append(
        "7. Label relationships: "
        + (f"D3 gain={_fmt(d3.get('gain'))} keep={d3.get('keep')}" if d3 else "D3 not fit")
        + "."
    )
    pred_path = os.path.join(exp_dir("decide"), "dev_pred.npz")
    if os.path.exists(pred_path) and os.path.exists(os.path.join(exp_dir("mix"), "dev_logits.npz")):
        from saul.lsi.decide import load_logits
        gold_blob = load_logits("mix", "dev")
        pred_blob = np.load(pred_path, allow_pickle=True)
        index = {str(i): row for row, i in enumerate(pred_blob["ids"])}
        if any(i not in index for i in gold_blob["ids"]):
            lines.append("8. Confusions: dev prediction ids do not match the mix.")
            gold = None
            pred = None
        else:
            gold = gold_blob["labels"]
            pred = pred_blob["pred"][[index[i] for i in gold_blob["ids"]]]
    else:
        gold = None
        pred = None
    if gold is not None and pred is not None:
        swaps = []
        gold_b = gold > 0
        pred_b = pred > 0
        for label_j in range(gold_b.shape[1]):
            mask = gold_b[:, label_j]
            count = int(mask.sum())
            if count == 0:
                continue
            rate = pred_b[mask].mean(0)
            rate[label_j] = 0
            label_i = int(rate.argmax())
            swaps.append((float(rate[label_i]), label_j, label_i))
        swaps.sort(reverse=True)
        shown = ", ".join(f"{j}->{i} ({rate:.2f})" for rate, j, i in swaps[:8])
        lines.append(f"8. Confusions (false positive of i given gold j): {shown}.")
    elif not any(line.startswith("8.") for line in lines):
        lines.append("8. Confusions: dev predictions are not ready.")
    if sections and sections.get("decile_mass"):
        mass = ", ".join(f"{value:.3f}" for value in sections["decile_mass"])
        lines.append(f"9. Sections: LAAT mass by decile [{mass}].")
    else:
        lines.append("9. Sections: sections.json is not ready.")
    teacher = _read(os.path.join(p5_dir(), "kd", "teacher.json"))
    kd_line = cell("A9") + "; " + cell("A9gold")
    if teacher:
        kd_line += f"; teacher members={teacher.get('members')}"
    rare_kd = _rare_line("A9", freq)
    if rare_kd:
        kd_line += f"; {rare_kd}"
    lines.append(f"10. KD: {kd_line}.")
    a1 = by_exp.get("A1") or {}
    a1t = by_exp.get("A1t") or {}
    tapt_gain = ""
    if a1.get("dev_f1") is not None and a1t.get("dev_f1") is not None:
        tapt_gain = f" TAPT gain={a1t['dev_f1'] - a1['dev_f1']:+.4f}."
    lines.append(
        f"11. Phase-1 features: {cell('A1probe')} vs {cell('A1')}.{tapt_gain}"
    )
    lines.append("12. Data facts: the data section above. The label-id to IPC-section mapping is not available.")
    lines.append("")
    path = os.path.join(repo_root(), "reports", "phase5_results.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[analyze] {path} runs={len(runs)}", flush=True)


if __name__ == "__main__":
    main()
