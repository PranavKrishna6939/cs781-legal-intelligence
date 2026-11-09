"""Write a Phase 6 zip. prediction.json sits at the archive root."""
from __future__ import annotations

import json
import os
import zipfile

from saul.summ.data import p6_dir, repeat_rate


def check_rows(rows: list[dict], expected_ids: list[str], earlier: list[list[dict]] | None = None) -> dict:
    by = {row["id"]: row.get("summary", "") for row in rows}
    missing = [i for i in expected_ids if i not in by]
    empty = [i for i in expected_ids if i in by and not str(by[i]).strip()]
    specials = [i for i in expected_ids if "<|" in str(by.get(i, ""))]
    rates = [repeat_rate(by[i]) for i in expected_ids if i in by and by[i].strip()]
    mean_rate = float(sum(rates) / len(rates)) if rates else 0.0
    signature = tuple(by.get(i, "") for i in expected_ids)
    duplicate = False
    if earlier:
        duplicate = any(tuple(row["summary"] for row in prev) == signature for prev in earlier)
    problems = []
    if missing:
        problems.append(f"missing {len(missing)}")
    if empty:
        problems.append(f"empty {len(empty)}")
    if specials:
        problems.append(f"specials {len(specials)}")
    if rates and not 0.02 <= mean_rate <= 0.07:
        problems.append(f"repeat {mean_rate:.3f}")
    if duplicate:
        problems.append("duplicate")
    words = [len(str(by[i]).split()) for i in expected_ids if i in by]
    return {
        "ok": not problems,
        "problems": problems,
        "mean_words": int(sum(words) / len(words)) if words else 0,
        "repeat": mean_rate,
        "n": len(by),
    }


def write_zip(name: str, rows: list[dict], report: dict) -> str:
    dest = os.path.join(p6_dir(), "submissions", name)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "prediction.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    zpath = os.path.join(dest, "submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, arcname="prediction.json")
    rank_path = os.path.join(p6_dir(), "rank_v2.json")
    rank = []
    if os.path.exists(rank_path):
        rank = json.load(open(rank_path, encoding="utf-8"))
    rank.append({"name": name, "zip": zpath, **{k: report[k] for k in ("ok", "problems", "mean_words", "repeat")}})
    with open(rank_path, "w", encoding="utf-8") as handle:
        json.dump(rank, handle, indent=2)
    return zpath
