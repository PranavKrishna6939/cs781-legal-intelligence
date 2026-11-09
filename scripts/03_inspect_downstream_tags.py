#!/usr/bin/env python
"""Stage 03a: find the exact anonymization placeholder strings used in the Phase 2-7 files.

IL-TUR anonymised RR/CJPE/LSI/PCR (e.g. PCR uses `<NAME>` / `<ORG>`). These strings become special tokens and
are injected into ~12% of pretraining documents (stage 03b). Writes artifacts/downstream_tags.json:

  {"tags": [...], "roles": {"name": "<NAME>", "org": "<ORG>", "statute": <tag or null>}, "per_file": {...}}

Review the output by hand; edit the JSON if a role is mis-assigned before running stage 03b / the tokenizer.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.data.stream import iter_json_strings  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import write_json  # noqa: E402

TAG_RE = re.compile(r"<[A-Z][A-Z_]*\d*>|\[[A-Z][A-Z_]*\d*\]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--min_count", type=int, default=50)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    total: Counter = Counter()
    per_file: dict = {}
    for rel in cfg.hf.downstream_files:
        p = os.path.join(cfg.paths.raw, rel)
        if not os.path.exists(p) or not p.endswith(".json"):
            continue
        c: Counter = Counter()
        for s in iter_json_strings(p):
            c.update(TAG_RE.findall(s))
        per_file[rel] = dict(c.most_common(30))
        total.update(c)
        print(f"{rel}: {c.most_common(10)}")
    tags = [t for t, n in total.most_common() if n >= a.min_count]
    # collapse numbered variants ([PERSON1], [PERSON2] ...) into a canonical family, keep the literal strings too
    roles = {"name": None, "org": None, "statute": None}
    for t in tags:
        u = t.upper()
        if roles["name"] is None and any(k in u for k in ("NAME", "PERSON")):
            roles["name"] = t
        elif roles["org"] is None and any(k in u for k in ("ORG", "ORGANIZATION", "COMPANY")):
            roles["org"] = t
        elif roles["statute"] is None and any(k in u for k in ("STAT", "PROVISION", "SECTION", "ACT")):
            roles["statute"] = t
    roles["name"] = roles["name"] or cfg.anonymize.default_tags.name
    roles["org"] = roles["org"] or cfg.anonymize.default_tags.org
    tags = list(dict.fromkeys([roles["name"], roles["org"]] + ([roles["statute"]] if roles["statute"] else []) + tags))
    # numbered placeholders ([PERSON1] ... [PERSON99]) would explode the special-token list: keep at most 40
    tags = tags[:40]
    out = {"tags": tags, "roles": roles, "per_file": per_file, "counts": dict(total.most_common(100))}
    write_json(out, cfg.anonymize.tags_file)
    print(f"-> {cfg.anonymize.tags_file}: roles={roles}, {len(tags)} tags")


if __name__ == "__main__":
    main()
