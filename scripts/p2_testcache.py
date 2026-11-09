"""Cache fold-averaged test distributions for one or more finished experiments.

    python scripts/p2_testcache.py E3v2 E4v2 E5v2
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.ner.submit import test_logps  # noqa: E402


def main() -> None:
    for exp in sys.argv[1:]:
        print(f"[testcache] {exp}", flush=True)
        test_logps(exp)
    print("[testcache] done", flush=True)


if __name__ == "__main__":
    main()
