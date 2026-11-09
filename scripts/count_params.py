#!/usr/bin/env python
"""Print parameter counts for every full model config. Fails if any model is outside [50M, 299M]."""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.model.factory import config_from_yaml  # noqa: E402
from saul.model.modeling_saul import build_model  # noqa: E402
from saul.utils.params import count_params, format_breakdown  # noqa: E402


def main() -> int:
    paths = sorted(glob.glob("configs/models/*.yaml"))
    bad = False
    for path in paths:
        cfg = config_from_yaml(path)
        # proxy configs are allowed under 50M only if the filename contains 55m and the count is still < 299M.
        # Full configs must clear 50M. We enforce the cap for all of them.
        enforce_min = "55m" not in os.path.basename(path)
        try:
            model = build_model(cfg, enforce_budget=True, enforce_min=enforce_min)
        except AssertionError as e:
            print(f"FAIL {path}: {e}")
            bad = True
            continue
        n = count_params(model)
        print(f"{path}: {n:,} ({n/1e6:.2f}M) arch={cfg.arch} layers={cfg.n_layers}")
        if os.environ.get("SAUL_LOG_PARAMS"):
            print(format_breakdown(model))
        del model
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
