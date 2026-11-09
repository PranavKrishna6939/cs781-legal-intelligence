"""Phase 6 metric, matching resources/evaluation_script/evaluate_phase6.py, plus a paired bootstrap."""
from __future__ import annotations

import importlib.util
import os

import numpy as np


def _evaluator():
    root = os.environ.get("SAUL_ROOT", os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")))
    path = os.path.join(root, "resources", "evaluation_script", "evaluate_phase6.py")
    if not os.path.exists(path):
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(here, "resources", "evaluation_script", "evaluate_phase6.py")
    spec = importlib.util.spec_from_file_location("evaluate_phase6", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_texts(preds: list[str], refs: list[str], device: str = "cuda") -> dict:
    import bert_score

    mod = _evaluator()
    n = max(len(preds), 1)
    _, _, f = bert_score.score(preds, refs, lang="en", batch_size=8, verbose=False, device=device)
    per = f.detach().float().cpu().numpy()
    scorer = mod.rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    word_tok = mod.tokenizers.DefaultTokenizer(use_stemmer=True)
    r1 = np.zeros(len(preds))
    rl = np.zeros(len(preds))
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        r1[i] = scorer.score(ref, pred)["rouge1"].fmeasure
        pt, rt = word_tok.tokenize(pred), word_tok.tokenize(ref)
        if not pt and not rt:
            rl[i] = 1.0
        elif pt and rt:
            z = mod.lcs(pt, rt)
            pp, rr = z / len(pt), z / len(rt)
            rl[i] = 0.0 if pp + rr == 0 else 2 * pp * rr / (pp + rr)
    return {
        "bert": float(per.mean()) if len(per) else 0.0,
        "rouge1": float(r1.mean()) if len(r1) else 0.0,
        "rougeL": float(rl.mean()) if len(rl) else 0.0,
        "n": len(preds),
        "per_bert": per,
        "per_rouge1": r1,
        "per_rougeL": rl,
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 781) -> dict:
    """Mean of a-b and the 95% interval. A change is kept when the gain is at least 0.002 and the interval excludes 0."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if len(diff) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "keep": False}
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    mean = float(diff.mean())
    keep = mean >= 0.002 and float(lo) > 0
    return {"mean": mean, "lo": float(lo), "hi": float(hi), "keep": bool(keep)}
