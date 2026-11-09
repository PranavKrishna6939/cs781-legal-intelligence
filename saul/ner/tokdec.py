"""Token-level document decoder.

Input per document: averaged BIOES distributions over its tokens (windows, folds, or
models averaged in probability space). Steps:

1. Viterbi over log marginals with legal BIOES transitions. An entity may only begin on
   a token that starts a word and only end on a token that ends a word (soft penalty).
2. Per-label confidence threshold on the mean label mass of each span.
3. Reconciliation: one label per surface string per document (summed confidence).
4. Propagation: an accepted string is copied to its uncovered whole-word repeats.
5. Lexicon: strings that the training documents label consistently are added where
   nothing else covers them.
"""
from __future__ import annotations

import collections
import json
import re

import numpy as np

from saul.ner.labels import LABELS, N_LABELS, N_TAGS, legal_transition_masks, spans_from_tags
from saul.ner.metrics import strict_scores

NEG = -1.0e4
_OK, _START, _END = (m.numpy() for m in legal_transition_masks())
_TRANS = np.where(_OK, 0.0, NEG)
_BEGIN_TAGS = np.array([t for t in range(1, N_TAGS) if (t - 1) % 4 in (0, 3)])  # B, S
_END_TAGS = np.array([t for t in range(1, N_TAGS) if (t - 1) % 4 in (2, 3)])  # E, S


def default_tok_config() -> dict:
    return {
        "penalty_begin": 1e4,
        "penalty_end": 2.0,
        "bias": {lab: 0.0 for lab in LABELS},
        "tau": {lab: 0.0 for lab in LABELS},
        "reconcile": True,
        "propagate": {lab: True for lab in LABELS},
        "prop_min_len": 3,
        "lexicon": {lab: False for lab in LABELS},
        "trim": {"punct": False, "honorific": False, "dated": False},
    }


def _with_bias(logp: np.ndarray, bias: dict | None) -> np.ndarray:
    """Add a per-label constant to that label's B/I/E/S log-marginals (plan C2)."""
    if not bias or not any(bias.values()):
        return logp
    out = np.array(logp, dtype=np.float64, copy=True)
    for lab, value in bias.items():
        if not value or lab not in LABELS:
            continue
        i = LABELS.index(lab)
        out[:, 1 + i * 4:1 + (i + 1) * 4] += float(value)
    return out


def _bias_key(bias: dict | None) -> tuple:
    if not bias:
        return tuple((lab, 0.0) for lab in LABELS)
    return tuple((lab, float(bias.get(lab, 0.0))) for lab in LABELS)


def word_edges(text: str, starts, ends) -> tuple[np.ndarray, np.ndarray]:
    n = len(starts)
    begin = np.ones(n, dtype=bool)
    end = np.ones(n, dtype=bool)
    for i in range(n):
        s, e = int(starts[i]), int(ends[i])
        begin[i] = s == 0 or not text[s - 1].isalnum()
        end[i] = e >= len(text) or not text[e].isalnum()
    return begin, end


def viterbi(logp: np.ndarray, begin: np.ndarray, end: np.ndarray, pen_b: float, pen_e: float) -> list[int]:
    n, k = logp.shape
    if n == 0:
        return []
    emit = logp.astype(np.float64).copy()
    if pen_b:
        emit[np.ix_(~begin, _BEGIN_TAGS)] -= pen_b
    if pen_e:
        emit[np.ix_(~end, _END_TAGS)] -= pen_e
    score = np.where(_START, 0.0, NEG) + emit[0]
    back = np.zeros((n, k), dtype=np.int16)
    for t in range(1, n):
        cand = score[:, None] + _TRANS
        back[t] = cand.argmax(0)
        score = cand[back[t], np.arange(k)] + emit[t]
    score = score + np.where(_END, 0.0, NEG)
    tag = int(score.argmax())
    path = [tag]
    for t in range(n - 1, 0, -1):
        tag = int(back[t, tag])
        path.append(tag)
    return path[::-1]


def label_mass(logp: np.ndarray) -> np.ndarray:
    """[n, 12] probability that a token belongs to each label (any of B/I/E/S)."""
    p = np.exp(logp.astype(np.float64))
    return p[:, 1:].reshape(p.shape[0], N_LABELS, 4).sum(-1)


def raw_spans(text: str, starts, ends, logp: np.ndarray, pen_b: float, pen_e: float,
              bias: dict | None = None) -> list[tuple]:
    """Viterbi spans as (char start, char end, label, confidence)."""
    if logp.shape[0] == 0:
        return []
    biased = _with_bias(logp, bias)
    begin, end = word_edges(text, starts, ends)
    path = viterbi(biased, begin, end, pen_b, pen_e)
    mass = label_mass(logp)
    start_at = {int(s): i for i, s in enumerate(starts)}
    end_at = {int(e): i for i, e in enumerate(ends)}
    out = []
    for s, e, lab in spans_from_tags(path, list(starts), list(ends)):
        i, j = start_at.get(s), end_at.get(e)
        if i is None or j is None:
            continue
        li = LABELS.index(lab)
        out.append((s, e, lab, float(mass[i:j + 1, li].mean())))
    return out


def _bounded(text: str, s: int, e: int) -> bool:
    return (s == 0 or not text[s - 1].isalnum()) and (e >= len(text) or not text[e].isalnum())


def _place(text: str, frag: str, lab: str, occupied: np.ndarray, out: list) -> None:
    start = 0
    while True:
        i = text.find(frag, start)
        if i < 0:
            return
        j = i + len(frag)
        if _bounded(text, i, j) and not occupied[i:j].any():
            occupied[i:j] = True
            out.append((i, j, lab))
        start = i + 1


PERSONS = {"APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT"}
HONORIFIC = re.compile(
    r"^(?:(?:Hon'?ble|Honourable|Justice|Mr|Mrs|Ms|Miss|Dr|Shri|Smt|Sri|Kumari|Km|Sardar|Advocate)\b\.?\s*)+",
    re.IGNORECASE,
)


def trim(text: str, s: int, e: int, lab: str, rules: dict) -> tuple[int, int]:
    if rules.get("punct"):
        while s < e and not (text[s].isalnum() or text[s] in "([\"'`"):
            s += 1
    if rules.get("honorific") and lab in PERSONS:
        m = HONORIFIC.match(text[s:e])
        if m and m.end() < e - s:
            s += m.end()
    if rules.get("dated") and lab == "DATE" and text[s:e].lower().startswith("dated "):
        s += 6
    return s, e


def finish(text: str, spans: list[tuple], cfg: dict, lexicon: dict[str, str] | None = None) -> list[tuple]:
    tau = cfg["tau"]
    kept = [sp for sp in spans if sp[3] >= float(tau.get(sp[2], 0.0))]
    rules = cfg.get("trim", {})
    if any(rules.values()):
        kept = [(*trim(text, s, e, lab, rules), lab, c) for s, e, lab, c in kept]
        kept = [sp for sp in kept if sp[1] > sp[0]]
    if cfg.get("reconcile", True):
        votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for s, e, lab, c in kept:
            votes[text[s:e]][lab] += c
        kept = [(s, e, votes[text[s:e]].most_common(1)[0][0], c) for s, e, lab, c in kept]
    occupied = np.zeros(len(text) + 1, dtype=bool)
    out = []
    for s, e, lab, _ in kept:
        if occupied[s:e].any():
            continue
        occupied[s:e] = True
        out.append((s, e, lab))
    prop = cfg.get("propagate", {})
    min_len = int(cfg.get("prop_min_len", 3))
    strings = {}
    for s, e, lab in out:
        frag = text[s:e]
        if prop.get(lab, False) and len(frag) >= min_len and not frag.isdigit():
            strings.setdefault(frag, lab)
    for frag in sorted(strings, key=len, reverse=True):
        _place(text, frag, strings[frag], occupied, out)
    if lexicon:
        use = cfg.get("lexicon", {})
        for frag in sorted(lexicon, key=len, reverse=True):
            lab = lexicon[frag]
            if use.get(lab, False):
                _place(text, frag, lab, occupied, out)
    return sorted(out)


def build_lexicon(docs, min_count: int = 2, min_docs: int = 2, purity: float = 0.9, min_len: int = 3) -> dict[str, str]:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    in_docs: dict[str, set] = collections.defaultdict(set)
    for d in docs:
        for s, e, lab in d.gold:
            frag = d.text[s:e]
            counts[frag][lab] += 1
            in_docs[frag].add(d.doc_id)
    lex = {}
    for frag, c in counts.items():
        lab, n = c.most_common(1)[0]
        total = sum(c.values())
        if len(frag) < min_len or frag.isdigit() or total < min_count or len(in_docs[frag]) < min_docs:
            continue
        if n / total < purity:
            continue
        lex[frag] = lab
    # A lexicon string must also be mostly tagged where it occurs, or it is a common word.
    hits = collections.Counter()
    seen = collections.Counter()
    for d in docs:
        gold = {(s, e) for s, e, _ in d.gold}
        for frag in lex:
            start = 0
            while True:
                i = d.text.find(frag, start)
                if i < 0:
                    break
                j = i + len(frag)
                if _bounded(d.text, i, j):
                    seen[frag] += 1
                    hits[frag] += (i, j) in gold
                start = i + 1
    return {f: l for f, l in lex.items() if seen[f] and hits[f] / seen[f] >= 0.6}


def oof_lexicons(docs, folds: list[list[str]], **kw) -> dict[str, dict[str, str]]:
    """doc_id -> lexicon built without that document's fold."""
    by_id = {d.doc_id: d for d in docs}
    out = {}
    for fold in folds:
        rest = [by_id[i] for f in folds if f is not fold for i in f if i in by_id]
        lex = build_lexicon(rest, **kw)
        for did in fold:
            out[did] = lex
    return out


class Decoder:
    """Holds per-document inputs and caches the Viterbi pass per penalty pair."""

    def __init__(self, docs, logps: dict[str, np.ndarray], lexicons: dict[str, dict] | None = None):
        self.docs = [d for d in docs if d.doc_id in logps]
        self.logps = logps
        self.lexicons = lexicons or {}
        self._cache: dict[tuple, dict] = {}

    def spans(self, pen_b: float, pen_e: float, bias: dict | None = None) -> dict[str, list]:
        key = (float(pen_b), float(pen_e), _bias_key(bias))
        if key not in self._cache:
            self._cache[key] = {
                d.doc_id: raw_spans(
                    d.text, d.starts, d.ends, self.logps[d.doc_id], pen_b, pen_e, bias,
                )
                for d in self.docs
            }
        return self._cache[key]

    def predict(self, cfg: dict) -> dict[str, list]:
        raw = self.spans(cfg["penalty_begin"], cfg["penalty_end"], cfg.get("bias"))
        return {d.doc_id: finish(d.text, raw[d.doc_id], cfg, self.lexicons.get(d.doc_id)) for d in self.docs}

    def score(self, cfg: dict) -> dict:
        gold = {d.doc_id: list(d.gold) for d in self.docs}
        return strict_scores(gold, self.predict(cfg))

    def tune(self, cfg: dict | None = None, verbose: bool = False) -> dict:
        cfg = json.loads(json.dumps(cfg or default_tok_config()))

        def f(c):
            return self.score(c)["macro_f1"]

        best = f(cfg)
        for pb, pe in [(1e4, 0.0), (1e4, 1.0), (1e4, 2.0), (1e4, 4.0), (1e4, 1e4), (3.0, 2.0), (0.0, 0.0)]:
            trial = {**cfg, "penalty_begin": pb, "penalty_end": pe}
            v = f(trial)
            if v > best + 1e-6:
                best, cfg = v, trial
        cfg.setdefault("bias", {lab: 0.0 for lab in LABELS})
        for lab in LABELS:
            for value in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
                trial = json.loads(json.dumps(cfg))
                trial["bias"][lab] = value
                v = f(trial)
                if v > best + 1e-6:
                    best, cfg = v, trial
        for flag in (False, True):
            trial = {**cfg, "reconcile": flag}
            v = f(trial)
            if v > best + 1e-6:
                best, cfg = v, trial
        for rule in ("punct", "honorific", "dated"):
            trial = json.loads(json.dumps(cfg))
            trial.setdefault("trim", {})[rule] = not trial.get("trim", {}).get(rule, False)
            v = f(trial)
            if v > best + 1e-6:
                best, cfg = v, trial
        for lab in LABELS:
            for key, values in (("propagate", (False, True)), ("lexicon", (False, True))):
                for val in values:
                    trial = json.loads(json.dumps(cfg))
                    trial[key][lab] = val
                    v = f(trial)
                    if v > best + 1e-6:
                        best, cfg = v, trial
        for ml in (2, 3, 4, 6):
            trial = {**cfg, "prop_min_len": ml}
            v = f(trial)
            if v > best + 1e-6:
                best, cfg = v, trial
        grid = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for _ in range(2):
            for lab in LABELS:
                for t in grid:
                    trial = json.loads(json.dumps(cfg))
                    trial["tau"][lab] = t
                    v = f(trial)
                    if v > best + 1e-6:
                        best, cfg = v, trial
        cfg["oof_f1"] = best
        if verbose:
            print(f"[tokdec] tuned oof_f1={best:.4f}", flush=True)
        return cfg


def load_run_logps(run_dirs: list[str]) -> dict[str, np.ndarray]:
    """doc_id -> tag log-probs from the per-document npz files of one or more fold dirs."""
    import os
    out = {}
    for run in run_dirs:
        if not os.path.isdir(run):
            continue
        for name in os.listdir(run):
            if name.endswith(".npz"):
                out[name[:-4]] = np.load(os.path.join(run, name))["tag_logprobs"].astype(np.float32)
    return out


def average_logps(sets: list[dict[str, np.ndarray]], weights=None) -> dict[str, np.ndarray]:
    weights = np.ones(len(sets)) / len(sets) if weights is None else np.asarray(weights, dtype=np.float64)
    docs = set.intersection(*[set(s) for s in sets])
    out = {}
    for did in docs:
        p = sum(w * np.exp(s[did].astype(np.float64)) for w, s in zip(weights, sets) if w > 0)
        p = p / max(float(np.sum(weights)), 1e-8)
        out[did] = np.log(np.maximum(p, 1e-8)).astype(np.float32)
    return out
