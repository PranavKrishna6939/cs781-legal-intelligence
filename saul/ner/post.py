"""tokdec, then the role resolver, then the boundary rules. One path for OOF and test."""
from __future__ import annotations

import json

import numpy as np

from saul.ner.header import counsel_cues, parse_header, witness_cues
from saul.ner.labels import LABELS
from saul.ner.metrics import strict_scores
from saul.ner.roles import (
    ROLE_LABELS,
    apply_decisions,
    apply_rule_fallback,
    document_rows,
    fit_classifier,
    predict_proba,
    string_priors,
)
from saul.ner.rules import apply_rules
from saul.ner.tokdec import Decoder, build_lexicon, default_tok_config, finish, oof_lexicons


def default_post_config() -> dict:
    cfg = default_tok_config()
    cfg.update({
        "p1_strip_ms": False,
        "p2_rules": True,
        "use_p2_model": False,
        "classifier": "rules",
        "theta_place": {lab: 0.5 for lab in ROLE_LABELS},
        "theta_drop": 1.01,
        "p2_place_header": {lab: False for lab in ROLE_LABELS},
        "p3_caseno": False,
        "p3_prec_split": False,
        "p3_prec_cite": False,
        "p3_prec_short": False,
        "p3_auth_gov": False,
        "p3_auth_cand": False,
        "theta_auth_cand": 0.15,
        "p3_court_split": False,
        "p3_junk": False,
    })
    return cfg


def post_process(doc, raw: list[tuple], logp, cfg: dict, bundle: dict | None = None,
                 lexicon: dict | None = None) -> list[tuple]:
    """raw is the Viterbi span list (with confidence). Returns character spans."""
    spans = finish(doc.text, raw, cfg, lexicon)
    header = parse_header(doc.text, strip_ms=bool(cfg.get("p1_strip_ms")))
    wits = witness_cues(doc.text)
    cues = counsel_cues(doc.text)
    bundle = bundle or {}
    cached = (bundle.get("oof") or {}).get(doc.doc_id)
    if cfg.get("use_p2_model") and cached is not None:
        entities, classes, proba = cached
        spans = apply_decisions(doc, spans, entities, classes, proba, cfg, header.end)
    elif cfg.get("use_p2_model") and bundle.get("clf") is not None:
        spans = _apply_fitted(doc, spans, logp, cfg, bundle, header, wits, cues)
    elif cfg.get("p2_rules", True):
        spans = apply_rule_fallback(doc, spans, header.parties, wits, header.end, cfg)
    return apply_rules(doc, spans, cfg, logp, lexicon)


def _prior_for(bundle: dict, doc_id: str) -> dict:
    """Surface priors for one document.

    Training rows store a fold-excluded map on `priors_by_doc`. Test documents and a
    held-out fold are not keys of that map; they use `priors` (every document the
    classifier was fit on). Older bundles copied that full map onto each training id.
    """
    by_doc = bundle.get("priors_by_doc") or {}
    if doc_id in by_doc:
        return by_doc[doc_id]
    if bundle.get("priors"):
        return bundle["priors"]
    if by_doc:
        return next(iter(by_doc.values()))
    return {}


def _apply_fitted(doc, spans, logp, cfg, bundle, header, wits, cues):
    prior_map = _prior_for(bundle, doc.doc_id)
    entities, rows, _ = document_rows(
        doc, spans, logp, header.parties, wits, cues, header.end, prior_map,
    )
    if not rows:
        return spans
    classes, proba = predict_proba(bundle["clf"], np.stack(rows))
    return apply_decisions(doc, spans, entities, classes, proba, cfg, header.end)


class Stack:
    """Scores a config. Viterbi is cached; role features are cached once tokdec is frozen."""

    def __init__(self, docs, logps, lexicons, fold_of: dict[str, int] | None = None):
        self.docs = [d for d in docs if d.doc_id in logps]
        self.logps = logps
        self.lexicons = lexicons or {}
        self.fold_of = fold_of or {}
        self.dec = Decoder(self.docs, logps, lexicons)
        self._oof = None
        self._oof_key = None

    def predict(self, cfg: dict, bundle: dict | None = None) -> dict[str, list]:
        raw = self.dec.spans(cfg["penalty_begin"], cfg["penalty_end"], cfg.get("bias"))
        bundle = bundle or {}
        if self._oof is not None and cfg.get("use_p2_model"):
            bundle = {**bundle, "oof": self._oof}
        return {
            d.doc_id: post_process(d, raw[d.doc_id], self.logps[d.doc_id], cfg, bundle, self.lexicons.get(d.doc_id))
            for d in self.docs
        }

    def score(self, cfg: dict, bundle: dict | None = None) -> dict:
        gold = {d.doc_id: list(d.gold) for d in self.docs}
        return strict_scores(gold, self.predict(cfg, bundle))

    def bind_oof(self, cfg: dict, kind: str) -> str | None:
        """Fit leave-one-fold classifiers and cache their probabilities. Returns the kind, or None."""
        key = _oof_cache_key(cfg, kind)
        if self._oof_key == key:
            return kind
        raw = self.dec.spans(cfg["penalty_begin"], cfg["penalty_end"], cfg.get("bias"))
        finished = {
            d.doc_id: finish(d.text, raw[d.doc_id], cfg, self.lexicons.get(d.doc_id)) for d in self.docs
        }
        priors = string_priors(self.docs, self.fold_of) if self.fold_of else {}
        rows = {}
        ys = {}
        entities = {}
        for d in self.docs:
            header = parse_header(d.text, strip_ms=bool(cfg.get("p1_strip_ms")))
            fold = self.fold_of.get(d.doc_id, -1)
            ent, x, y = document_rows(
                d, finished[d.doc_id], self.logps[d.doc_id], header.parties,
                witness_cues(d.text), counsel_cues(d.text), header.end, priors.get(fold, {}),
            )
            entities[d.doc_id] = ent
            rows[d.doc_id] = x
            ys[d.doc_id] = y
        by_fold: dict[int, list[str]] = {}
        for d in self.docs:
            by_fold.setdefault(self.fold_of.get(d.doc_id, 0), []).append(d.doc_id)
        oof = {}
        try:
            for held, ids in by_fold.items():
                train_ids = [i for f, group in by_fold.items() if f != held for i in group]
                x, y = _stack_xy(train_ids, rows, ys)
                if x is None or len(set(y)) < 2:
                    return None
                clf = fit_classifier(kind, x, y)
                for did in ids:
                    if not rows[did]:
                        oof[did] = (entities[did], [], np.zeros((0, 1)))
                        continue
                    classes, proba = predict_proba(clf, np.stack(rows[did]))
                    oof[did] = (entities[did], classes, proba)
        except Exception as exc:
            print(f"[post] {kind} was not fit: {exc}", flush=True)
            return None
        self._oof = oof
        self._oof_key = key
        return kind


def _oof_cache_key(cfg: dict, kind: str) -> str:
    """Probabilities depend on the token-decoder spans, not only on the penalty pair."""
    fields = (
        "penalty_begin", "penalty_end", "bias", "tau", "reconcile", "propagate",
        "lexicon", "trim", "prop_min_len", "p1_strip_ms",
    )
    payload = {name: cfg.get(name) for name in fields}
    return kind + "\n" + json.dumps(payload, sort_keys=True, default=str)


def _stack_xy(ids, rows, ys):
    xs, yt = [], []
    for i in ids:
        xs.extend(rows[i])
        yt.extend(ys[i])
    if not xs:
        return None, []
    return np.stack(xs), yt


def _accept(best, trial_score, cfg, trial):
    if trial_score > best + 1e-6:
        return trial_score, trial
    return best, cfg


def tune_stack(stack: Stack, cfg: dict | None = None, shared: bool = False, verbose: bool = False) -> dict:
    """Tokdec is tuned first. Role thresholds and P3 switches are tuned on top of it."""
    cfg = json.loads(json.dumps(cfg or default_post_config()))
    tuned = stack.dec.tune(cfg, verbose=False)
    for key, value in default_post_config().items():
        tuned.setdefault(key, value)
    cfg = tuned
    if shared:
        cfg = _share_labels(stack, cfg)
        # Start from one placement threshold. The search below only tries global values,
        # so a per-label mix cannot win by being the baseline.
        cfg["theta_place"] = {lab: 0.5 for lab in ROLE_LABELS}
        cfg["p2_place_header"] = {lab: False for lab in ROLE_LABELS}

    def f(c, bundle=None):
        return stack.score(c, bundle)["macro_f1"]

    best = f(cfg)
    for flag in (False, True):
        trial = {**cfg, "p2_rules": flag, "use_p2_model": False}
        best, cfg = _accept(best, f(trial), cfg, trial)
    for flag in (False, True):
        trial = {**cfg, "p1_strip_ms": flag}
        best, cfg = _accept(best, f(trial), cfg, trial)
    # Prefer a fitted resolver when it beats the rule fallback.
    chosen = None
    for kind in ("gbdt", "logreg"):
        got = stack.bind_oof(cfg, kind)
        if got is None:
            continue
        trial = {**cfg, "use_p2_model": True, "p2_rules": False, "classifier": kind}
        score = f(trial)
        if score > best + 1e-6:
            best, cfg, chosen = score, trial, kind
    if cfg.get("use_p2_model") and chosen:
        stack.bind_oof(cfg, chosen)
    bundle_on = cfg.get("use_p2_model")
    if not shared:
        for lab in ROLE_LABELS:
            for value in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                trial = json.loads(json.dumps(cfg))
                trial["theta_place"][lab] = value
                best, cfg = _accept(best, f(trial), cfg, trial)
        for lab in ROLE_LABELS:
            for flag in (False, True):
                trial = json.loads(json.dumps(cfg))
                trial["p2_place_header"][lab] = flag
                best, cfg = _accept(best, f(trial), cfg, trial)
    else:
        for value in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            trial = json.loads(json.dumps(cfg))
            trial["theta_place"] = {lab: value for lab in ROLE_LABELS}
            best, cfg = _accept(best, f(trial), cfg, trial)
        for flag in (False, True):
            trial = json.loads(json.dumps(cfg))
            trial["p2_place_header"] = {lab: flag for lab in ROLE_LABELS}
            best, cfg = _accept(best, f(trial), cfg, trial)
    for value in (0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        trial = {**cfg, "theta_drop": value}
        best, cfg = _accept(best, f(trial), cfg, trial)
    for switch in ("p3_caseno", "p3_prec_split", "p3_prec_cite", "p3_prec_short",
                   "p3_auth_gov", "p3_auth_cand", "p3_court_split", "p3_junk"):
        trial = {**cfg, switch: not cfg.get(switch, False)}
        best, cfg = _accept(best, f(trial), cfg, trial)
    for value in (0.05, 0.1, 0.15, 0.2, 0.3):
        trial = {**cfg, "theta_auth_cand": value}
        best, cfg = _accept(best, f(trial), cfg, trial)
    cfg["oof_f1"] = best
    cfg["classifier"] = cfg.get("classifier", "rules")
    if not bundle_on:
        cfg["use_p2_model"] = False
        cfg["classifier"] = "rules"
    if verbose:
        print(f"[post] tuned oof_f1={best:.4f} classifier={cfg['classifier']}", flush=True)
    return cfg


def _share_labels(stack: Stack, cfg: dict) -> dict:
    """One propagate flag and one lexicon flag for every label.

    The choice is among the global settings only. A per-label mix is dropped even
    when it scores higher on the documents used to tune it.
    """
    best = -1.0
    out = None
    for prop in (False, True):
        for lex in (False, True):
            trial = json.loads(json.dumps(cfg))
            trial["propagate"] = {lab: prop for lab in LABELS}
            trial["lexicon"] = {lab: lex for lab in LABELS}
            score = stack.dec.score(trial)["macro_f1"]
            if out is None or score > best + 1e-6:
                best, out = score, trial
    return out


def _fit_rows(stack: Stack, cfg: dict, kind: str):
    """Feature rows with fold-excluded priors, and a classifier fit on all of them."""
    raw = stack.dec.spans(cfg["penalty_begin"], cfg["penalty_end"], cfg.get("bias"))
    priors = string_priors(stack.docs, stack.fold_of) if stack.fold_of else {}
    xs, ys = [], []
    for d in stack.docs:
        spans = finish(d.text, raw[d.doc_id], cfg, stack.lexicons.get(d.doc_id))
        header = parse_header(d.text, strip_ms=bool(cfg.get("p1_strip_ms")))
        _ent, rows, targets = document_rows(
            d, spans, stack.logps[d.doc_id], header.parties, witness_cues(d.text), counsel_cues(d.text),
            header.end, priors.get(stack.fold_of.get(d.doc_id, -1), {}),
        )
        xs.extend(rows)
        ys.extend(targets)
    if len(set(ys)) < 2 or not xs:
        return None
    all_counts: dict[str, dict[str, int]] = {}
    for d in stack.docs:
        for s, e, lab in d.gold:
            if lab not in ROLE_LABELS:
                continue
            bucket = all_counts.setdefault(d.text[s:e], {})
            bucket[lab] = bucket.get(lab, 0) + 1
    by_doc = {}
    for d in stack.docs:
        by_doc[d.doc_id] = priors.get(stack.fold_of.get(d.doc_id, -1), {})
    return {
        "clf": fit_classifier(kind, np.stack(xs), ys),
        "priors": all_counts,
        "priors_by_doc": by_doc,
    }


def fit_test_model(docs, logps, folds, cfg: dict):
    """One classifier on every training document's OOF features, for test decoding."""
    if not cfg.get("use_p2_model"):
        return None
    fold_of = {did: i for i, ids in enumerate(folds) for did in ids}
    stack = Stack(docs, logps, oof_lexicons(docs, folds), fold_of)
    kind = cfg.get("classifier", "gbdt")
    fitted = _fit_rows(stack, cfg, kind)
    if fitted is None:
        return None
    fitted["fold_of"] = {}
    return fitted


def nested_macro(docs, logps, folds, lexicons=None, shared: bool = False) -> tuple[float, list[float]]:
    """Leave one document fold out: fit and tune on the rest, score the held fold.

    Lexicons are rebuilt from the outer-train documents. `lexicons` is accepted so
    older callers keep working; it is not used, because a map built on all five
    folds would copy the held-out fold's gold strings into the score.
    """
    del lexicons
    fold_of = {did: i for i, ids in enumerate(folds) for did in ids}
    scores = []
    for ids in folds:
        held = set(ids)
        train = [d for d in docs if d.doc_id not in held and d.doc_id in logps]
        dev = [d for d in docs if d.doc_id in held and d.doc_id in logps]
        if not train or not dev:
            continue
        train_ids = {d.doc_id for d in train}
        groups = []
        for fold_ids in folds:
            group = [i for i in fold_ids if i in train_ids]
            if group:
                groups.append(group)
        train_lex = oof_lexicons(train, groups)
        held_one = build_lexicon(train)
        held_lex = {d.doc_id: held_one for d in dev}
        stack = Stack(train, logps, train_lex, {d.doc_id: fold_of[d.doc_id] for d in train})
        cfg = tune_stack(stack, shared=shared)
        bundle = None
        if cfg.get("use_p2_model"):
            bundle = _fit_rows(stack, cfg, cfg.get("classifier", "gbdt"))
        held_stack = Stack(dev, logps, held_lex, {d.doc_id: fold_of[d.doc_id] for d in dev})
        scores.append(held_stack.score(cfg, bundle)["macro_f1"])
    mean = float(sum(scores) / len(scores)) if scores else 0.0
    return mean, scores
