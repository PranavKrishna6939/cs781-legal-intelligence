"""Equal-weight log-odds mix and the member rule."""
from __future__ import annotations

import json
import os

import numpy as np

from saul.cjpe import exp_dir, p4_dir
from saul.cjpe.calibrate import apply_prior, cross_fitted_stacker, cross_fitted_threshold, checked_temperature, log_odds
from saul.cjpe.const import MEMBER_MARGIN, MIX_ADD
from saul.cjpe.data import load_split
from saul.cjpe.score import macro_scores, predict_logits
from saul.utils.io import ensure_dir


def mix_logits(members: list[np.ndarray]) -> np.ndarray:
    stacked = np.mean([log_odds(logits) for logits in members], axis=0)
    return np.stack([-stacked / 2, stacked / 2], axis=1)


def _load(exp: str, kind: str) -> tuple[list[str], np.ndarray]:
    blob = np.load(os.path.join(exp_dir(exp), f"logits_{kind}.npz"), allow_pickle=True)
    ids = [str(i) for i in blob["ids"].tolist()]
    return ids, np.asarray(blob["logits"], dtype=np.float64)


def _align(exp: str, kind: str, order: list[str], plain: bool) -> np.ndarray:
    ids, logits = _load(exp, kind)
    logits = apply_prior(logits, plain)
    index = {case_id: i for i, case_id in enumerate(ids)}
    return np.stack([logits[index[case_id]] for case_id in order])


def _done(exp: str) -> dict:
    path = os.path.join(exp_dir(exp), "done.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


def select_members(scores: dict[str, float], mix_fn, margin: float = MEMBER_MARGIN) -> list[str]:
    """Keep models within `margin` of the best, then drop any that leave-one-out shows do not help."""
    if not scores:
        return []
    best = max(scores.values())
    ranked = [name for name, score in sorted(scores.items(), key=lambda item: -item[1]) if best - score <= margin]
    if not ranked:
        return []
    chosen = list(ranked)
    while len(chosen) > 1:
        full = mix_fn(chosen)
        dropped = None
        for name in chosen:
            if name == ranked[0]:
                continue
            without = [item for item in chosen if item != name]
            if mix_fn(without) + 1e-12 >= full:
                dropped = name
                break
        if dropped is None:
            break
        chosen = [item for item in chosen if item != dropped]
    return chosen


def _decade(years) -> np.ndarray:
    return (np.asarray(years) // 10).astype(np.float64)


def _prob_logits(positive) -> np.ndarray:
    clipped = np.clip(np.asarray(positive, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.stack([np.log(1.0 - clipped), np.log(clipped)], axis=1)


def _lengths(split) -> np.ndarray:
    return np.asarray(split.offsets[1:] - split.offsets[:-1], dtype=np.float64)


def build_mix(exps: list[str], name: str, force_members: list[str] | None = None) -> dict:
    dsel = load_split("dsel")
    present = [exp for exp in exps if os.path.exists(os.path.join(exp_dir(exp), "logits_dsel.npz"))]
    if not present:
        raise RuntimeError("no member logits")
    plain = {exp: _done(exp).get("loss") == "ce" for exp in present}
    temperatures = {}
    corrected = {}
    scores = {}
    for exp in present:
        logits = _align(exp, "dsel", dsel.ids, plain[exp])
        info = checked_temperature(logits, dsel.labels, np.asarray(dsel.halves))
        temperatures[exp] = info["temperature"]
        corrected[exp] = logits / info["temperature"]
        scores[exp] = macro_scores(dsel.labels, predict_logits(corrected[exp]))["f1"]

    def mix_fn(names: list[str]) -> float:
        logits = mix_logits([corrected[name] for name in names])
        return macro_scores(dsel.labels, predict_logits(logits))["f1"]

    members = force_members or select_members(scores, mix_fn)
    members = [name for name in members if name in corrected]
    if not force_members:
        # A3's gate: a hierarchical model outside the 0.06 band can enter when it adds +0.5 points.
        for exp in sorted(scores, key=lambda name: -scores[name]):
            if exp in members or not str(exp).startswith("A3"):
                continue
            if mix_fn(members + [exp]) >= mix_fn(members) + MIX_ADD:
                members.append(exp)
    mixed = mix_logits([corrected[name] for name in members])
    base_scores = log_odds(mixed)
    threshold = cross_fitted_threshold(base_scores, dsel.labels, np.asarray(dsel.halves))
    stack = cross_fitted_stacker(
        [corrected[name] for name in members], dsel.labels, np.asarray(dsel.halves),
        np.log(np.maximum(_lengths(dsel), 1.0)), _decade(dsel.years), base_scores,
    )
    best_single = max(scores[name] for name in members)
    chosen_threshold = threshold["threshold"] if threshold["kept"] else 0.0
    directory = os.path.join(p4_dir(), "mix", name)
    ensure_dir(directory)
    # The default zip stays at threshold 0. The tuned threshold is a separate zip.
    pred = predict_logits(mixed, 0.0)
    dev = macro_scores(dsel.labels, pred)
    tuned = macro_scores(dsel.labels, predict_logits(mixed, chosen_threshold))
    blob = {
        "name": name,
        "members": members,
        "scores": scores,
        "temperatures": temperatures,
        "threshold_kept": threshold["kept"],
        "threshold": chosen_threshold,
        "threshold_gain": threshold["gain"],
        "stack_kept": stack["kept"],
        "stack_gain": stack["gain"],
        "dev_f1": dev["f1"],
        "threshold_dev_f1": tuned["f1"],
        "best_single": best_single,
    }
    with open(os.path.join(directory, "calib.json"), "w", encoding="utf-8") as handle:
        json.dump(blob, handle, indent=2)
    np.savez(os.path.join(directory, "logits_dsel.npz"), ids=np.asarray(dsel.ids), logits=mixed, labels=dsel.labels)
    test = load_split("test")
    test_members = [_align(exp, "test", test.ids, plain[exp]) / temperatures[exp] for exp in members]
    test_mix = mix_logits(test_members)
    np.savez(os.path.join(directory, "logits_test.npz"), ids=np.asarray(test.ids), logits=test_mix)
    if stack["kept"]:
        _save_stack(name, members, plain, temperatures, dsel, test, stack["oof_positive"])
    print(f"[mix] {name} members={members} dev={dev['f1']:.4f} threshold_kept={threshold['kept']} stack_kept={stack['kept']}", flush=True)
    return blob


def _save_stack(name, members, plain, temperatures, dsel, test, oof_positive) -> None:
    """Refit the stacker on all of D-sel only after the cross-fit kept it, and predict test.

    D-sel logits stay the cross-fit scores. The refit is used for test only.
    """
    features_d = []
    ys = dsel.labels
    for exp in members:
        features_d.append(log_odds(_align(exp, "dsel", dsel.ids, plain[exp]) / temperatures[exp]))
    extra = np.stack([np.log(np.maximum(_lengths(dsel), 1.0)), _decade(dsel.years)], axis=1)
    matrix = np.concatenate([np.stack(features_d, axis=1), extra], axis=1)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200)
    clf.fit(matrix, ys)
    test_cols = [log_odds(_align(exp, "test", test.ids, plain[exp]) / temperatures[exp]) for exp in members]
    test_extra = np.stack([np.log(np.maximum(_lengths(test), 1.0)), _decade(test.years)], axis=1)
    test_matrix = np.concatenate([np.stack(test_cols, axis=1), test_extra], axis=1)
    proba = clf.predict_proba(test_matrix)
    classes = list(clf.classes_)
    positive = proba[:, classes.index(1)]
    logits = np.stack([np.log(1 - positive + 1e-6), np.log(positive + 1e-6)], axis=1)
    directory = os.path.join(p4_dir(), "mix", name + "_stack")
    ensure_dir(directory)
    np.savez(os.path.join(directory, "logits_test.npz"), ids=np.asarray(test.ids), logits=logits)
    np.savez(
        os.path.join(directory, "logits_dsel.npz"),
        ids=np.asarray(dsel.ids), logits=_prob_logits(oof_positive), labels=ys,
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--members", default="")
    args = parser.parse_args()
    force = [item for item in args.members.split(",") if item] or None
    build_mix(args.exps.split(","), args.name, force)


if __name__ == "__main__":
    main()
