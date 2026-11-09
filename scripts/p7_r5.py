"""R5: R2 ranker (top-200 shortlist) + reporter-citation links + cited-year features, seed bagging.

The dev gate uses models fit on train only. The test file comes from the same configuration
refit on train + dev. A reporter link is forced into its query's 7.
"""
from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("P7_SHORT", "200")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p7_common import OUT, check_submission, half_scores, save_metrics, split_bundle, years
from p7_probe8 import body_keys, body_text, header_keys
from p7_r1 import named_zip
from p7_r2 import (
    apply_decoder,
    build,
    decode_grid,
    feature_names,
    fill_scores,
    fit_ranker,
    oof_predict,
    rows_of,
)

SEEDS = (781, 782, 783, 784, 785)
CITED = re.compile(r"<CITATION>([^<]{0,60})")
YEAR = re.compile(r"(1[89]\d\d|20[0-2]\d)")
NEW = ["reporter", "reporter_any", "year_match0", "year_match1", "n_cited_years"]


def extra_features(split_name: str, qids, cand):
    bundle = split_bundle(split_name)
    if [str(q) for q in bundle["qids"]] != list(qids) or [str(c) for c in bundle["cand"]] != list(cand):
        raise SystemExit(f"{split_name} id order differs")
    index: dict = {}
    for j, doc in enumerate(bundle["docs"]):
        for key in header_keys(doc):
            index.setdefault(key, set()).add(j)
    cy = years(bundle["sents"]).astype(np.int64)
    n_q, n_c = len(qids), len(cand)
    rep = np.zeros((n_q, n_c), dtype=np.float32)
    m0 = np.zeros((n_q, n_c), dtype=np.float32)
    m1 = np.zeros((n_q, n_c), dtype=np.float32)
    ny = np.zeros((n_q, n_c), dtype=np.float32)
    for i, sents in enumerate(bundle["q_sents"]):
        text = " ".join(sents)
        for key in body_keys(body_text(text)):
            for j in index.get(key, ()):
                if cand[j] != qids[i]:
                    rep[i, j] = 1.0
        cited = {int(y) for seg in CITED.findall(text) for y in YEAR.findall(seg)}
        ny[i] = len(cited)
        if cited:
            arr = np.array(sorted(cited))
            known = cy > 0
            diff = np.abs(cy[:, None] - arr[None, :]).min(axis=1)
            m0[i] = (known & (diff == 0)).astype(np.float32)
            m1[i] = (known & (diff <= 1)).astype(np.float32)
    any_rep = np.repeat(rep.max(axis=1, keepdims=True), n_c, axis=1)
    print(f"[r5] {split_name} reporter links {int(rep.sum())} queries {int((rep.sum(1) > 0).sum())}", flush=True)
    return {"reporter": rep, "reporter_any": any_rep, "year_match0": m0, "year_match1": m1, "n_cited_years": ny}


def add_extras(built, extras) -> None:
    built["cols"].update(extras)
    rep = extras["reporter"]
    for row in range(len(built["qids"])):
        linked = np.where(rep[row] > 0)[0]
        if len(linked):
            built["short"][row] = np.unique(np.concatenate([built["short"][row], linked.astype(np.int32)]))


def bagged_scores(models, built, names) -> np.ndarray:
    xx, _, _, keep = rows_of(built, names, None)
    pred = np.mean([m.predict(xx) for m in models], axis=0)
    return fill_scores(len(built["qids"]), len(built["cand"]), built["short"], keep, pred)


def force_links(pred: dict, built, rep: np.ndarray) -> dict:
    out = {}
    cand = built["cand"]
    for row, q in enumerate(built["qids"]):
        linked = [cand[j] for j in np.where(rep[row] > 0)[0]
                  if not built["self_mask"][row, j] and not built["after"][row, j]][:7]
        rest = [c for c in pred[q] if c not in linked]
        out[q] = (linked + rest)[:7]
    return out


def fit_bag(x, y, groups, params):
    return [fit_ranker(x, y, groups, params, params["rounds"], seed=s) for s in SEEDS]


def stacked(parts):
    xs, ys, gs = zip(*parts)
    return np.vstack(xs), np.concatenate(ys), [g for group in gs for g in group]


def run_variant(label, names, train, dev, test, train_gold, dev_gold_sets, dev_gold, params, ref):
    print(f"[r5] variant {label} features {len(names)}", flush=True)
    oof = oof_predict(train, names, train_gold, params)
    choice = decode_grid(train["qids"], train["cand"], oof, train["marker_counts"], train_gold,
                         train["self_mask"], train["after"])
    xt, yt, gt, _ = rows_of(train, names, train_gold)
    models = fit_bag(xt, yt, gt, params)
    dev_scores = bagged_scores(models, dev, names)
    dev_pred = apply_decoder(dev, dev_scores, dev["marker_counts"], choice)
    plain = half_scores(dev_gold, dev_pred, dev["qids"])
    forced_pred = force_links(dev_pred, dev, dev["cols"]["reporter"])
    forced = half_scores(dev_gold, forced_pred, dev["qids"])
    use_force = forced["all"]["f1"] >= plain["all"]["f1"]
    report = forced if use_force else plain
    print(f"[r5] {label} dev plain", {k: round(plain[k]["f1"], 4) for k in ("all", "A", "B")}, flush=True)
    print(f"[r5] {label} dev forced", {k: round(forced[k]["f1"], 4) for k in ("all", "A", "B")}, flush=True)
    ok = (report["all"]["f1"] >= ref["all"] + 0.02
          and report["A"]["f1"] >= ref["A"] - 0.01
          and report["B"]["f1"] >= ref["B"] - 0.01)
    print(f"[r5] {label} gate {ok} (force {use_force})", flush=True)
    # Final file: same configuration refit on train + dev.
    xd, yd, gd, _ = rows_of(dev, names, dev_gold_sets)
    xa, ya, ga = stacked([(xt, yt, gt), (xd, yd, gd)])
    final = fit_bag(xa, ya, ga, params)
    test_scores = bagged_scores(final, test, names)
    test_pred = apply_decoder(test, test_scores, test["marker_counts"], choice)
    if use_force:
        test_pred = force_links(test_pred, test, test["cols"]["reporter"])
    check_submission(test_pred, test["qids"], test["cand"])
    summary = {
        "dev_plain": plain, "dev_forced": forced, "force": use_force, "gate": ok,
        "decoder": {"kind": choice[1], "knob": choice[2], "train_oof": choice[0]},
        "importance": sorted(zip(names, models[0].feature_importances_.tolist()), key=lambda t: -t[1])[:15],
    }
    return summary, test_pred, ok


def main() -> None:
    e0 = json.load(open(os.path.join(OUT, "e0", "metrics.json")))
    r1 = json.load(open(os.path.join(OUT, "r1", "metrics.json")))
    s200 = json.load(open(os.path.join(OUT, "r2_s200", "metrics.json")))
    params = s200["params"]
    ref_dev = s200["results"]["citator"]["dev"]
    ref = {k: ref_dev[k]["f1"] for k in ("all", "A", "B")}
    print(f"[r5] params {params} reference {ref}", flush=True)
    k1, b = r1["params"]["full4"]["k1"], r1["params"]["full4"]["b"]
    tb = split_bundle("Train")
    db = split_bundle("Dev")
    train_gold_raw = {str(q): {str(c) for c in (row.get("relevant_candidates") or [])} for q, row in tb["queries"].items()}
    dev_gold = {str(q): [str(c) for c in (row.get("relevant_candidates") or [])] for q, row in db["queries"].items()}
    del tb, db
    train = build("train", e0, train_gold_raw, k1, b, leave_out=True)
    dev = build("dev", e0, train_gold_raw, k1, b, leave_out=False)
    test = build("test", e0, train_gold_raw, k1, b, leave_out=False)
    for built, name in ((train, "Train"), (dev, "Dev"), (test, "Test")):
        add_extras(built, extra_features(name, built["qids"], built["cand"]))
    train_gold = {q: train_gold_raw.get(q, set()) & set(train["cand"]) for q in train["qids"]}
    dev_gold_sets = {q: set(dev_gold.get(q, [])) & set(dev["cand"]) for q in dev["qids"]}
    variants = {
        "reporter_citator": feature_names(with_citator=True) + NEW,
        "reporter": feature_names(with_citator=False) + NEW,
    }
    only = os.environ.get("P7_ONLY")
    if only:
        variants = {only: variants[only]}
    force_zip = os.environ.get("P7_FORCE_ZIP") == "1"
    results = {}
    for label, names in variants.items():
        summary, pred, ok = run_variant(label, names, train, dev, test, train_gold, dev_gold_sets, dev_gold, params, ref)
        results[label] = summary
        name = f"P7_R5_{label}"
        if ok or force_zip or label == "reporter":
            named_zip(name, pred)
        else:
            print(f"[r5] {label} gate failed, no zip", flush=True)
    save_metrics("r5" + (f"_{only}" if only else ""), {"params": params, "reference": ref, "results": results})
    print("[r5] done", flush=True)


if __name__ == "__main__":
    main()
