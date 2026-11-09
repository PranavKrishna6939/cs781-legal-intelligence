"""R2: LightGBM LambdaMART over the R1 lexical views and structural features.

Hyperparameters and ablations are chosen by grouped cross-validation on train.
Dev is only the gate. A zip is written only when that gate passes.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p7_bm25 import micro_f1
from p7_common import (
    OUT,
    apply_mask,
    check_submission,
    decode,
    fit_marker_mass,
    half_scores,
    masks,
    save_metrics,
    split_bundle,
    zrows,
)
from p7_e0 import marker_counts
from p7_r1 import named_zip

SHORT = int(os.environ.get("P7_SHORT", "100"))
TAG = os.environ.get("P7_TAG", "")
SEED = 781
BASE_VIEWS = ("full4", "full5", "head3", "ctx4", "shingle")


def load_views(split: str) -> dict[str, np.ndarray]:
    path = os.path.join(OUT, "r1", f"{split}.npz")
    stored = np.load(path)
    need = ("full3", "full4", "full5", "head1", "head3", "head4", "ctx1", "ctx4",
            "shingle", "statute", "year_gap", "full", "head", "context", "citator")
    missing = [name for name in need if name not in stored.files]
    if missing:
        raise SystemExit(f"{path} is missing {missing}. R1 has not finished saving views.")
    qids = [str(x) for x in stored["qids"]]
    cand = [str(x) for x in stored["cand"]]
    return {"qids": qids, "cand": cand, **{name: stored[name] for name in need}}


def ranks(scores: np.ndarray, axis: int) -> np.ndarray:
    order = np.argsort(-scores, axis=axis)
    rank = np.empty(order.shape, dtype=np.int32)
    if axis == 1:
        rank[np.arange(scores.shape[0])[:, None], order] = np.arange(scores.shape[1])
    else:
        rank[order, np.arange(scores.shape[1])[None, :]] = np.arange(scores.shape[0])[:, None]
    return rank


def normalized(raw: np.ndarray, self_mask, after) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masked = apply_mask(raw, self_mask, after)
    return zrows(masked), ranks(masked, 1).astype(np.float32), zrows(masked.T).T, ranks(masked, 0).astype(np.float32)


def top_idx(scores: np.ndarray, k: int) -> np.ndarray:
    take = min(k, scores.shape[1])
    part = np.argpartition(-scores, take - 1, axis=1)[:, :take]
    return part


def e0_fusion(views, meta) -> np.ndarray:
    names = ["full", "head", "context", "shingle"]
    return sum(w * zrows(views[n]) for w, n in zip(meta["weights"], names))


def gold_sets(queries, pool: set[str]) -> dict[str, set[str]]:
    out = {}
    for q, row in queries.items():
        out[str(q)] = {str(c) for c in (row.get("relevant_candidates") or [])} & pool
    return out


def cocite(qids, cand, signal: np.ndarray, train_gold: dict[str, set[str]], leave_out: bool):
    pool = set(cand)
    index = {c: i for i, c in enumerate(cand)}
    owners = defaultdict(list)
    for q, ids in train_gold.items():
        for c in ids:
            if c in index:
                owners[c].append(q)
    partner = {}
    for c, qs in owners.items():
        partner[c] = {q: train_gold[q] - {c} for q in qs}
    n_q, n_c = signal.shape
    feat_max = np.zeros((n_q, n_c), dtype=np.float32)
    feat_cnt = np.zeros((n_q, n_c), dtype=np.float32)
    qpos = {q: i for i, q in enumerate(qids)}
    for c, by_q in partner.items():
        i = index[c]
        everyone = set()
        for ps in by_q.values():
            everyone |= ps
        js_all = [index[p] for p in everyone if p in pool]
        rows_done = set()
        if leave_out:
            for q in by_q:
                if q not in qpos:
                    continue
                others = set()
                for q2, ps in by_q.items():
                    if q2 != q:
                        others |= ps
                js = [index[p] for p in others if p in pool]
                row = qpos[q]
                feat_cnt[row, i] = len(js)
                if js:
                    feat_max[row, i] = signal[row, js].max()
                rows_done.add(row)
        if js_all:
            block = signal[:, js_all].max(axis=1)
            for row in range(n_q):
                if row in rows_done:
                    continue
                feat_max[row, i] = block[row]
                feat_cnt[row, i] = len(js_all)
    return feat_max, feat_cnt


def prf_feature(head_docs: list[str], query_scores: np.ndarray, k1: float, b: float) -> np.ndarray:
    """Mean head-3 similarity to the query's current top 3, in candidate space."""
    from p7_r1 import doc_stats, hashing, okapi, query_counts

    vec = hashing(3)
    print(f"[r2] candidate head similarity {len(head_docs)}", flush=True)
    counts, idf, length = doc_stats(vec, head_docs)
    sim = okapi(counts, idf, length, query_counts(vec, head_docs), k1, b)
    np.fill_diagonal(sim, 0.0)
    sim_z = zrows(sim)
    top = top_idx(query_scores, 3)
    out = np.zeros_like(query_scores, dtype=np.float32)
    for i in range(len(top)):
        out[i] = sim_z[top[i]].mean(axis=0)
    return out


def build(split: str, meta, train_gold, k1: float, b: float, leave_out: bool):
    print(f"[r2] build {split}", flush=True)
    views = load_views(split)
    bundle = split_bundle({"train": "Train", "dev": "Dev", "test": "Test"}[split])
    self_mask, after = masks(bundle["qids"], bundle["cand"], bundle["q_sents"], bundle["sents"])
    if [str(q) for q in bundle["qids"]] != views["qids"] or [str(c) for c in bundle["cand"]] != views["cand"]:
        raise SystemExit(f"{split} id order differs from the R1 file")
    heads = [" ".join(s[:40]) for s in bundle["sents"]]
    masked4 = apply_mask(views["full4"], self_mask, after)
    fused = apply_mask(e0_fusion(views, meta), self_mask, after)
    prf = prf_feature(heads, masked4, k1, b)
    signal, _, _, _ = normalized(views["full4"], self_mask, after)
    cite_max, cite_cnt = cocite(views["qids"], views["cand"], signal, train_gold, leave_out)
    cols = {}
    for name in ("full3", "full4", "full5", "head1", "head3", "head4", "ctx1", "ctx4",
                 "shingle", "statute", "full", "head", "context", "year_gap"):
        z, row_rank, col_z, col_rank = normalized(views[name], self_mask, after)
        cols[f"z_{name}"] = z
        if name in BASE_VIEWS:
            cols[f"rank_{name}"] = row_rank
            cols[f"colz_{name}"] = col_z
            cols[f"colrank_{name}"] = col_rank
    cols["z_e0"] = zrows(fused)
    cols["prf"] = zrows(apply_mask(prf, self_mask, after))
    cols["cocite_max"] = cite_max
    cols["cocite_cnt"] = cite_cnt
    cols["citator"] = views["citator"]
    counts = marker_counts(bundle)
    del bundle, heads
    short = []
    for row in range(len(views["qids"])):
        pick = set(top_idx(masked4[row][None, :], SHORT)[0]) | set(top_idx(fused[row][None, :], SHORT)[0])
        short.append(np.fromiter(pick, dtype=np.int32))
    print(f"[r2] {split} features {len(cols)} shortlist mean {np.mean([len(s) for s in short]):.1f}", flush=True)
    return {
        "qids": views["qids"], "cand": views["cand"], "cols": cols, "short": short,
        "self_mask": self_mask, "after": after, "marker_counts": counts,
        "full4": masked4,
    }


def matrix_from(names, cols) -> np.ndarray:
    return np.stack([cols[n] for n in names], axis=-1)


def rows_of(built, names, gold: dict[str, set[str]] | None):
    x_parts, y_parts, groups, keep_q = [], [], [], []
    data = matrix_from(names, built["cols"])
    for row, q in enumerate(built["qids"]):
        idx = built["short"][row]
        labels = None
        if gold is not None:
            g = gold.get(q, set())
            labels = np.array([1 if built["cand"][j] in g else 0 for j in idx], dtype=np.int32)
            if labels.sum() == 0:
                continue
        x_parts.append(data[row, idx])
        if labels is not None:
            y_parts.append(labels)
        groups.append(len(idx))
        keep_q.append(row)
    return np.vstack(x_parts), (None if gold is None else np.concatenate(y_parts)), groups, keep_q


def fit_ranker(x, y, groups, params, rounds, seed=SEED):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=rounds,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=8,
        verbosity=-1,
        label_gain=[0, 1],
        random_state=seed,
    )
    model.fit(x, y, group=groups)
    return model


def fill_scores(n_q, n_c, short, keep_q, pred) -> np.ndarray:
    """Shortlist gets the ranker score. Every other candidate sits just below that row's minimum."""
    scores = np.zeros((n_q, n_c), dtype=np.float32)
    cursor = 0
    seen = set()
    for row, idx in zip(keep_q, [short[r] for r in keep_q]):
        values = pred[cursor:cursor + len(idx)]
        cursor += len(idx)
        scores[row] = float(values.min()) - 1.0
        scores[row, idx] = values
        seen.add(row)
    for row in range(n_q):
        if row not in seen:
            scores[row] = -1.0
    return scores


def decode_grid(pack_qids, cand, scores, marker, gold, self_mask, after):
    masked = apply_mask(zrows(scores), self_mask, after)
    coef = fit_marker_mass(marker, np.array([len(gold[q]) for q in pack_qids], dtype=np.float64))
    best = (-1.0, "column", 0.5, coef)
    for tau in (0.2, 0.5, 1.0):
        pred = decode(pack_qids, cand, masked, "column", marker, coef, 0.5, tau)
        f1 = micro_f1({q: list(gold[q]) for q in pack_qids}, pred, pack_qids)["f1"]
        if f1 > best[0]:
            best = (f1, "column", tau, coef)
    for eps in (0.1, 0.2, 0.5, 1.0, 2.0):
        pred = decode(pack_qids, cand, masked, "sinkhorn", marker, coef, eps, 0.5)
        f1 = micro_f1({q: list(gold[q]) for q in pack_qids}, pred, pack_qids)["f1"]
        if f1 > best[0]:
            best = (f1, "sinkhorn", eps, coef)
    print(f"[r2] decoder {best[1]} {best[2]} train {best[0]:.4f}", flush=True)
    return best


def query_blocks(built, names, gold):
    x_all, y_all, groups_all, keep_all = rows_of(built, names, gold)
    blocks = {}
    cursor = 0
    for row, size in zip(keep_all, groups_all):
        blocks[built["qids"][row]] = (cursor, cursor + size, row)
        cursor += size
    return x_all, y_all, blocks


def oof_predict(built, names, gold, params) -> np.ndarray:
    x_all, y_all, blocks = query_blocks(built, names, gold)
    order = np.array(sorted(blocks))
    rng = np.random.default_rng(SEED)
    rng.shuffle(order)
    folds = np.array_split(order, 5)
    oof = np.full((len(built["qids"]), len(built["cand"])), np.nan, dtype=np.float32)
    held = np.zeros(len(built["qids"]), dtype=bool)
    for fold in folds:
        xs, ys, gs = [], [], []
        for q, (lo, hi, _) in blocks.items():
            if q in set(fold):
                continue
            xs.append(x_all[lo:hi])
            ys.append(y_all[lo:hi])
            gs.append(hi - lo)
        model = fit_ranker(np.vstack(xs), np.concatenate(ys), gs, params, params["rounds"])
        for q in fold:
            if q not in blocks:
                continue
            lo, hi, row = blocks[q]
            values = model.predict(x_all[lo:hi])
            oof[row] = float(values.min()) - 1.0
            oof[row, built["short"][row]] = values
            held[row] = True
    # Queries with no positive in the shortlist keep the lexical score.
    oof[~held] = built["full4"][~held]
    return oof


def row_f1(built, scores, gold) -> float:
    masked = apply_mask(zrows(scores), built["self_mask"], built["after"])
    pred = decode(built["qids"], built["cand"], masked, "row")
    return micro_f1({q: list(gold[q]) for q in built["qids"]}, pred, built["qids"])["f1"]


def cv_select(built, names, gold):
    grid = []
    for leaves in (31, 63):
        for child in (20, 50):
            for rounds in (200, 500):
                grid.append({"num_leaves": leaves, "min_child_samples": child, "learning_rate": 0.05, "rounds": rounds})
    best = (-1.0, None, None)
    for params in grid:
        oof = oof_predict(built, names, gold, params)
        f1 = row_f1(built, oof, gold)
        if f1 > best[0]:
            best = (f1, params, oof)
        print(
            f"[r2] cv leaves {params['num_leaves']} child {params['min_child_samples']} "
            f"rounds {params['rounds']} row-f1 {f1:.4f}",
            flush=True,
        )
    print(f"[r2] best cv {best[0]:.4f} {best[1]}", flush=True)
    return best[1], best[2], best[0]


def apply_decoder(built, scores, marker, choice):
    _, kind, knob, coef = choice
    masked = apply_mask(zrows(scores), built["self_mask"], built["after"])
    tau = knob if kind == "column" else 0.5
    eps = knob if kind == "sinkhorn" else 0.5
    return decode(built["qids"], built["cand"], masked, kind, marker, coef, eps, tau)


def recall100(built, gold) -> float:
    hit = tot = 0
    for row, q in enumerate(built["qids"]):
        g = gold.get(q, set())
        tot += len(g)
        got = {built["cand"][j] for j in built["short"][row]}
        hit += len(g & got)
    return hit / max(tot, 1)


def feature_names(with_citator=True, columns=True, cocite=True):
    names = [f"z_{n}" for n in (
        "full3", "full4", "full5", "head1", "head3", "head4", "ctx1", "ctx4",
        "shingle", "statute", "full", "head", "context", "year_gap", "e0",
    )]
    if columns:
        for n in BASE_VIEWS:
            names += [f"rank_{n}", f"colz_{n}", f"colrank_{n}"]
    else:
        for n in BASE_VIEWS:
            names.append(f"rank_{n}")
    names += ["prf"]
    if cocite:
        names += ["cocite_max", "cocite_cnt"]
    if with_citator:
        names.append("citator")
    return names


def train_and_score(train, dev, test, names, params, train_gold, dev_gold, choice):
    x, y, groups, _ = rows_of(train, names, train_gold)
    model = fit_ranker(x, y, groups, params, params["rounds"])
    out = {
        "importance": sorted(zip(names, model.feature_importances_.tolist()), key=lambda item: -item[1])[:15],
        "decoder": {"kind": choice[1], "knob": choice[2], "mass_coef": list(choice[3]), "train_oof": choice[0]},
    }
    scored = {}
    for tag, built in (("dev", dev), ("test", test)):
        xx, _, _, keep = rows_of(built, names, None)
        pred = model.predict(xx)
        scored[tag] = fill_scores(len(built["qids"]), len(built["cand"]), built["short"], keep, pred)
    dev_pred = apply_decoder(dev, scored["dev"], dev["marker_counts"], choice)
    out["dev"] = half_scores(dev_gold, dev_pred, dev["qids"])
    test_pred = apply_decoder(test, scored["test"], test["marker_counts"], choice)
    print("[r2] dev", {k: round(out["dev"][k]["f1"], 4) for k in ("all", "A", "B")}, names[-3:], flush=True)
    return out, test_pred


def gate(report, ref) -> bool:
    return (
        report["all"]["f1"] >= ref["all"]["f1"] + 0.02
        and report["A"]["f1"] >= ref["A"]["f1"] - 0.01
        and report["B"]["f1"] >= ref["B"]["f1"] - 0.01
    )


def main() -> None:
    print(f"[r2] shortlist {SHORT} tag '{TAG}'", flush=True)
    e0 = json.load(open(os.path.join(OUT, "e0", "metrics.json")))
    r1 = json.load(open(os.path.join(OUT, "r1", "metrics.json")))
    k1 = r1["params"]["full4"]["k1"]
    b = r1["params"]["full4"]["b"]
    train_bundle = split_bundle("Train")
    dev_bundle = split_bundle("Dev")
    train_gold_raw = {str(q): {str(c) for c in (row.get("relevant_candidates") or [])} for q, row in train_bundle["queries"].items()}
    dev_gold = {str(q): [str(c) for c in (row.get("relevant_candidates") or [])] for q, row in dev_bundle["queries"].items()}
    del train_bundle, dev_bundle
    train = build("train", e0, train_gold_raw, k1, b, leave_out=True)
    dev = build("dev", e0, train_gold_raw, k1, b, leave_out=False)
    test = build("test", e0, train_gold_raw, k1, b, leave_out=False)
    train_gold = {q: train_gold_raw.get(q, set()) & set(train["cand"]) for q in train["qids"]}
    dev_gold_sets = {q: set(dev_gold.get(q, [])) for q in dev["qids"]}
    print(f"[r2] dev shortlist recall@100 {recall100(dev, dev_gold_sets):.4f}", flush=True)
    variants = {
        "lexical": feature_names(with_citator=False),
        "no_column": feature_names(with_citator=False, columns=False),
        "no_cocite": feature_names(with_citator=False, cocite=False),
        "citator": feature_names(with_citator=True),
    }
    params, lexical_oof, best_f1 = cv_select(train, variants["lexical"], train_gold)
    oof_f1 = {"lexical": row_f1(train, lexical_oof, train_gold)}
    oofs = {"lexical": lexical_oof}
    for label in ("no_column", "no_cocite", "citator"):
        oofs[label] = oof_predict(train, variants[label], train_gold, params)
        oof_f1[label] = row_f1(train, oofs[label], train_gold)
        print(f"[r2] oof {label} row-f1 {oof_f1[label]:.4f}", flush=True)
    lead = max(("lexical", "no_column", "no_cocite"), key=lambda name: oof_f1[name])
    print(f"[r2] lead by train oof: {lead}", flush=True)
    choices = {}
    for label in (lead, "citator"):
        choices[label] = decode_grid(
            train["qids"], train["cand"], oofs[label], train["marker_counts"], train_gold,
            train["self_mask"], train["after"],
        )
    results = {}
    preds = {}
    for label in ("lexical", "no_column", "no_cocite", "citator"):
        choice = choices[label] if label in choices else choices[lead]
        print(f"[r2] train {label} features {len(variants[label])}", flush=True)
        summary, pred = train_and_score(
            train, dev, test, variants[label], params, train_gold, dev_gold, choice,
        )
        summary["oof_row_f1"] = oof_f1[label]
        summary["cv_reference"] = best_f1 if label == "lexical" else None
        results[label] = summary
        preds[label] = pred
    ref = r1["dev"] if r1.get("gate") else e0["dev"]
    ref_cit = r1["dev_citator"] if r1.get("gate_citator") else e0["dev_citator"]
    lexical_ok = gate(results["lexical"]["dev"], ref)
    lead_ok = gate(results[lead]["dev"], ref)
    cit_ok = gate(results["citator"]["dev"], ref_cit)
    print(f"[r2] lead {lead} dev-gate {lead_ok} lexical-gate {lexical_ok} citator-gate {cit_ok}", flush=True)
    if lead_ok:
        check_submission(preds[lead], test["qids"], test["cand"])
        named_zip(f"P7_R2{TAG}_ltr", preds[lead])
    else:
        print("[r2] lead gate failed, no zip", flush=True)
    if cit_ok:
        check_submission(preds["citator"], test["qids"], test["cand"])
        named_zip(f"P7_R2{TAG}_ltr_citator", preds["citator"])
    else:
        print("[r2] citator gate failed, no zip", flush=True)
    payload = {"params": params, "lead": lead, "results": results, "gate": lead_ok, "gate_citator": cit_ok,
               "ref": {k: ref[k]["f1"] for k in ("all", "A", "B")},
               "dev_shortlist_recall@100": recall100(dev, dev_gold_sets)}
    path = save_metrics(f"r2{TAG}", payload)
    print("[r2] metrics", path, flush=True)


if __name__ == "__main__":
    main()
