"""Decision layer. Every fit is scored on the dev half it did not see."""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import f1_score

from saul.lsi import exp_dir, p5_dir
from saul.lsi.const import DECISION_GAIN, KAPPAS, N_LABELS, SEED, STACK_GAIN
from saul.lsi.data import dev_half_array
from saul.lsi.score import macro_f1, scores, sigmoid
from saul.utils.io import write_json


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = np.clip(sigmoid(logits / temperature), 1e-6, 1 - 1e-6)
    gold = labels > 0
    return float(-(gold * np.log(probs) + (1 - gold) * np.log(1 - probs)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature that minimises mean binary NLL. The caller must pass only the fit half."""
    best_t, best = 1.0, float("inf")
    for temperature in np.geomspace(0.5, 5.0, 21):
        score = _nll(logits, labels, float(temperature))
        if score < best:
            best, best_t = score, float(temperature)
    return best_t


def choose_temperature(logits: np.ndarray, labels: np.ndarray, half) -> dict:
    """Fit T on one half and check NLL on the other, then swap. The reported T is their mean."""
    half = np.asarray(half)
    by_half = {}
    gains = []
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit, score = half == fit_name, half == score_name
        temperature = fit_temperature(logits[fit], labels[fit])
        by_half[fit_name] = temperature
        gains.append(_nll(logits[score], labels[score], 1.0) - _nll(logits[score], labels[score], temperature))
    return {"T": float(np.mean(list(by_half.values()))), "by_half": by_half, "nll_gain": float(np.mean(gains))}


def load_logits(exp: str, kind: str) -> dict:
    path = os.path.join(exp_dir(exp), f"{kind}_logits.npz")
    blob = np.load(path, allow_pickle=True)
    out = {"ids": [str(i) for i in blob["ids"]], "logits": np.asarray(blob["logits"], dtype=np.float64)}
    if "labels" in blob.files:
        out["labels"] = np.asarray(blob["labels"], dtype=np.float64)
    return out


def align(exp: str, kind: str, order: list[str]) -> np.ndarray:
    blob = load_logits(exp, kind)
    index = {case_id: i for i, case_id in enumerate(blob["ids"])}
    missing = [case_id for case_id in order if case_id not in index]
    if missing:
        raise KeyError(f"{exp} {kind} missing {len(missing)} ids")
    return blob["logits"][[index[case_id] for case_id in order]]


def cross_fit(values, half, fit_fn, apply_fn):
    """Fit on A and score B, then the reverse. `fit_fn` returns a state. `apply_fn(state, rows)` predicts."""
    half = np.asarray(half)
    pred = np.zeros(values.shape[:1] + ((values.shape[1],) if values.ndim == 2 else ()), dtype=np.int64)
    # predictions are boolean matrices; callers pass apply_fn that returns bool [n, L]
    pred = np.zeros((len(values), values.shape[-1]), dtype=bool)
    states = {}
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit_rows = half == fit_name
        score_rows = half == score_name
        state = fit_fn(values[fit_rows], fit_rows)
        states[fit_name] = state
        pred[score_rows] = apply_fn(state, values[score_rows])
    return pred, states


def _label_f1(y, pred) -> float:
    if y.sum() == 0 and pred.sum() == 0:
        return 0.0
    return float(f1_score(y, pred, zero_division=0))


def optimal_label_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    thresholds = np.full(probs.shape[1], 0.5)
    for label in range(probs.shape[1]):
        column = probs[:, label]
        grid = np.unique(np.quantile(column, np.linspace(0.01, 0.99, 99)))
        best_f, best_t = -1.0, 0.5
        gold = labels[:, label] > 0
        for threshold in grid:
            score = _label_f1(gold, column >= threshold)
            if score > best_f:
                best_f, best_t = score, float(threshold)
        thresholds[label] = best_t
    return thresholds


def best_global_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 19)
    best_f, best_t = -1.0, 0.5
    for threshold in grid:
        score = macro_f1(labels, probs >= threshold)
        if score > best_f:
            best_f, best_t = score, float(threshold)
    return best_t


def shrink_thresholds(per_label: np.ndarray, global_t: float, counts: np.ndarray, kappa) -> np.ndarray:
    if kappa is None:
        return np.full_like(per_label, global_t)
    counts = np.asarray(counts, dtype=np.float64)
    return (counts * per_label + float(kappa) * global_t) / (counts + float(kappa))


def choose_kappa(probs: np.ndarray, labels: np.ndarray, half) -> dict:
    half = np.asarray(half)
    gains = []
    chosen = {}
    base_parts = []
    new_parts = []
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit, score = half == fit_name, half == score_name
        counts = labels[fit].sum(0)
        raw = optimal_label_thresholds(probs[fit], labels[fit])
        global_t = best_global_threshold(probs[fit], labels[fit])
        base = macro_f1(labels[score], probs[score] >= global_t)
        best_k, best_f = None, -1.0
        for kappa in KAPPAS:
            thr = shrink_thresholds(raw, global_t, counts, kappa)
            score_f = macro_f1(labels[score], probs[score] >= thr)
            if score_f > best_f:
                best_f, best_k = score_f, kappa
        chosen[fit_name] = {"kappa": best_k, "global": global_t, "raw": raw, "counts": counts}
        base_parts.append(base)
        new_parts.append(best_f)
        gains.append(best_f - base)
    # One kappa for the final fit: the one that won on both directions, else the larger (more shrinkage).
    winners = [chosen["A"]["kappa"], chosen["B"]["kappa"]]
    if winners[0] == winners[1]:
        kappa = winners[0]
    else:
        rank = {value: index for index, value in enumerate(KAPPAS)}
        kappa = winners[0] if rank[winners[0]] >= rank[winners[1]] else winners[1]
    gain = float(np.mean(gains))
    # Unshrunk or lightly shrunk thresholds have to clear +0.3. Otherwise stay on the global threshold.
    if gain < DECISION_GAIN:
        kappa = None
    insample_thr = shrink_thresholds(
        optimal_label_thresholds(probs, labels),
        best_global_threshold(probs, labels),
        labels.sum(0),
        kappa,
    )
    return {
        "kappa": kappa,
        "gain": gain,
        "half_gain": [float(g) for g in gains],
        "base": float(np.mean(base_parts)),
        "tuned": float(np.mean(new_parts)),
        "insample": float(macro_f1(labels, probs >= insample_thr)),
        "states": chosen,
    }


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return probs >= thresholds


def at_least_one(probs: np.ndarray, thresholds: np.ndarray, second: bool = False) -> np.ndarray:
    pred = probs >= thresholds
    ratio = probs / np.clip(thresholds, 1e-6, None)
    empty = ~pred.any(axis=1)
    top = ratio.argmax(axis=1)
    pred[empty, top[empty]] = True
    if second:
        order = np.argsort(-ratio, axis=1)
        for row in range(len(pred)):
            nxt = int(order[row, 1]) if order.shape[1] > 1 else int(order[row, 0])
            if (not pred[row, nxt]) and ratio[row, nxt] >= 0.8:
                pred[row, nxt] = True
    return pred


def d2_gain(probs, labels, half, kappa) -> dict:
    """Thresholds for the at-least-one comparison are fit on the half that is not scored."""
    half = np.asarray(half)
    base_parts, one_parts, two_parts = [], [], []
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit, score = half == fit_name, half == score_name
        thresholds = shrink_thresholds(
            optimal_label_thresholds(probs[fit], labels[fit]),
            best_global_threshold(probs[fit], labels[fit]),
            labels[fit].sum(0),
            kappa,
        )
        base = apply_thresholds(probs[score], thresholds)
        one = at_least_one(probs[score], thresholds, second=False)
        two = at_least_one(probs[score], thresholds, second=True)
        base_parts.append(macro_f1(labels[score], base))
        one_parts.append(macro_f1(labels[score], one))
        two_parts.append(macro_f1(labels[score], two))
    base_m = float(np.mean(base_parts))
    one_g = float(np.mean(one_parts) - base_m)
    two_g = float(np.mean(two_parts) - base_m)
    mode = "off"
    if max(one_g, two_g) >= DECISION_GAIN:
        mode = "one" if one_g >= two_g else "two"
    return {"mode": mode, "gain_one": one_g, "gain_two": two_g, "base": base_m}


class Stacker:
    """z + W sigmoid(z) + b, diagonal of W fixed at 0."""

    def __init__(self, n_labels: int = N_LABELS, l2: float = 1e-2, steps: int = 200, lr: float = 0.05):
        self.n_labels = n_labels
        self.l2 = l2
        self.steps = steps
        self.lr = lr
        self.W = np.zeros((n_labels, n_labels), dtype=np.float64)
        self.b = np.zeros(n_labels, dtype=np.float64)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "Stacker":
        import torch
        z = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        W = torch.zeros(self.n_labels, self.n_labels, requires_grad=True)
        b = torch.zeros(self.n_labels, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=self.lr)
        eye = torch.eye(self.n_labels)
        for _ in range(self.steps):
            masked = W * (1 - eye)
            pred = z + torch.sigmoid(z) @ masked.T + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y)
            loss = loss + self.l2 * masked.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            self.W = (W * (1 - eye)).numpy()
            self.b = b.numpy()
        return self

    def apply(self, logits: np.ndarray) -> np.ndarray:
        prob = sigmoid(logits)
        return logits + prob @ self.W.T + self.b


def stacker_oof(logits: np.ndarray, labels: np.ndarray, half, n_folds: int = 5, seed: int = SEED) -> np.ndarray:
    """5-fold OOF inside each dev half is not required: folds span all dev, and each row is predicted
    by a stacker that did not train on it."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(logits))
    folds = np.array_split(order, n_folds)
    oof = np.zeros_like(logits)
    for fold in folds:
        train = np.ones(len(logits), dtype=bool)
        train[fold] = False
        model = Stacker(n_labels=logits.shape[1]).fit(logits[train], labels[train])
        oof[fold] = model.apply(logits[fold])
    return oof


def d5_modes(pred: np.ndarray, duplicate_labels: list[list[int] | None]) -> dict[str, np.ndarray]:
    out = {"off": pred.copy()}
    replace = np.zeros_like(pred)
    union = pred.copy()
    for row, labels in enumerate(duplicate_labels):
        if not labels:
            replace[row] = pred[row]
            continue
        replace[row, labels] = True
        union[row, labels] = True
    out["replace"] = replace
    out["union"] = union
    return out


def choose_d5(pred: np.ndarray, labels: np.ndarray, duplicate_labels: list[list[int] | None], half) -> dict:
    """Pick replace/union on one half's duplicate cases and score it on the other half."""
    half = np.asarray(half)
    has = np.asarray([bool(item) for item in duplicate_labels])
    if not has.any():
        return {"mode": "off", "gain": 0.0, "n": 0, "by_half": {"A": "off", "B": "off"}}
    modes = d5_modes(pred, duplicate_labels)
    gains = []
    chosen = {}
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit_rows = (half == fit_name) & has
        score_rows = (half == score_name) & has
        if not fit_rows.any() or not score_rows.any():
            chosen[fit_name] = "off"
            gains.append(0.0)
            continue
        fit_base = macro_f1(labels[fit_rows], modes["off"][fit_rows])
        best_mode, best_fit = "off", 0.0
        for mode in ("replace", "union"):
            gain = macro_f1(labels[fit_rows], modes[mode][fit_rows]) - fit_base
            if gain > best_fit:
                best_mode, best_fit = mode, float(gain)
        score_base = macro_f1(labels[score_rows], modes["off"][score_rows])
        gains.append(macro_f1(labels[score_rows], modes[best_mode][score_rows]) - score_base)
        chosen[fit_name] = best_mode
    gain = float(np.mean(gains))
    agreed = chosen.get("A") == chosen.get("B")
    mode = chosen.get("A", "off") if agreed and gain >= DECISION_GAIN else "off"
    return {"mode": mode, "gain": gain, "n": int(has.sum()), "by_half": chosen}


def final_thresholds(probs: np.ndarray, labels: np.ndarray, kappa) -> np.ndarray:
    raw = optimal_label_thresholds(probs, labels)
    global_t = best_global_threshold(probs, labels)
    return shrink_thresholds(raw, global_t, labels.sum(0), kappa)


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def predict_matrix(probs: np.ndarray, thresholds: np.ndarray, d2_mode: str,
                   duplicate_labels: list[list[int] | None] | None, d5_mode: str) -> np.ndarray:
    if d2_mode == "one":
        pred = at_least_one(probs, thresholds, second=False)
    elif d2_mode == "two":
        pred = at_least_one(probs, thresholds, second=True)
    else:
        pred = apply_thresholds(probs, thresholds)
    if duplicate_labels and d5_mode != "off":
        pred = d5_modes(pred, duplicate_labels)[d5_mode]
    return pred


def quick_macro(logits: np.ndarray, labels: np.ndarray, half, kappa: int = 50) -> dict:
    """Cross-fitted macro-F1 used while training. Thresholds are shrunk with a fixed kappa."""
    probs = sigmoid(logits)
    half = np.asarray(half)
    parts = []
    pred = np.zeros_like(probs, dtype=bool)
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit, score = half == fit_name, half == score_name
        thr = shrink_thresholds(
            optimal_label_thresholds(probs[fit], labels[fit]),
            best_global_threshold(probs[fit], labels[fit]),
            labels[fit].sum(0),
            kappa,
        )
        pred[score] = probs[score] >= thr
        parts.append(macro_f1(labels[score], pred[score]))
    return {"macro_f1": float(np.mean(parts)), "half": {"A": parts[1], "B": parts[0]}, "pred": pred}


def run_decision(dev_logits: np.ndarray, dev_labels: np.ndarray, dev_ids: list[str],
                 test_logits: np.ndarray, test_ids: list[str], knn_dev: np.ndarray | None = None,
                 knn_test: np.ndarray | None = None, duplicate_dev: list | None = None,
                 duplicate_test: list | None = None, name: str = "decide") -> dict:
    half = np.asarray(dev_half_array(dev_ids))
    pieces = []
    # D0. Each dev half is scaled by the temperature fit on the other half. Test uses their mean.
    d0 = choose_temperature(dev_logits, dev_labels, half)
    dev_scaled = np.array(dev_logits, dtype=np.float64, copy=True)
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        rows = half == score_name
        dev_scaled[rows] = dev_logits[rows] / d0["by_half"][fit_name]
    test_scaled = np.asarray(test_logits, dtype=np.float64) / d0["T"]
    pieces.append({"piece": "D0", "T": d0["T"], "nll_gain": d0["nll_gain"]})
    # D3, then D4. Dev rows are out of fold. Test is fit on all of dev, after the stacker.
    oof = stacker_oof(dev_scaled, dev_labels, half)
    base_m = quick_macro(dev_scaled, dev_labels, half)["macro_f1"]
    stack_m = quick_macro(oof, dev_labels, half)["macro_f1"]
    stack_gain = stack_m - base_m
    keep_stack = stack_gain >= STACK_GAIN
    pieces.append({"piece": "D3", "gain": float(stack_gain), "keep": keep_stack})
    current = oof if keep_stack else dev_scaled
    pre_stack = dev_scaled
    test_current = test_scaled
    if keep_stack:
        test_current = Stacker(n_labels=pre_stack.shape[1]).fit(pre_stack, dev_labels).apply(test_current)
    keep_knn = False
    if knn_dev is not None:
        from saul.lsi.knn import choose_lambda
        lam = choose_lambda(sigmoid(current), knn_dev, dev_labels, half)
        pieces.append({"piece": "D4", **lam})
        keep_knn = bool(lam["keep"])
        if keep_knn:
            probs_now = sigmoid(current)
            mixed = np.zeros_like(probs_now)
            for fit_name, score_name in (("A", "B"), ("B", "A")):
                rows = half == score_name
                value = float(lam["by_half"][fit_name])
                mixed[rows] = (1.0 - value) * probs_now[rows] + value * knn_dev[rows]
            current = _logit(mixed)
            if knn_test is not None:
                value = float(lam["lambda"])
                test_current = _logit((1.0 - value) * sigmoid(test_current) + value * knn_test)
    probs = sigmoid(current)
    kappa_info = choose_kappa(probs, dev_labels, half)
    kappa = kappa_info["kappa"]
    pieces.append({
        "piece": "D1", "kappa": kappa, "gain": kappa_info["gain"],
        "half_gain": kappa_info["half_gain"], "cross": kappa_info["tuned"],
        "insample": kappa_info["insample"],
    })
    thresholds = final_thresholds(probs, dev_labels, kappa)
    d2 = d2_gain(probs, dev_labels, half, kappa)
    pieces.append({"piece": "D2", "mode": d2["mode"], "gain": max(d2["gain_one"], d2["gain_two"])})
    dups = duplicate_dev or [None] * len(dev_ids)
    d5 = choose_d5(apply_thresholds(probs, thresholds), dev_labels, dups, half)
    pieces.append({"piece": "D5", "mode": d5["mode"], "gain": d5["gain"], "n": d5["n"]})
    dev_pred = np.zeros_like(probs, dtype=bool)
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        state = kappa_info["states"][fit_name]
        thr = shrink_thresholds(state["raw"], state["global"], state["counts"], kappa)
        rows = np.flatnonzero(half == score_name)
        row_dups = [dups[i] for i in rows]
        dev_pred[rows] = predict_matrix(probs[rows], thr, d2["mode"], row_dups, d5["mode"])
    report = {
        "pieces": pieces,
        "kappa": kappa,
        "d2": d2["mode"],
        "d3": keep_stack,
        "d4": keep_knn,
        "d5": d5["mode"],
        "dev": scores(dev_labels, dev_pred),
    }
    test_probs = sigmoid(test_current)
    test_pred = predict_matrix(test_probs, thresholds, d2["mode"], duplicate_test, d5["mode"])
    directory = exp_dir(name)
    os.makedirs(directory, exist_ok=True)
    write_json(report, os.path.join(directory, "decide.json"))
    np.savez(os.path.join(directory, "dev_pred.npz"), ids=np.asarray(dev_ids), pred=dev_pred.astype(np.uint8))
    np.savez(os.path.join(directory, "test_pred.npz"), ids=np.asarray(test_ids), pred=test_pred.astype(np.uint8))
    print(json.dumps(report["dev"]), flush=True)
    return report


def _knn_matrix(exp: str, kind: str, order: list[str]) -> np.ndarray | None:
    path = os.path.join(exp_dir(exp), f"{kind}_knn.npz")
    if not os.path.exists(path):
        return None
    blob = np.load(path, allow_pickle=True)
    index = {str(i): row for row, i in enumerate(blob["ids"])}
    return np.asarray(blob["probs"], dtype=np.float64)[[index[i] for i in order]]


def main() -> None:
    import argparse

    from saul.lsi.data import load_raw, lookup_duplicates

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="mix")
    parser.add_argument("--name", default="decide")
    parser.add_argument("--knn-exp", default="")
    args = parser.parse_args()
    dev = load_logits(args.exp, "dev")
    test = load_logits(args.exp, "test")
    raw = load_raw()
    dev_rows = {row["id"]: row for row in raw["dev"]}
    test_rows = {row["id"]: row for row in raw["test"]}
    knn_exp = args.knn_exp or None
    run_decision(
        dev["logits"], dev["labels"], dev["ids"], test["logits"], test["ids"],
        knn_dev=_knn_matrix(knn_exp, "dev", dev["ids"]) if knn_exp else None,
        knn_test=_knn_matrix(knn_exp, "test", test["ids"]) if knn_exp else None,
        duplicate_dev=lookup_duplicates([dev_rows[i] for i in dev["ids"]], raw["train"]),
        duplicate_test=lookup_duplicates([test_rows[i] for i in test["ids"]], raw["train"], raw["dev"]),
        name=args.name,
    )


if __name__ == "__main__":
    main()
