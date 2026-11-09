"""Equal-weight mixes and the member rule."""
from __future__ import annotations

import json
import os

import numpy as np

from saul.lsi import exp_dir
from saul.lsi.const import DECISION_GAIN, MEMBER_MARGIN, STACK_GAIN
from saul.lsi.decide import load_logits, quick_macro
from saul.lsi.score import macro_f1, sigmoid
from saul.utils.io import write_json


def member_scores(members: dict[str, np.ndarray], labels: np.ndarray, half) -> dict[str, float]:
    return {name: quick_macro(logits, labels, half)["macro_f1"] for name, logits in members.items()}


def select_members(members: dict[str, np.ndarray], labels: np.ndarray, half,
                   margin: float = MEMBER_MARGIN) -> list[str]:
    """Within `margin` of the best, and not lowering the equal-weight mix in leave-one-out."""
    scores = member_scores(members, labels, half)
    if not scores:
        return []
    best = max(scores.values())
    cands = [name for name, score in scores.items() if score >= best - margin]
    if len(cands) <= 1:
        return cands
    full = np.mean([members[name] for name in cands], axis=0)
    full_score = quick_macro(full, labels, half)["macro_f1"]
    kept = []
    for name in cands:
        others = [other for other in cands if other != name]
        without = np.mean([members[other] for other in others], axis=0)
        if quick_macro(without, labels, half)["macro_f1"] <= full_score + 1e-9:
            kept.append(name)
    return kept or cands[:1]


def equal_mix(members: list[np.ndarray]) -> np.ndarray:
    return np.mean(members, axis=0)


def fit_weights(member_probs: list[np.ndarray], labels: np.ndarray, steps: int = 80) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    stacked = torch.tensor(np.stack(member_probs), dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.float32)
    raw = torch.zeros(len(member_probs), requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.1)
    for _ in range(steps):
        weight = torch.softmax(raw, dim=0)
        mixed = (weight[:, None, None] * stacked).sum(0)
        loss = F.binary_cross_entropy(mixed.clamp(1e-4, 1 - 1e-4), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    weight = torch.softmax(raw.detach(), dim=0).numpy()
    return weight


def cross_fitted_weights(members: list[np.ndarray], labels: np.ndarray, half) -> dict:
    half = np.asarray(half)
    probs = [sigmoid(logits) for logits in members]
    equal = sigmoid(equal_mix(members))
    gains = []
    weights = {}
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit, score = half == fit_name, half == score_name
        weight = fit_weights([p[fit] for p in probs], labels[fit])
        weights[fit_name] = weight
        mixed = sum(w * p[score] for w, p in zip(weight, probs))
        gains.append(macro_f1(labels[score], mixed >= 0.5) - macro_f1(labels[score], equal[score] >= 0.5))
    gain = float(np.mean(gains))
    keep = gain >= STACK_GAIN
    final = fit_weights(probs, labels) if keep else np.full(len(members), 1.0 / len(members))
    return {
        "keep": keep,
        "gain": gain,
        "weights": [float(w) for w in final],
        "by_half": {name: [float(w) for w in value] for name, value in weights.items()},
    }


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def mix_learned(members: list[np.ndarray], weight_fit: dict, half=None) -> np.ndarray:
    """Out-of-fold weights on dev. The all-dev weights are for test, where `half` is omitted."""
    probs = [sigmoid(logits) for logits in members]
    if half is not None:
        mixed = np.zeros_like(probs[0])
        half = np.asarray(half)
        for fit_name, score_name in (("A", "B"), ("B", "A")):
            rows = half == score_name
            mix = sum(w * p[rows] for w, p in zip(weight_fit["by_half"][fit_name], probs))
            mixed[rows] = mix
        return _logit(mixed)
    mixed = sum(w * p for w, p in zip(weight_fit["weights"], probs))
    return _logit(mixed)


def _scale_member(matrix: np.ndarray, labels: np.ndarray | None, half, temperature: float | None,
                  cached: dict | None) -> tuple[np.ndarray, float]:
    """Per-member temperature. Dev halves use the temperature fit on the other half."""
    from saul.lsi.decide import choose_temperature
    if cached is not None:
        return matrix / float(cached), float(cached)
    if labels is None or half is None:
        return matrix, 1.0
    info = choose_temperature(matrix, labels, half)
    scaled = np.array(matrix, dtype=np.float64, copy=True)
    half = np.asarray(half)
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        rows = half == score_name
        scaled[rows] = matrix[rows] / info["by_half"][fit_name]
    if temperature is not None:
        return matrix / float(temperature), float(temperature)
    return scaled, float(info["T"])


def build_mix(names: list[str], order_ids: list[str], kind: str, labels: np.ndarray | None,
              half, out_name: str, temperatures: dict | None = None,
              weight_fit: dict | None = None) -> dict:
    from saul.lsi.decide import align
    raw = [align(name, kind, order_ids) for name in names]
    temps = {}
    logits = []
    for name, matrix in zip(names, raw):
        cached = None
        if temperatures and name in temperatures:
            cached = float(temperatures[name])
        scaled, temperature = _scale_member(matrix, labels, half, None, cached)
        temps[name] = temperature
        logits.append(scaled)
    fit = weight_fit
    if fit is None and labels is not None and half is not None and len(logits) >= 2:
        fit = cross_fitted_weights(logits, labels, half)
    if fit and fit.get("keep"):
        mixed = mix_learned(logits, fit, half if labels is not None else None)
    else:
        mixed = equal_mix(logits)
    directory = exp_dir(out_name)
    os.makedirs(directory, exist_ok=True)
    payload = {"ids": np.asarray(order_ids), "logits": mixed.astype(np.float32)}
    if labels is not None:
        payload["labels"] = labels
    np.savez(os.path.join(directory, f"{kind}_logits.npz"), **payload)
    info = {"members": names, "kind": kind, "temperature": temps, "out": out_name}
    if labels is not None and half is not None:
        info["macro_f1"] = quick_macro(mixed, labels, half)["macro_f1"]
        info["weights"] = fit
    return info


def save_mix_report(name: str, info: dict) -> None:
    path = os.path.join(exp_dir(name), "mix.json")
    previous = {}
    if os.path.exists(path):
        import json
        previous = json.load(open(path, encoding="utf-8"))
    previous.update(info)
    write_json(previous, path)


def _load_member(exp: str, kind: str):
    return load_logits(exp, kind)


# Subset probes and reserve controls are not ensemble members. Fold files are assembled first.
_SKIP = {"P1", "P2", "P3", "P4", "P5", "P6", "A1probe", "A9gold", "A13"}


def discover_members(root: str | None = None) -> list[str]:
    import glob
    root = root or os.path.join(os.environ.get("SAUL_P5_DIR", "/data/cs781-p5"), "exp")
    names = []
    for path in sorted(glob.glob(os.path.join(root, "*", "dev_logits.npz"))):
        name = os.path.basename(os.path.dirname(path))
        if name in _SKIP or name.startswith(("smoke", "bench", "mix")):
            continue
        names.append(name)
    return names


def apply_member_rules(members: dict[str, np.ndarray], labels: np.ndarray, half) -> list[str]:
    """0.06 margin and leave-one-out, plus the CF2 replacement rule and the A7 +0.3 bar."""
    scores = member_scores(members, labels, half)
    names = list(members)
    dense = [name for name in ("A1", "A1t") if name in scores]
    if "A4" in scores and dense:
        reference = max(scores[name] for name in dense)
        if scores["A4"] < reference - 0.005:
            names = [name for name in names if name != "A4"]
        elif scores["A4"] >= reference:
            names = [name for name in names if name not in dense]
    pool = {name: members[name] for name in names}
    kept = select_members(pool, labels, half)
    if "A7" in kept and len(kept) > 1:
        others = [name for name in kept if name != "A7"]
        full = quick_macro(equal_mix([pool[name] for name in kept]), labels, half)["macro_f1"]
        without = quick_macro(equal_mix([pool[name] for name in others]), labels, half)["macro_f1"]
        if full - without < DECISION_GAIN:
            kept = others
    return kept


def assemble_cf2(folds: tuple[str, str] = ("A4A", "A4B"), out: str = "A4") -> bool:
    """Concatenate the two out-of-fold dev files and average the two test logit files.

    The fold tagged A was trained on dev half A, so its file holds predictions for half B.
    If the test set's predicted label count at 0.5 differs from the dev out-of-fold count by
    more than 5%, one temperature is chosen so the counts match. It never reads test labels.
    """
    dev_parts = []
    test_parts = []
    for exp, tag in zip(folds, ("A", "B")):
        dev_path = os.path.join(exp_dir(exp), f"dev_{tag}_oof_logits.npz")
        test_path = os.path.join(exp_dir(exp), "test_logits.npz")
        if not (os.path.exists(dev_path) and os.path.exists(test_path)):
            print(f"[mix] cf2 {exp} not ready", flush=True)
            return False
        dev_parts.append(np.load(dev_path, allow_pickle=True))
        test_parts.append(np.load(test_path, allow_pickle=True))
    dev_ids = [str(i) for part in dev_parts for i in part["ids"]]
    dev_logits = np.concatenate([np.asarray(part["logits"], dtype=np.float64) for part in dev_parts])
    dev_labels = np.concatenate([np.asarray(part["labels"], dtype=np.float64) for part in dev_parts])
    order = [str(i) for i in test_parts[0]["ids"]]
    index = {case_id: i for i, case_id in enumerate(str(x) for x in test_parts[1]["ids"])}
    second = np.asarray(test_parts[1]["logits"], dtype=np.float64)[[index[i] for i in order]]
    test_logits = (np.asarray(test_parts[0]["logits"], dtype=np.float64) + second) / 2
    dev_count = float((sigmoid(dev_logits) >= 0.5).sum() / max(len(dev_logits), 1))
    temperature = 1.0
    test_count = float((sigmoid(test_logits) >= 0.5).sum() / max(len(test_logits), 1))
    if dev_count > 0 and abs(test_count - dev_count) / dev_count > 0.05:
        best_gap = abs(test_count - dev_count)
        for candidate in np.geomspace(0.5, 3.0, 25):
            count = float((sigmoid(test_logits / candidate) >= 0.5).sum() / len(test_logits))
            gap = abs(count - dev_count)
            if gap < best_gap:
                best_gap, temperature = gap, float(candidate)
        test_logits = test_logits / temperature
    directory = exp_dir(out)
    os.makedirs(directory, exist_ok=True)
    np.savez(
        os.path.join(directory, "dev_logits.npz"),
        ids=np.asarray(dev_ids), logits=dev_logits.astype(np.float32), labels=dev_labels.astype(np.float32),
    )
    np.savez(os.path.join(directory, "test_logits.npz"), ids=np.asarray(order), logits=test_logits.astype(np.float32))
    with open(os.path.join(directory, "cf2.json"), "w", encoding="utf-8") as handle:
        json.dump({"folds": list(folds), "temperature": temperature, "dev_count": dev_count}, handle)
    print(f"[mix] cf2 {out} temperature={temperature:.3f}", flush=True)
    return True


def main() -> None:
    """Build a dev/test mix from finished members, or assemble a CF2 pair."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--assemble-cf2", action="store_true")
    parser.add_argument("--folds", default="A4A,A4B")
    parser.add_argument("--tag", default="A4")
    parser.add_argument("--out", default="mix")
    args = parser.parse_args()
    if args.assemble_cf2:
        folds = tuple(part for part in args.folds.split(",") if part)
        assemble_cf2((folds[0], folds[1]), args.tag)
        return
    names = discover_members()
    if len(names) < 1:
        print("[mix] no members yet")
        return
    anchor = "A1" if "A1" in names else names[0]
    dev = load_logits(anchor, "dev")
    if "labels" not in dev:
        print("[mix] anchor has no dev labels")
        return
    from saul.lsi.data import dev_half_array
    labels = dev["labels"]
    half = dev_half_array(dev["ids"])
    members = {}
    for name in names:
        try:
            members[name] = align_safe(name, "dev", dev["ids"])
        except (KeyError, FileNotFoundError, OSError):
            continue
    kept = apply_member_rules(members, labels, half)
    if not kept:
        print("[mix] no member passed")
        return
    info = build_mix(kept, dev["ids"], "dev", labels, half, args.out)
    test = load_logits(kept[0], "test")
    build_mix(
        kept, test["ids"], "test", None, None, args.out,
        temperatures=info.get("temperature"), weight_fit=info.get("weights"),
    )
    save_mix_report(args.out, info)
    print(json_dumps({"out": args.out, "members": kept, "macro_f1": info.get("macro_f1")}))


def align_safe(name, kind, order):
    from saul.lsi.decide import align
    return align(name, kind, order)


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj)
