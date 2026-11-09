"""YAML configuration loading with `_base_` inheritance and dotted CLI overrides.

    cfg = load_config("configs/train/dense_286m.yaml", overrides=["optim.lr=2e-3", "train.total_tokens=4.5e9"])

* `_base_: path` (or list of paths) is resolved relative to the including file and deep-merged first.
* Overrides use `a.b.c=value`; the value is parsed as YAML so numbers/bools/lists work.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml


class AttrDict(dict):
    """dict with attribute access (recursively applied by `to_attrdict`)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:  # pragma: no cover - mirrors attribute semantics
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return AttrDict(copy.deepcopy(dict(self), memo))


def to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(v) for v in obj]
    return obj


def to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    bases = data.pop("_base_", None)
    if bases is None:
        return data
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        bpath = b if os.path.isabs(b) else os.path.normpath(os.path.join(os.path.dirname(path), b))
        merged = deep_merge(merged, _load_yaml(bpath))
    return deep_merge(merged, data)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return cfg


def load_config(path: str, overrides: Iterable[str] | None = None) -> AttrDict:
    return to_attrdict(apply_overrides(_load_yaml(path), overrides))


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
