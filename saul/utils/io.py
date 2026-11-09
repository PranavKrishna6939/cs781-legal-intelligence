"""Small I/O helpers: gzip JSONL shards, JSON, run directories."""
from __future__ import annotations

import datetime as _dt
import glob
import gzip
import json
import os
import platform
import subprocess
import sys
from typing import Any, Iterable, Iterator


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(obj: Any, path: str, indent: int | None = 2) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ShardWriter:
    """Writes records into numbered gzip JSONL shards of at most `max_records` each."""

    def __init__(self, out_dir: str, prefix: str = "shard", max_records: int = 50_000):
        self.out_dir = ensure_dir(out_dir)
        self.prefix = prefix
        self.max_records = max_records
        self._idx = 0
        self._n = 0
        self._f = None
        self.total = 0

    def _open(self) -> None:
        path = os.path.join(self.out_dir, f"{self.prefix}_{self._idx:05d}.jsonl.gz")
        self._f = gzip.open(path, "wt", encoding="utf-8", compresslevel=3)

    def write(self, rec: dict) -> None:
        if self._f is None or self._n >= self.max_records:
            if self._f is not None:
                self._f.close()
                self._idx += 1
            self._open()
            self._n = 0
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._n += 1
        self.total += 1

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def list_shards(d: str, prefix: str = "shard") -> list[str]:
    return sorted(glob.glob(os.path.join(d, f"{prefix}_*.jsonl.gz")))


def iter_shards(d: str, prefix: str = "shard") -> Iterator[dict]:
    for p in list_shards(d, prefix):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def iter_jsonl(paths: Iterable[str]) -> Iterator[dict]:
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    info: dict[str, Any] = {
        "time": _dt.datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    return info


def init_run_dir(run_dir: str, cfg: dict) -> str:
    from saul.utils.config import save_config

    ensure_dir(run_dir)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "git_sha.txt"), "w") as f:
        f.write(git_sha() + "\n")
    write_json(env_info(), os.path.join(run_dir, "env.json"))
    return run_dir


class MetricsLogger:
    """Appends metrics to run_dir/metrics.jsonl and optionally to Weights & Biases."""

    def __init__(self, run_dir: str, wandb_cfg: dict | None = None, full_cfg: dict | None = None):
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self._wandb = None
        wandb_cfg = wandb_cfg or {}
        if wandb_cfg.get("enabled", False):
            try:
                import wandb

                parts = os.path.normpath(run_dir).split(os.sep)
                self._wandb = wandb.init(
                    project=wandb_cfg.get("project", "saul-phase1"),
                    name=wandb_cfg.get("name") or "-".join(parts[-2:]),
                    group=wandb_cfg.get("group"),
                    config=full_cfg or {},
                    dir=run_dir,
                    mode=wandb_cfg.get("mode", "online"),
                    resume="allow",
                    id=wandb_cfg.get("id"),
                )
            except Exception as e:  # never let tracking kill a 12h run
                print(f"[metrics] wandb disabled: {e}")
                self._wandb = None

    def log(self, metrics: dict, step: int) -> None:
        rec = {"step": int(step), **{k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=int(step))

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
