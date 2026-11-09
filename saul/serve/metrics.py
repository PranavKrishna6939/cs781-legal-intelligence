"""Per-iteration counters. The HTTP /stats endpoint and the benchmark both read `snapshot`."""
from __future__ import annotations

import time


class Metrics:
    def __init__(self):
        self.prefill_tokens = 0
        self.generated_tokens = 0
        self.iterations = 0
        self.graph_hits = 0
        self.graph_captures = 0
        self.admits = 0
        self.finished = 0
        self.seconds = {"prefill": 0.0, "decode": 0.0, "sample": 0.0, "finish": 0.0}
        self.t0 = time.perf_counter()
        self.occupancy: list[int] = []

    def add(self, key: str, dt: float) -> None:
        self.seconds[key] = self.seconds.get(key, 0.0) + dt

    def snapshot(self, extra: dict | None = None) -> dict:
        out = {
            "prefill_tokens": self.prefill_tokens,
            "generated_tokens": self.generated_tokens,
            "iterations": self.iterations,
            "graph_hits": self.graph_hits,
            "graph_captures": self.graph_captures,
            "admits": self.admits,
            "finished": self.finished,
            "seconds": dict(self.seconds),
            "wall_s": time.perf_counter() - self.t0,
            "mean_occupancy": (sum(self.occupancy) / len(self.occupancy)) if self.occupancy else 0.0,
        }
        if extra:
            out.update(extra)
        return out
