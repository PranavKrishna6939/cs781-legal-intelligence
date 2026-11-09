"""FCFS admission. One request owns the lowest free block; its `n` samples share that block's prompt KV."""
from __future__ import annotations

import time
from bisect import insort
from dataclasses import dataclass, field

from saul.serve.sampling import SamplingParams


@dataclass
class RequestOutput:
    text: str
    candidates: list[str]
    finish_reasons: list[str]
    prompt_tokens: int
    gen_tokens: list[int]
    latency_s: float
    logits: list | None = None


@dataclass
class Request:
    rid: int
    text: str
    ids: list[int]
    params: SamplingParams
    arrival: float = field(default_factory=time.perf_counter)
    status: str = "WAITING"
    output: RequestOutput | None = None
    error: BaseException | None = None
    traces: list[list] | None = None
    s0: int = -1
    block: int = -1

    def __post_init__(self) -> None:
        if self.params.record_logits:
            self.traces = [[] for _ in range(self.params.n)]


class Scheduler:
    def __init__(self, n_blocks: int, block_size: int, token_budget: int, admit_batch: int, max_waiting: int = 4096):
        self.free = list(range(n_blocks))
        self.block_size = block_size
        self.token_budget = token_budget
        self.admit_batch = admit_batch
        self.max_waiting = max_waiting
        self.queue: list[Request] = []
        self.groups: dict[int, Request] = {}

    def submit(self, req: Request) -> None:
        if req.params.n > self.block_size:
            raise ValueError(f"n={req.params.n} exceeds block_size={self.block_size}")
        if len(self.queue) >= self.max_waiting:
            raise BufferError("request queue is full")
        self.queue.append(req)

    def admit(self, max_prompt: int) -> list[Request]:
        admitted = []
        tokens = 0
        while self.queue and self.free and len(admitted) < self.admit_batch:
            req = self.queue[0]
            length = len(req.ids)
            if length > max_prompt:
                raise ValueError(f"prompt has {length} tokens, max_prompt_len is {max_prompt}")
            if tokens + length > self.token_budget and admitted:
                break
            self.queue.pop(0)
            g = self.free.pop(0)
            req.block = g
            req.s0 = g * self.block_size
            req.status = "RUNNING"
            self.groups[g] = req
            admitted.append(req)
            tokens += length
        return admitted

    def free_block(self, g: int) -> None:
        self.groups.pop(g, None)
        insort(self.free, g)

    @property
    def busy(self) -> bool:
        return bool(self.queue or self.groups)
