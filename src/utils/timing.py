"""
timing.py - named stage timings for the developer diagnostics panel.

    t = Timings()
    with t.stage("land_cover_retrieval"):
        ...
    t.as_dict()   # {"land_cover_retrieval": 0.0123, ..., "total": ...} in seconds

Stages may repeat (e.g. one land-cover load per domain expansion); repeated
stages are summed and counted.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


class Timings:
    def __init__(self):
        self._t0 = time.perf_counter()
        self.seconds: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t)

    def add(self, name: str, seconds: float):
        self.seconds[name] = self.seconds.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + 1

    def merge(self, other: "Timings", prefix: str = ""):
        for k, v in other.seconds.items():
            self.add(prefix + k, v)

    def total(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> Dict[str, float]:
        out = {k: round(v, 4) for k, v in self.seconds.items()}
        out["total"] = round(self.total(), 4)
        return out
