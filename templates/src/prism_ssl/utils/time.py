"""Timing helpers and percentiles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DurationSummary:
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0


@dataclass
class StepTimeTracker:
    durations_ms: list[float] = field(default_factory=list)
    stall_steps_ge_2000ms: int = 0
    stall_steps_ge_10000ms: int = 0

    def add(self, duration_ms: float) -> None:
        duration_ms = float(duration_ms)
        self.durations_ms.append(duration_ms)
        if duration_ms >= 2000.0:
            self.stall_steps_ge_2000ms += 1
        if duration_ms >= 10000.0:
            self.stall_steps_ge_10000ms += 1

    def summary(self) -> DurationSummary:
        if not self.durations_ms:
            return DurationSummary()
        arr = np.asarray(self.durations_ms, dtype=np.float64)
        return DurationSummary(
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(arr.max()),
        )
