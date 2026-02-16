"""Training schedules."""

from __future__ import annotations


def supcon_weight(step: int, warmup: int, ramp: int, target: float) -> float:
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return float(target)
    t = min(1.0, (step - warmup) / float(ramp))
    return float(target) * t
