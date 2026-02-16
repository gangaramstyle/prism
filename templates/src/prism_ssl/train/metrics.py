"""Training metric accumulation."""

from __future__ import annotations

from dataclasses import dataclass

from prism_ssl.utils.time import StepTimeTracker


@dataclass
class RunAccumulator:
    running_wall_s: float = 0.0
    running_patches: int = 0
    replacement_completed_count: int = 0
    replacement_failed_count: int = 0
    replacement_wait_time_ms_total: float = 0.0
    attempted_series_count: int = 0
    broken_series_count: int = 0

    def update_step(
        self,
        *,
        step_time_s: float,
        patches_this_step: int,
        replacement_completed_delta: int,
        replacement_failed_delta: int,
        replacement_wait_ms_delta: float,
        attempted_series_delta: int,
        broken_series_delta: int,
    ) -> float:
        self.running_wall_s += float(step_time_s)
        self.running_patches += int(patches_this_step)
        self.replacement_completed_count += int(replacement_completed_delta)
        self.replacement_failed_count += int(replacement_failed_delta)
        self.replacement_wait_time_ms_total += float(replacement_wait_ms_delta)
        self.attempted_series_count += int(attempted_series_delta)
        self.broken_series_count += int(broken_series_delta)
        return self.throughput_effective_patches_per_sec

    @property
    def throughput_effective_patches_per_sec(self) -> float:
        if self.running_wall_s <= 0:
            return 0.0
        return self.running_patches / self.running_wall_s

    @property
    def broken_ratio(self) -> float:
        if self.attempted_series_count <= 0:
            return 0.0
        return self.broken_series_count / float(self.attempted_series_count)


def build_tail_metrics(step_times: StepTimeTracker) -> dict[str, float]:
    summary = step_times.summary()
    return {
        "total_step_time_ms_mean": summary.mean_ms,
        "total_step_time_ms_p50": summary.p50_ms,
        "total_step_time_ms_p95": summary.p95_ms,
        "total_step_time_ms_p99": summary.p99_ms,
        "total_step_time_ms_max": summary.max_ms,
        "stall_steps_ge_2000ms": float(step_times.stall_steps_ge_2000ms),
        "stall_steps_ge_10000ms": float(step_times.stall_steps_ge_10000ms),
    }
