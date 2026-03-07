"""Training metric accumulation."""

from __future__ import annotations

from dataclasses import dataclass

from prism_ssl.utils.time import StepTimeTracker


@dataclass
class RunAccumulator:
    running_wall_s: float = 0.0
    running_patch_views: int = 0
    running_sample_views: int = 0
    replacement_completed_count: int = 0
    replacement_failed_count: int = 0
    replacement_wait_time_ms_total: float = 0.0
    attempted_series_count: int = 0
    broken_series_count: int = 0
    loaded_series_count: int = 0
    loaded_with_body_series_count: int = 0
    sampled_body_center_view_count: int = 0

    def update_step(
        self,
        *,
        elapsed_wall_s: float,
        patch_views_this_step: int,
        sample_views_this_step: int,
        replacement_completed_delta: int,
        replacement_failed_delta: int,
        replacement_wait_ms_delta: float,
        attempted_series_delta: int,
        broken_series_delta: int,
        loaded_series_delta: int,
        loaded_with_body_delta: int,
        sampled_body_center_views_delta: int,
    ) -> float:
        self.running_wall_s = max(self.running_wall_s, float(elapsed_wall_s))
        self.running_patch_views += int(patch_views_this_step)
        self.running_sample_views += int(sample_views_this_step)
        self.replacement_completed_count += int(replacement_completed_delta)
        self.replacement_failed_count += int(replacement_failed_delta)
        self.replacement_wait_time_ms_total += float(replacement_wait_ms_delta)
        self.attempted_series_count += int(attempted_series_delta)
        self.broken_series_count += int(broken_series_delta)
        self.loaded_series_count += int(loaded_series_delta)
        self.loaded_with_body_series_count += int(loaded_with_body_delta)
        self.sampled_body_center_view_count += int(sampled_body_center_views_delta)
        return self.throughput_effective_patches_per_sec

    @property
    def throughput_effective_patches_per_sec(self) -> float:
        if self.running_wall_s <= 0:
            return 0.0
        return self.running_patch_views / self.running_wall_s

    @property
    def broken_ratio(self) -> float:
        if self.attempted_series_count <= 0:
            return 0.0
        return self.broken_series_count / float(self.attempted_series_count)

    @property
    def loaded_with_body_ratio(self) -> float:
        if self.loaded_series_count <= 0:
            return 0.0
        return self.loaded_with_body_series_count / float(self.loaded_series_count)

    @property
    def sampled_body_center_view_ratio(self) -> float:
        if self.running_sample_views <= 0:
            return 0.0
        return self.sampled_body_center_view_count / float(self.running_sample_views)


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
