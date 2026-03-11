from __future__ import annotations

from prism_ssl.train.metrics import RunAccumulator


def test_run_accumulator_uses_elapsed_wall_time() -> None:
    accum = RunAccumulator()

    first = accum.update_step(
        elapsed_wall_s=2.0,
        patch_views_this_step=16,
        sample_views_this_step=2,
        replacement_completed_delta=0,
        replacement_failed_delta=0,
        replacement_wait_ms_delta=0.0,
        attempted_series_delta=0,
        broken_series_delta=0,
        loaded_series_delta=0,
        loaded_with_body_delta=0,
        sampled_body_center_views_delta=0,
    )
    second = accum.update_step(
        elapsed_wall_s=5.0,
        patch_views_this_step=24,
        sample_views_this_step=2,
        replacement_completed_delta=0,
        replacement_failed_delta=0,
        replacement_wait_ms_delta=0.0,
        attempted_series_delta=0,
        broken_series_delta=0,
        loaded_series_delta=0,
        loaded_with_body_delta=0,
        sampled_body_center_views_delta=0,
    )

    assert first == 8.0
    assert second == 8.0
    assert accum.throughput_effective_patches_per_sec == 8.0
