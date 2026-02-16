"""Proxy metric computation for ablation ranking."""

from __future__ import annotations

from typing import Any

import polars as pl


def compute_proxy_quality_score(metrics: dict[str, Any]) -> float:
    """Compute a scalar proxy score (higher is better)."""
    loss = float(metrics.get("train/loss", metrics.get("loss", 0.0)))
    throughput = float(metrics.get("throughput_effective_patches_per_sec", 0.0))
    stall_2s = float(metrics.get("stall_steps_ge_2000ms", 0.0))
    stall_10s = float(metrics.get("stall_steps_ge_10000ms", 0.0))

    # Penalize loss and stalls, reward throughput.
    return float((-10.0 * loss) + (0.01 * throughput) - (0.5 * stall_2s) - (2.0 * stall_10s))


def add_proxy_quality_column(df: pl.DataFrame) -> pl.DataFrame:
    if len(df) == 0:
        return df
    return df.with_columns(
        (
            (-10.0 * pl.col("loss").fill_null(0.0))
            + (0.01 * pl.col("throughput_effective_patches_per_sec").fill_null(0.0))
            - (0.5 * pl.col("stall_steps_ge_2000ms").fill_null(0.0))
            - (2.0 * pl.col("stall_steps_ge_10000ms").fill_null(0.0))
        ).alias("proxy_quality_score")
    )
