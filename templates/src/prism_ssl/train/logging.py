"""Logging helpers for training diagnostics."""

from __future__ import annotations


def sci(x: float) -> str:
    return f"{x:.3e}"


def format_step_log(
    *,
    step: int,
    total_loss: float,
    loss_distance: float,
    loss_rotation: float,
    loss_window: float,
    loss_supcon: float,
    supcon_weight: float,
    step_time_ms: float,
    throughput_effective: float,
    broken_ratio: float,
) -> str:
    return (
        f"[train] step={step} "
        f"loss={sci(total_loss)} d={sci(loss_distance)} r={sci(loss_rotation)} "
        f"w={sci(loss_window)} c={sci(loss_supcon)} w_supcon={supcon_weight:.4f} "
        f"step_ms={step_time_ms:.1f} eff_tput={throughput_effective:.1f} "
        f"broken_ratio={broken_ratio:.4f}"
    )
