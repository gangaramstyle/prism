"""Logging helpers for training diagnostics."""

from __future__ import annotations


def sci(x: float) -> str:
    return f"{x:.3e}"


def format_step_log(
    *,
    step: int,
    total_loss: float,
    loss_direction: float,
    direction_acc: float,
    loss_window: float,
    loss_supcon: float,
    supcon_weight: float,
    step_time_ms: float,
    data_wait_ms: float = 0.0,
    throughput_effective: float,
    broken_ratio: float,
) -> str:
    gpu_ms = step_time_ms - data_wait_ms
    return (
        f"[train] step={step} "
        f"loss={sci(total_loss)} dir={sci(loss_direction)} dir_acc={direction_acc:.3f} "
        f"w={sci(loss_window)} c={sci(loss_supcon)} w_supcon={supcon_weight:.4f} "
        f"step_ms={step_time_ms:.1f} data_ms={data_wait_ms:.1f} gpu_ms={gpu_ms:.1f} "
        f"eff_tput={throughput_effective:.1f} broken_ratio={broken_ratio:.4f}"
    )
