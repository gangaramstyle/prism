"""Logging helpers for training diagnostics."""

from __future__ import annotations


def sci(x: float) -> str:
    return f"{x:.3e}"


def format_step_log(
    *,
    step: int,
    total_loss: float,
    loss_distance: float,
    distance_acc: float,
    loss_supcon: float,
    loss_mim: float,
    supcon_weight: float,
    step_time_ms: float,
    data_wait_ms: float = 0.0,
    gpu_time_ms: float = 0.0,
    post_step_ms: float = 0.0,
    step_throughput: float = 0.0,
    throughput_effective: float,
    broken_ratio: float,
) -> str:
    return (
        f"[train] step={step} "
        f"loss={sci(total_loss)} dist={sci(loss_distance)} dist_acc={distance_acc:.3f} "
        f"c={sci(loss_supcon)} mim={sci(loss_mim)} w_supcon={supcon_weight:.4f} "
        f"step_ms={step_time_ms:.1f} data_ms={data_wait_ms:.1f} gpu_ms={gpu_time_ms:.1f} "
        f"post_ms={post_step_ms:.1f} step_tput={step_throughput:.1f} "
        f"eff_tput={throughput_effective:.1f} broken_ratio={broken_ratio:.4f}"
    )
