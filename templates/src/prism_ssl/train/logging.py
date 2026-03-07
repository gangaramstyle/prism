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
    distance_acc_shared: float,
    window_acc_wc: float,
    window_acc_ww: float,
    loss_supcon: float,
    loss_mim: float,
    supcon_weight: float,
    step_time_ms: float,
    data_wait_ms: float = 0.0,
    gpu_time_ms: float = 0.0,
    gpu_mem_peak_mb: float = 0.0,
    gpu_mem_reserved_mb: float = 0.0,
    post_step_ms: float = 0.0,
    step_throughput: float = 0.0,
    throughput_effective: float,
    broken_ratio: float,
    ts_loaded_ratio: float = 0.0,
    ts_view_ratio: float = 0.0,
) -> str:
    return (
        f"[train] step={step} "
        f"loss={sci(total_loss)} dist={sci(loss_distance)} dist_acc={distance_acc:.3f} "
        f"shared_acc={distance_acc_shared:.3f} wc_acc={window_acc_wc:.3f} ww_acc={window_acc_ww:.3f} "
        f"c={sci(loss_supcon)} mim={sci(loss_mim)} w_supcon={supcon_weight:.4f} "
        f"ts_scan_ratio={ts_loaded_ratio:.3f} ts_view_ratio={ts_view_ratio:.3f} "
        f"step_ms={step_time_ms:.1f} data_ms={data_wait_ms:.1f} gpu_ms={gpu_time_ms:.1f} "
        f"gpu_mem_mb={gpu_mem_peak_mb:.0f} gpu_res_mb={gpu_mem_reserved_mb:.0f} "
        f"post_ms={post_step_ms:.1f} step_tput={step_throughput:.1f} "
        f"eff_tput={throughput_effective:.1f} broken_ratio={broken_ratio:.4f}"
    )
