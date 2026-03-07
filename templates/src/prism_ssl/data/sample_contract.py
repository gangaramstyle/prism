"""Shared helpers to convert sampled views into training-contract tensors."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def tensorize_sample_view(result: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Convert a `train_sample` result to tensor fields used by the training loop."""
    patches_np = np.asarray(result["normalized_patches"], dtype=np.float32)
    if patches_np.ndim == 2:
        patches_np = patches_np[np.newaxis, ...]
    patches = torch.from_numpy(np.ascontiguousarray(patches_np[..., np.newaxis], dtype=np.float32))

    positions_np = np.asarray(result["relative_patch_centers_pt"], dtype=np.float32)
    positions = torch.from_numpy(np.ascontiguousarray(np.atleast_2d(positions_np), dtype=np.float32))
    prism_center_pt = torch.from_numpy(np.asarray(result["prism_center_pt"], dtype=np.float32).reshape(3))
    window_params = torch.from_numpy(np.asarray([result["wc"], result["ww"]], dtype=np.float32))

    return {
        "patches": patches,
        "positions": positions,
        "prism_center_pt": prism_center_pt,
        "window_params": window_params,
    }


def compute_pair_targets(
    view_a: Mapping[str, torch.Tensor],
    view_b: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute pair labels used by regression heads."""
    center_delta_mm = view_b["prism_center_pt"] - view_a["prism_center_pt"]
    center_distance_mm = torch.linalg.norm(center_delta_mm, dim=0)
    window_delta = view_b["window_params"] - view_a["window_params"]
    return {
        "center_delta_mm": center_delta_mm,
        "center_distance_mm": center_distance_mm,
        "window_delta": window_delta,
    }


def build_dataset_item(
    *,
    result_a: Mapping[str, Any],
    result_b: Mapping[str, Any],
    scan_id: str,
    series_id: str,
    replacement_completed_count_delta: int = 0,
    replacement_failed_count_delta: int = 0,
    replacement_wait_time_ms_delta: float = 0.0,
    attempted_series_delta: int = 0,
    broken_series_delta: int = 0,
    loaded_series_delta: int = 0,
    loaded_with_body_delta: int = 0,
    replacement_requested: bool = False,
) -> dict[str, Any]:
    """Build one dataset sample dict matching the collate contract."""
    view_a = tensorize_sample_view(result_a)
    view_b = tensorize_sample_view(result_b)
    pair = compute_pair_targets(view_a, view_b)

    return {
        "patches": view_a["patches"],
        "positions": view_a["positions"],
        "prism_center_pt": view_a["prism_center_pt"],
        "window_params": view_a["window_params"],
        "patches_a": view_a["patches"],
        "positions_a": view_a["positions"],
        "prism_center_pt_a": view_a["prism_center_pt"],
        "window_params_a": view_a["window_params"],
        "patches_b": view_b["patches"],
        "positions_b": view_b["positions"],
        "prism_center_pt_b": view_b["prism_center_pt"],
        "window_params_b": view_b["window_params"],
        "center_delta_mm": pair["center_delta_mm"],
        "center_distance_mm": pair["center_distance_mm"],
        "window_delta": pair["window_delta"],
        "scan_id": scan_id,
        "series_id": series_id,
        "replacement_completed_count_delta": int(replacement_completed_count_delta),
        "replacement_failed_count_delta": int(replacement_failed_count_delta),
        "replacement_wait_time_ms_delta": float(replacement_wait_time_ms_delta),
        "attempted_series_delta": int(attempted_series_delta),
        "broken_series_delta": int(broken_series_delta),
        "loaded_series_delta": int(loaded_series_delta),
        "loaded_with_body_delta": int(loaded_with_body_delta),
        "sampled_body_center_a": bool(result_a.get("sampled_body_center", False)),
        "sampled_body_center_b": bool(result_b.get("sampled_body_center", False)),
        "replacement_requested": bool(replacement_requested),
    }


def build_study4_dataset_item(
    *,
    result_a: Mapping[str, Any],
    result_ap: Mapping[str, Any],
    result_b: Mapping[str, Any],
    result_bp: Mapping[str, Any],
    study_id: str,
    series_id_x: str,
    series_id_y: str,
    cross_valid: bool,
    cross_mode: str,
    replacement_completed_count_delta: int = 0,
    replacement_failed_count_delta: int = 0,
    replacement_wait_time_ms_delta: float = 0.0,
    attempted_series_delta: int = 0,
    broken_series_delta: int = 0,
    loaded_series_delta: int = 0,
    loaded_with_body_delta: int = 0,
    replacement_requested: bool = False,
) -> dict[str, Any]:
    """Build one 4-view dataset sample dict for study-aware training."""
    view_a = tensorize_sample_view(result_a)
    view_ap = tensorize_sample_view(result_ap)
    view_b = tensorize_sample_view(result_b)
    view_bp = tensorize_sample_view(result_bp)
    pair_x = compute_pair_targets(view_a, view_b)
    pair_y = compute_pair_targets(view_ap, view_bp)

    patches_views = torch.stack(
        [view_a["patches"], view_ap["patches"], view_b["patches"], view_bp["patches"]],
        dim=0,
    )
    positions_views = torch.stack(
        [view_a["positions"], view_ap["positions"], view_b["positions"], view_bp["positions"]],
        dim=0,
    )
    prism_center_pt_views = torch.stack(
        [
            view_a["prism_center_pt"],
            view_ap["prism_center_pt"],
            view_b["prism_center_pt"],
            view_bp["prism_center_pt"],
        ],
        dim=0,
    )
    window_params_views = torch.stack(
        [
            view_a["window_params"],
            view_ap["window_params"],
            view_b["window_params"],
            view_bp["window_params"],
        ],
        dim=0,
    )

    sampled_body_center_flags = [
        bool(result_a.get("sampled_body_center", False)),
        bool(result_ap.get("sampled_body_center", False)),
        bool(result_b.get("sampled_body_center", False)),
        bool(result_bp.get("sampled_body_center", False)),
    ]

    return {
        "patches_views": patches_views,
        "positions_views": positions_views,
        "prism_center_pt_views": prism_center_pt_views,
        "window_params_views": window_params_views,
        "center_delta_mm_x": pair_x["center_delta_mm"],
        "center_distance_mm_x": pair_x["center_distance_mm"],
        "window_delta_x": pair_x["window_delta"],
        "center_delta_mm_y": pair_y["center_delta_mm"],
        "center_distance_mm_y": pair_y["center_distance_mm"],
        "window_delta_y": pair_y["window_delta"],
        "study_id": study_id,
        "series_id_x": series_id_x,
        "series_id_y": series_id_y,
        "cross_valid": bool(cross_valid),
        "cross_mode": str(cross_mode),
        "replacement_completed_count_delta": int(replacement_completed_count_delta),
        "replacement_failed_count_delta": int(replacement_failed_count_delta),
        "replacement_wait_time_ms_delta": float(replacement_wait_time_ms_delta),
        "attempted_series_delta": int(attempted_series_delta),
        "broken_series_delta": int(broken_series_delta),
        "loaded_series_delta": int(loaded_series_delta),
        "loaded_with_body_delta": int(loaded_with_body_delta),
        "sampled_body_center_views_delta": int(sum(int(flag) for flag in sampled_body_center_flags)),
        "replacement_requested": bool(replacement_requested),
    }
