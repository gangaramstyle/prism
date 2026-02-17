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
    rotation = torch.from_numpy(np.asarray(result["rotation_matrix_ras"], dtype=np.float32))
    prism_center_pt = torch.from_numpy(np.asarray(result["prism_center_pt"], dtype=np.float32).reshape(3))
    rotation_degrees = torch.from_numpy(np.asarray(result["rotation_degrees"], dtype=np.float32).reshape(3))
    window_params = torch.from_numpy(np.asarray([result["wc"], result["ww"]], dtype=np.float32))

    return {
        "patches": patches,
        "positions": positions,
        "rotation": rotation,
        "prism_center_pt": prism_center_pt,
        "rotation_degrees": rotation_degrees,
        "window_params": window_params,
    }


def compute_pair_targets(
    view_a: Mapping[str, torch.Tensor],
    view_b: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute pair labels used by regression heads."""
    center_delta_mm = view_b["prism_center_pt"] - view_a["prism_center_pt"]
    center_distance_mm = torch.linalg.norm(center_delta_mm, dim=0)
    rotation_delta_deg = view_b["rotation_degrees"] - view_a["rotation_degrees"]
    window_delta = view_b["window_params"] - view_a["window_params"]
    return {
        "center_delta_mm": center_delta_mm,
        "center_distance_mm": center_distance_mm,
        "rotation_delta_deg": rotation_delta_deg,
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
    replacement_requested: bool = False,
) -> dict[str, Any]:
    """Build one dataset sample dict matching the collate contract."""
    view_a = tensorize_sample_view(result_a)
    view_b = tensorize_sample_view(result_b)
    pair = compute_pair_targets(view_a, view_b)

    return {
        "patches": view_a["patches"],
        "positions": view_a["positions"],
        "rotation": view_a["rotation"],
        "prism_center_pt": view_a["prism_center_pt"],
        "rotation_degrees": view_a["rotation_degrees"],
        "window_params": view_a["window_params"],
        "patches_a": view_a["patches"],
        "positions_a": view_a["positions"],
        "rotation_a": view_a["rotation"],
        "prism_center_pt_a": view_a["prism_center_pt"],
        "rotation_degrees_a": view_a["rotation_degrees"],
        "window_params_a": view_a["window_params"],
        "patches_b": view_b["patches"],
        "positions_b": view_b["positions"],
        "rotation_b": view_b["rotation"],
        "prism_center_pt_b": view_b["prism_center_pt"],
        "rotation_degrees_b": view_b["rotation_degrees"],
        "window_params_b": view_b["window_params"],
        "center_delta_mm": pair["center_delta_mm"],
        "center_distance_mm": pair["center_distance_mm"],
        "rotation_delta_deg": pair["rotation_delta_deg"],
        "window_delta": pair["window_delta"],
        "scan_id": scan_id,
        "series_id": series_id,
        "replacement_completed_count_delta": int(replacement_completed_count_delta),
        "replacement_failed_count_delta": int(replacement_failed_count_delta),
        "replacement_wait_time_ms_delta": float(replacement_wait_time_ms_delta),
        "attempted_series_delta": int(attempted_series_delta),
        "broken_series_delta": int(broken_series_delta),
        "replacement_requested": bool(replacement_requested),
    }
