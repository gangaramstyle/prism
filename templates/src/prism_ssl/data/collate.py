"""Batch collation utilities."""

from __future__ import annotations

from typing import Any

import torch


def collate_prism_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    series_ids = [str(b["series_id"]) for b in batch]
    series_to_label: dict[str, int] = {}
    series_labels = []
    for sid in series_ids:
        if sid not in series_to_label:
            series_to_label[sid] = len(series_to_label)
        series_labels.append(series_to_label[sid])

    out: dict[str, Any] = {
        "patches": torch.stack([b["patches"] for b in batch]),
        "positions": torch.stack([b["positions"] for b in batch]),
        "rotation": torch.stack([b["rotation"] for b in batch]),
        "prism_center_pt": torch.stack([b["prism_center_pt"] for b in batch]),
        "rotation_degrees": torch.stack([b["rotation_degrees"] for b in batch]),
        "window_params": torch.stack([b["window_params"] for b in batch]),
        "patches_a": torch.stack([b["patches_a"] for b in batch]),
        "positions_a": torch.stack([b["positions_a"] for b in batch]),
        "rotation_a": torch.stack([b["rotation_a"] for b in batch]),
        "prism_center_pt_a": torch.stack([b["prism_center_pt_a"] for b in batch]),
        "rotation_degrees_a": torch.stack([b["rotation_degrees_a"] for b in batch]),
        "window_params_a": torch.stack([b["window_params_a"] for b in batch]),
        "patches_b": torch.stack([b["patches_b"] for b in batch]),
        "positions_b": torch.stack([b["positions_b"] for b in batch]),
        "rotation_b": torch.stack([b["rotation_b"] for b in batch]),
        "prism_center_pt_b": torch.stack([b["prism_center_pt_b"] for b in batch]),
        "rotation_degrees_b": torch.stack([b["rotation_degrees_b"] for b in batch]),
        "window_params_b": torch.stack([b["window_params_b"] for b in batch]),
        "center_delta_mm": torch.stack([b["center_delta_mm"] for b in batch]),
        "center_distance_mm": torch.stack([b["center_distance_mm"] for b in batch]),
        "rotation_delta_deg": torch.stack([b["rotation_delta_deg"] for b in batch]),
        "window_delta": torch.stack([b["window_delta"] for b in batch]),
        "series_label": torch.tensor(series_labels, dtype=torch.long),
        "series_id": series_ids,
        "scan_id": [str(b["scan_id"]) for b in batch],
        "replacement_completed_count_delta": int(sum(int(b.get("replacement_completed_count_delta", 0)) for b in batch)),
        "replacement_failed_count_delta": int(sum(int(b.get("replacement_failed_count_delta", 0)) for b in batch)),
        "replacement_wait_time_ms_delta": float(sum(float(b.get("replacement_wait_time_ms_delta", 0.0)) for b in batch)),
        "attempted_series_delta": int(sum(int(b.get("attempted_series_delta", 0)) for b in batch)),
        "broken_series_delta": int(sum(int(b.get("broken_series_delta", 0)) for b in batch)),
    }
    return out
