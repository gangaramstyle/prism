"""Batch collation utilities."""

from __future__ import annotations

from typing import Any

import torch


def _labels_for_keys(keys: list[str]) -> torch.Tensor:
    key_to_label: dict[str, int] = {}
    labels: list[int] = []
    for key in keys:
        if key not in key_to_label:
            key_to_label[key] = len(key_to_label)
        labels.append(key_to_label[key])
    return torch.tensor(labels, dtype=torch.long)


def collate_prism_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    series_ids = [str(b["series_id"]) for b in batch]
    protocol_keys = [str(b["protocol_key"]) for b in batch]

    return {
        "patches": torch.stack([b["patches"] for b in batch]),
        "positions": torch.stack([b["positions"] for b in batch]),
        "prism_center_pt": torch.stack([b["prism_center_pt"] for b in batch]),
        "window_params": torch.stack([b["window_params"] for b in batch]),
        "source_patch_mm": torch.stack([b["source_patch_mm"] for b in batch]),
        "patches_a": torch.stack([b["patches_a"] for b in batch]),
        "positions_a": torch.stack([b["positions_a"] for b in batch]),
        "prism_center_pt_a": torch.stack([b["prism_center_pt_a"] for b in batch]),
        "window_params_a": torch.stack([b["window_params_a"] for b in batch]),
        "source_patch_mm_a": torch.stack([b["source_patch_mm_a"] for b in batch]),
        "patches_b": torch.stack([b["patches_b"] for b in batch]),
        "positions_b": torch.stack([b["positions_b"] for b in batch]),
        "prism_center_pt_b": torch.stack([b["prism_center_pt_b"] for b in batch]),
        "window_params_b": torch.stack([b["window_params_b"] for b in batch]),
        "source_patch_mm_b": torch.stack([b["source_patch_mm_b"] for b in batch]),
        "center_delta_mm": torch.stack([b["center_delta_mm"] for b in batch]),
        "center_distance_mm": torch.stack([b["center_distance_mm"] for b in batch]),
        "window_delta": torch.stack([b["window_delta"] for b in batch]),
        "series_instance_label": _labels_for_keys(series_ids),
        "series_protocol_label": _labels_for_keys(protocol_keys),
        "series_id": series_ids,
        "protocol_key": protocol_keys,
        "scan_id": [str(b["scan_id"]) for b in batch],
        "replacement_completed_count_delta": int(sum(int(b.get("replacement_completed_count_delta", 0)) for b in batch)),
        "replacement_failed_count_delta": int(sum(int(b.get("replacement_failed_count_delta", 0)) for b in batch)),
        "replacement_wait_time_ms_delta": float(sum(float(b.get("replacement_wait_time_ms_delta", 0.0)) for b in batch)),
        "attempted_series_delta": int(sum(int(b.get("attempted_series_delta", 0)) for b in batch)),
        "broken_series_delta": int(sum(int(b.get("broken_series_delta", 0)) for b in batch)),
        "loaded_series_delta": int(sum(int(b.get("loaded_series_delta", 0)) for b in batch)),
        "loaded_with_body_delta": int(sum(int(b.get("loaded_with_body_delta", 0)) for b in batch)),
        "sampled_body_center_views_delta": int(
            sum(int(bool(b.get("sampled_body_center_a", False))) + int(bool(b.get("sampled_body_center_b", False))) for b in batch)
        ),
    }

