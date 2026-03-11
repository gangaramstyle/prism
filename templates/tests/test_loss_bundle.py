from __future__ import annotations

import torch

from prism_ssl.config.schema import LossConfig
from prism_ssl.model import PrismModelOutput, compute_loss_bundle


def test_loss_bundle_reports_center_distance_and_low_variation_metrics() -> None:
    center_delta = torch.tensor(
        [
            [3.0, 4.0, 0.0],
            [0.0, 6.0, 8.0],
        ],
        dtype=torch.float32,
    )
    center_distance = torch.linalg.norm(center_delta, dim=1)

    flat_view = torch.zeros((1, 2, 16, 16, 1), dtype=torch.float32)
    textured_view = torch.linspace(-1.0, 1.0, steps=2 * 16 * 16, dtype=torch.float32).reshape(1, 2, 16, 16, 1)

    batch = {
        "center_delta_mm": center_delta,
        "center_distance_mm": center_distance,
        "window_delta": torch.tensor([[1.0, -1.0], [-0.5, 0.5]], dtype=torch.float32),
        "series_instance_label": torch.tensor([0, 1], dtype=torch.long),
        "series_protocol_label": torch.tensor([0, 0], dtype=torch.long),
        "source_patch_mm_a": torch.tensor([16.0, 32.0], dtype=torch.float32),
        "source_patch_mm_b": torch.tensor([24.0, 48.0], dtype=torch.float32),
        "patches_a": torch.cat([flat_view, textured_view], dim=0),
        "patches_b": torch.cat([textured_view, textured_view], dim=0),
    }

    outputs = PrismModelOutput(
        pair_relation_logits=torch.zeros((2, 5), dtype=torch.float32),
        proj_instance_a=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        proj_instance_b=torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32),
        proj_protocol_a=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        proj_protocol_b=torch.tensor([[0.9, 0.1], [0.9, 0.1]], dtype=torch.float32),
        patch_size_pred_a=torch.log2(batch["source_patch_mm_a"]),
        patch_size_pred_b=torch.log2(batch["source_patch_mm_b"]),
        mim_pred_a=None,
        mim_pred_b=None,
        mim_target_a=None,
        mim_target_b=None,
    )

    loss_bundle, diagnostics = compute_loss_bundle(outputs, batch, LossConfig(), step=5000)

    assert torch.isfinite(loss_bundle.total)
    assert diagnostics["center_distance_mm_mean"] == 7.5
    assert diagnostics["center_distance_mm_min"] == 5.0
    assert diagnostics["center_distance_mm_max"] == 10.0
    assert diagnostics["low_variation_sample_count"] == 1.0
    assert diagnostics["low_variation_sample_ratio"] == 0.5
    assert diagnostics["low_variation_view_count"] == 1.0
    assert diagnostics["low_variation_both_views_count"] == 0.0
