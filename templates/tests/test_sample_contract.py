from __future__ import annotations

import numpy as np
import torch

from prism_ssl.data.collate import collate_prism_batch
from prism_ssl.data.preflight import NiftiScan
from prism_ssl.data.sample_contract import build_dataset_item, compute_pair_targets, tensorize_sample_view


def _make_scan(seed: int = 7) -> NiftiScan:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(48, 48, 12)).astype(np.float32)
    robust_low = float(np.percentile(data, 0.5))
    robust_high = float(np.percentile(data, 99.5))
    return NiftiScan(
        data=data,
        affine=np.eye(4, dtype=np.float32),
        spacing=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        modality="CT",
        base_patch_mm=16.0,
        robust_median=float(np.median(data)),
        robust_std=float(np.std(data)),
        robust_low=robust_low,
        robust_high=robust_high,
    )


def test_build_dataset_item_matches_collate_contract() -> None:
    scan = _make_scan()
    result_a = scan.train_sample(16, seed=101, method="optimized_fused", wc=0.0, ww=4.0)
    result_b = scan.train_sample(16, seed=202, method="optimized_fused", wc=0.0, ww=4.0)

    item = build_dataset_item(
        result_a=result_a,
        result_b=result_b,
        scan_id="scan_1",
        series_id="series_1",
    )
    batch = collate_prism_batch([item])

    assert tuple(batch["patches_a"].shape) == (1, 16, 16, 16, 1)
    assert tuple(batch["positions_a"].shape) == (1, 16, 3)
    assert tuple(batch["patches_b"].shape) == (1, 16, 16, 16, 1)
    assert tuple(batch["positions_b"].shape) == (1, 16, 3)
    assert tuple(batch["rotation_delta_deg"].shape) == (1, 3)
    assert tuple(batch["window_delta"].shape) == (1, 2)
    assert tuple(batch["center_distance_mm"].shape) == (1,)
    assert int(batch["series_label"][0].item()) == 0


def test_compute_pair_targets_matches_item_outputs() -> None:
    scan = _make_scan(seed=11)
    result_a = scan.train_sample(8, seed=5, method="optimized_fused", wc=0.0, ww=4.0)
    result_b = scan.train_sample(8, seed=6, method="optimized_fused", wc=0.0, ww=4.0)

    view_a = tensorize_sample_view(result_a)
    view_b = tensorize_sample_view(result_b)
    pair = compute_pair_targets(view_a, view_b)
    item = build_dataset_item(
        result_a=result_a,
        result_b=result_b,
        scan_id="scan_2",
        series_id="series_2",
    )

    assert torch.allclose(pair["center_delta_mm"], item["center_delta_mm"])
    assert torch.allclose(pair["center_distance_mm"], item["center_distance_mm"])
    assert torch.allclose(pair["rotation_delta_deg"], item["rotation_delta_deg"])
    assert torch.allclose(pair["window_delta"], item["window_delta"])
