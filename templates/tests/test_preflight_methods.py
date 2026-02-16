from __future__ import annotations

import numpy as np
import pytest

from prism_ssl.data.preflight import NiftiScan


def _make_scan(seed: int = 123) -> NiftiScan:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(64, 64, 16)).astype(np.float32)
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


def test_train_sample_methods_are_active_and_shape_stable() -> None:
    scan = _make_scan()

    legacy = scan.train_sample(32, seed=77, method="legacy_loop", wc=0.0, ww=4.0)
    fused = scan.train_sample(32, seed=77, method="optimized_fused", wc=0.0, ww=4.0)
    alias = scan.train_sample(32, seed=77, method="optimized_patch", wc=0.0, ww=4.0)

    assert legacy["method"] == "legacy_loop"
    assert fused["method"] == "optimized_fused"
    assert alias["method"] == "optimized_fused"

    assert legacy["normalized_patches"].shape == (32, 16, 16)
    assert fused["normalized_patches"].shape == (32, 16, 16)
    np.testing.assert_array_equal(legacy["patch_centers_pt"], fused["patch_centers_pt"])
    np.testing.assert_allclose(alias["normalized_patches"], fused["normalized_patches"], rtol=0.0, atol=1e-6)

    mean_abs_delta = float(np.mean(np.abs(legacy["normalized_patches"] - fused["normalized_patches"])))
    assert mean_abs_delta < 0.5


def test_unknown_train_sample_method_raises() -> None:
    scan = _make_scan()
    with pytest.raises(ValueError):
        scan.train_sample(8, seed=1, method="unknown_method")
