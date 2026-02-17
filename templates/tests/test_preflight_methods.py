from __future__ import annotations

import numpy as np
import pytest

from prism_ssl.data.preflight import (
    NiftiScan,
    infer_scan_geometry,
    rotate_volume_about_center,
    rotated_relative_points_to_voxel,
)


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


def test_train_sample_manual_overrides_are_respected() -> None:
    scan = _make_scan()
    center = np.asarray([20, 30, 10], dtype=np.int64)
    patch_centers = np.asarray(
        [
            [20, 30, 10],
            [21, 31, 10],
            [19, 29, 9],
            [22, 27, 11],
        ],
        dtype=np.int64,
    )
    result = scan.train_sample(
        4,
        seed=99,
        method="optimized_fused",
        wc=50.0,
        ww=250.0,
        sampling_radius_mm=12.0,
        rotation_degrees=(10.0, -5.0, 3.0),
        subset_center_vox=center,
        patch_centers_vox=patch_centers,
    )

    np.testing.assert_array_equal(result["prism_center_vox"], center)
    np.testing.assert_array_equal(result["patch_centers_vox"], patch_centers)
    np.testing.assert_allclose(result["rotation_degrees"], np.asarray([10.0, -5.0, 3.0], dtype=np.float32))
    assert result["wc"] == 50.0
    assert result["ww"] == 250.0
    assert 0.0 < result["sampling_radius_mm"] <= 12.0


def test_infer_scan_geometry_prefers_spacing_for_thin_axis() -> None:
    geom = infer_scan_geometry(shape_vox=(512, 512, 233), spacing_mm=(0.810546875, 0.810546875, 2.0))
    assert geom.thin_axis == 2
    assert geom.acquisition_plane == "axial"
    assert geom.baseline_rotation_degrees == (0.0, 0.0, 0.0)
    assert geom.inference_reason == "largest_spacing_mm"


def test_patch_shape_uses_geometry_thin_axis() -> None:
    data = np.zeros((64, 64, 29), dtype=np.float32)
    scan = NiftiScan(
        data=data,
        affine=np.eye(4, dtype=np.float32),
        spacing=np.asarray([0.810546875, 0.810546875, 2.0], dtype=np.float32),
        modality="CT",
        base_patch_mm=64.0,
        robust_median=0.0,
        robust_std=1.0,
        robust_low=-1.0,
        robust_high=1.0,
    )
    assert tuple(int(v) for v in scan.patch_shape_vox.tolist()) == (79, 79, 2)


def test_train_sample_exposes_rotated_relative_coordinates() -> None:
    scan = _make_scan(seed=7)
    center = np.asarray([20, 30, 10], dtype=np.int64)
    patch_centers = np.asarray(
        [
            [20, 30, 10],
            [25, 31, 9],
            [18, 28, 11],
        ],
        dtype=np.int64,
    )
    result = scan.train_sample(
        3,
        seed=11,
        method="optimized_fused",
        rotation_degrees=(15.0, -10.0, 5.0),
        subset_center_vox=center,
        patch_centers_vox=patch_centers,
    )

    rel = np.asarray(result["relative_patch_centers_pt"], dtype=np.float32)
    rot = np.asarray(result["rotation_matrix_ras"], dtype=np.float32)
    expected = (rot @ rel.T).T
    np.testing.assert_allclose(
        np.asarray(result["relative_patch_centers_pt_rotated"], dtype=np.float32),
        expected,
        rtol=0.0,
        atol=1e-5,
    )


def test_rotate_volume_about_center_identity_is_stable() -> None:
    rng = np.random.default_rng(123)
    volume = rng.normal(size=(12, 10, 8)).astype(np.float32)
    out = rotate_volume_about_center(
        volume,
        center_vox=np.asarray([6.0, 5.0, 4.0], dtype=np.float32),
        rotation_matrix=np.eye(3, dtype=np.float32),
    )
    np.testing.assert_allclose(out, volume, rtol=0.0, atol=1e-5)


def test_rotated_relative_points_to_voxel_rounds_and_clips() -> None:
    rel = np.asarray([[1.0, 2.0, 4.0], [-2.0, -3.0, -4.0]], dtype=np.float32)
    points = rotated_relative_points_to_voxel(
        rel,
        prism_center_vox=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        spacing_mm=np.asarray([1.0, 1.0, 2.0], dtype=np.float32),
        shape_vox=(12, 20, 12),
    )
    expected = np.asarray([[11, 12, 11], [8, 7, 8]], dtype=np.int64)
    np.testing.assert_array_equal(points, expected)
