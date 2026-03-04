from __future__ import annotations

import numpy as np
import pytest

from prism_ssl.data.preflight import (
    NiftiScan,
    infer_scan_geometry,
    voxel_points_to_world,
    world_points_to_voxel,
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


def _make_coronal_scan(seed: int = 456) -> NiftiScan:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(64, 16, 64)).astype(np.float32)
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


def test_train_sample_shape_and_keys() -> None:
    scan = _make_scan()
    result = scan.train_sample(32, seed=77, wc=0.0, ww=4.0)

    assert result["normalized_patches"].shape == (32, 16, 16)
    assert result["patch_centers_vox"].shape == (32, 3)
    assert result["patch_centers_pt"].shape == (32, 3)
    assert result["relative_patch_centers_pt"].shape == (32, 3)
    assert result["prism_center_vox"].shape == (3,)
    assert result["prism_center_pt"].shape == (3,)
    assert "wc" in result
    assert "ww" in result
    assert "native_acquisition_plane" in result


def test_world_voxel_roundtrip_uses_affine_linear_part() -> None:
    affine = np.asarray(
        [
            [0.8, 0.1, 0.0, 10.0],
            [0.0, 0.9, 0.2, -5.0],
            [0.1, 0.0, 1.2, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pts_vox = np.asarray([[10.0, 20.0, 30.0], [4.0, 6.0, 2.0], [31.0, 15.0, 8.0]], dtype=np.float32)
    pts_world = voxel_points_to_world(pts_vox, affine)
    pts_back = world_points_to_voxel(pts_world, affine)
    np.testing.assert_allclose(pts_back, pts_vox, atol=1e-5)


def test_train_sample_manual_overrides() -> None:
    scan = _make_scan()
    center = np.asarray([20, 30, 8], dtype=np.int64)
    patch_centers = np.asarray(
        [[20, 30, 8], [21, 31, 8], [19, 29, 7], [22, 27, 8]],
        dtype=np.int64,
    )
    result = scan.train_sample(
        4,
        seed=99,
        wc=50.0,
        ww=250.0,
        sampling_radius_mm=12.0,
        subset_center_vox=center,
        patch_centers_vox=patch_centers,
    )

    np.testing.assert_array_equal(result["prism_center_vox"], center)
    np.testing.assert_array_equal(result["patch_centers_vox"], patch_centers)
    assert result["wc"] == 50.0
    assert result["ww"] == 250.0
    assert 0.0 < result["sampling_radius_mm"] <= 12.0


def test_infer_scan_geometry_prefers_spacing_for_thin_axis() -> None:
    geom = infer_scan_geometry(shape_vox=(512, 512, 233), spacing_mm=(0.810546875, 0.810546875, 2.0))
    assert geom.thin_axis == 2
    assert geom.acquisition_plane == "axial"
    assert geom.inference_reason == "largest_spacing_mm"


def test_patch_shape_uses_geometry_thin_axis_axial() -> None:
    """Axial scan: thin axis is z (2), patch is 1 voxel thick along z."""
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
    ps = tuple(int(v) for v in scan.patch_shape_vox.tolist())
    # thin_axis=2 (z), so z dim = ceil(2.0/2.0) = 1
    assert ps[2] == 1
    assert ps[0] > 1
    assert ps[1] > 1


def test_patch_shape_uses_geometry_thin_axis_sagittal() -> None:
    """Sagittal scan: thin axis is x (0), patch is 1 voxel thick along x."""
    data = np.zeros((29, 128, 128), dtype=np.float32)
    scan = NiftiScan(
        data=data,
        affine=np.eye(4, dtype=np.float32),
        spacing=np.asarray([2.0, 0.8, 0.8], dtype=np.float32),
        modality="CT",
        base_patch_mm=64.0,
        robust_median=0.0,
        robust_std=1.0,
        robust_low=-1.0,
        robust_high=1.0,
    )
    ps = tuple(int(v) for v in scan.patch_shape_vox.tolist())
    # thin_axis=0 (x), so x dim = ceil(2.0/2.0) = 1
    assert ps[0] == 1
    assert ps[1] > 1
    assert ps[2] > 1


def test_patch_shape_coronal() -> None:
    """Coronal scan: thin axis is y (1)."""
    scan = _make_coronal_scan()
    ps = tuple(int(v) for v in scan.patch_shape_vox.tolist())
    assert scan.geometry.thin_axis == 1
    assert scan.geometry.acquisition_plane == "coronal"
    # With isotropic 1mm spacing, thin axis gets 1 voxel
    assert ps[1] == 1


def test_train_sample_allows_target_patch_size_override() -> None:
    scan = _make_scan(seed=9)
    result = scan.train_sample(5, seed=4, target_patch_size=64)
    assert result["target_patch_size"] == 64
    assert tuple(result["normalized_patches"].shape) == (5, 64, 64)


def test_train_sample_allows_zero_sampling_radius() -> None:
    scan = _make_scan(seed=21)
    result = scan.train_sample(6, seed=8, sampling_radius_mm=0.0)
    assert result["sampling_radius_mm"] == 0.0
    centers = np.asarray(result["patch_centers_vox"], dtype=np.int64)
    center = np.asarray(result["prism_center_vox"], dtype=np.int64)
    np.testing.assert_array_equal(centers, np.repeat(center[None, :], repeats=centers.shape[0], axis=0))


def test_relative_positions_are_relative_to_prism_center() -> None:
    scan = _make_scan(seed=7)
    result = scan.train_sample(8, seed=42)
    prism_pt = result["prism_center_pt"]
    patch_pts = result["patch_centers_pt"]
    rel_pts = result["relative_patch_centers_pt"]
    np.testing.assert_allclose(rel_pts, patch_pts - prism_pt, atol=1e-5)


def test_no_rotation_keys_in_result() -> None:
    scan = _make_scan()
    result = scan.train_sample(4, seed=1)
    for key in result:
        assert "rotation" not in key.lower(), f"Unexpected rotation key: {key}"
