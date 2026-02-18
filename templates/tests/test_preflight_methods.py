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


def _make_coronal_scan(seed: int = 456) -> NiftiScan:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(64, 48, 64)).astype(np.float32)
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
    np.testing.assert_allclose(
        result["rotation_augmentation_degrees"],
        np.asarray([10.0, -5.0, 3.0], dtype=np.float32),
    )
    assert result["wc"] == 50.0
    assert result["ww"] == 250.0
    assert 0.0 < result["sampling_radius_mm"] <= 12.0


def test_train_sample_applies_native_hint_plus_rotation_augmentation() -> None:
    scan = _make_coronal_scan()
    result = scan.train_sample(
        3,
        seed=17,
        method="optimized_fused",
        rotation_augmentation_degrees=(5.0, -3.0, 2.0),
        apply_native_orientation_hint=True,
    )
    np.testing.assert_allclose(result["rotation_hint_degrees"], np.asarray([90.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(
        result["rotation_augmentation_degrees"],
        np.asarray([5.0, -3.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(result["rotation_degrees"], np.asarray([5.0, -3.0, 2.0], dtype=np.float32))
    hint = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )  # +90 around X
    aug = np.asarray(result["rotation_augmentation_degrees"], dtype=np.float32)
    ax, ay, az = [float(v) for v in aug.tolist()]
    # Compose as global-RAS augmentation, then apply hint: R_eff = R_hint @ R_aug
    from prism_ssl.data.preflight import _euler_xyz_to_matrix

    expected = hint @ _euler_xyz_to_matrix((ax, ay, az))
    np.testing.assert_allclose(np.asarray(result["rotation_matrix_ras"], dtype=np.float32), expected, atol=1e-5)


def test_train_sample_random_rotation_augmentation_is_bounded() -> None:
    scan = _make_scan(seed=12)
    result = scan.train_sample(
        4,
        seed=55,
        method="optimized_fused",
        rotation_augmentation_max_degrees=7.5,
    )
    aug = np.asarray(result["rotation_augmentation_degrees"], dtype=np.float32)
    assert np.all(np.abs(aug) <= 7.5 + 1e-6)


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


def test_patch_shape_uses_ras_axial_axis_even_for_sagittal_native() -> None:
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
    # RAS-axial baseline: z stays the thin patch axis regardless of native plane inference.
    assert tuple(int(v) for v in scan.patch_shape_vox.tolist()) == (32, 80, 5)


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


def test_train_sample_allows_target_patch_size_override() -> None:
    scan = _make_scan(seed=9)
    result = scan.train_sample(5, seed=4, method="optimized_fused", target_patch_size=64)
    assert result["target_patch_size"] == 64
    assert tuple(result["normalized_patches"].shape) == (5, 64, 64)


def test_train_sample_allows_zero_sampling_radius() -> None:
    scan = _make_scan(seed=21)
    result = scan.train_sample(
        6,
        seed=8,
        method="optimized_fused",
        sampling_radius_mm=0.0,
    )
    assert result["sampling_radius_mm"] == 0.0
    centers = np.asarray(result["patch_centers_vox"], dtype=np.int64)
    center = np.asarray(result["prism_center_vox"], dtype=np.int64)
    np.testing.assert_array_equal(centers, np.repeat(center[None, :], repeats=centers.shape[0], axis=0))


def test_rotation_changes_patch_pixels_for_same_center() -> None:
    scan = _make_scan(seed=44)
    center = np.asarray([20, 20, 8], dtype=np.int64)
    centers = np.asarray(
        [
            [20, 20, 8],
            [20, 20, 8],
            [20, 20, 8],
        ],
        dtype=np.int64,
    )
    res_a = scan.train_sample(
        3,
        seed=123,
        method="optimized_fused",
        wc=0.0,
        ww=4.0,
        subset_center_vox=center,
        patch_centers_vox=centers,
        rotation_degrees=(0.0, 0.0, 0.0),
        target_patch_size=32,
    )
    res_b = scan.train_sample(
        3,
        seed=123,
        method="optimized_fused",
        wc=0.0,
        ww=4.0,
        subset_center_vox=center,
        patch_centers_vox=centers,
        rotation_degrees=(25.0, -10.0, 15.0),
        target_patch_size=32,
    )
    mean_abs_delta = float(np.mean(np.abs(res_a["normalized_patches"] - res_b["normalized_patches"])))
    assert mean_abs_delta > 1e-3


def test_rotate_volume_about_center_identity_is_stable() -> None:
    rng = np.random.default_rng(123)
    volume = rng.normal(size=(12, 10, 8)).astype(np.float32)
    out = rotate_volume_about_center(
        volume,
        center_vox=np.asarray([6.0, 5.0, 4.0], dtype=np.float32),
        rotation_matrix=np.eye(3, dtype=np.float32),
    )
    np.testing.assert_allclose(out, volume, rtol=0.0, atol=1e-5)


def test_rotate_volume_about_center_spacing_aware_avoids_anisotropic_warp() -> None:
    volume = np.zeros((33, 33, 17), dtype=np.float32)
    center = np.asarray([16.0, 16.0, 8.0], dtype=np.float32)
    src = np.asarray([16, 16, 10], dtype=np.int64)  # +2 vox along z => +4 mm when spacing z=2
    volume[src[0], src[1], src[2]] = 1.0

    rot_x_90 = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    out_spacing = rotate_volume_about_center(
        volume,
        center_vox=center,
        rotation_matrix=rot_x_90,
        spacing_mm=np.asarray([1.0, 1.0, 2.0], dtype=np.float32),
        interpolation_order=0,
        mode="constant",
    )
    out_voxel = rotate_volume_about_center(
        volume,
        center_vox=center,
        rotation_matrix=rot_x_90,
        interpolation_order=0,
        mode="constant",
    )

    idx_spacing = np.asarray(np.unravel_index(int(np.argmax(out_spacing)), out_spacing.shape), dtype=np.int64)
    idx_voxel = np.asarray(np.unravel_index(int(np.argmax(out_voxel)), out_voxel.shape), dtype=np.int64)
    disp_spacing = idx_spacing - center.astype(np.int64)
    disp_voxel = idx_voxel - center.astype(np.int64)

    assert int(disp_spacing[0]) == 0
    assert int(disp_spacing[2]) == 0
    assert abs(int(disp_spacing[1])) == 4  # 4 mm mapped onto y-axis with 1 mm spacing
    assert abs(int(disp_voxel[1])) == 2  # old voxel-space behavior shrinks motion by z spacing factor


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
