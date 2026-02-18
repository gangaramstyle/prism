"""Lazy NIfTI resolution/loading utilities used by streaming dataset workers."""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from prism_ssl.config.schema import ScanRecord


class NiftiResolutionError(RuntimeError):
    pass


class NiftiLoadError(RuntimeError):
    pass


class SmallScanError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScanGeometry:
    """Spacing/shape-derived orientation metadata for a canonical RAS volume."""

    thin_axis: int
    thin_axis_name: str
    acquisition_plane: str
    baseline_rotation_degrees: tuple[float, float, float]
    shape_vox: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    extent_mm: tuple[float, float, float]
    inference_reason: str


_AXIS_NAMES = ("x", "y", "z")
_RAS_AXIAL_AXIS = 2
_PLANE_BY_THIN_AXIS = {0: "sagittal", 1: "coronal", 2: "axial"}
_BASELINE_ROTATION_BY_PLANE = {
    "axial": (0.0, 0.0, 0.0),
    "coronal": (90.0, 0.0, 0.0),
    "sagittal": (0.0, -90.0, 0.0),
}


def infer_scan_geometry(
    shape_vox: tuple[int, int, int] | np.ndarray,
    spacing_mm: tuple[float, float, float] | np.ndarray,
) -> ScanGeometry:
    """Infer the native acquisition plane and thin axis from shape/spacing."""
    shape = np.asarray(shape_vox, dtype=np.float32).reshape(-1)
    spacing = np.asarray(spacing_mm, dtype=np.float32).reshape(-1)
    if shape.size != 3 or spacing.size != 3:
        raise ValueError(f"Expected 3D shape/spacing, got shape={tuple(shape.shape)} spacing={tuple(spacing.shape)}")
    if np.any(shape <= 0) or np.any(spacing <= 0):
        raise ValueError(f"Shape and spacing must be positive, got shape={shape} spacing={spacing}")

    extent = shape * spacing
    spacing_sorted = np.sort(spacing)
    spacing_gap = float(spacing_sorted[-1] / max(spacing_sorted[-2], 1e-6))

    if spacing_gap >= 1.2:
        thin_axis = int(np.argmax(spacing))
        reason = "largest_spacing_mm"
    else:
        shape_sorted = np.sort(shape)
        voxel_gap = float(shape_sorted[1] / max(shape_sorted[0], 1e-6))
        if voxel_gap >= 1.2:
            thin_axis = int(np.argmin(shape))
            reason = "fewest_voxels"
        else:
            thin_axis = int(np.argmin(extent))
            reason = "smallest_extent_mm"

    plane = _PLANE_BY_THIN_AXIS[thin_axis]
    return ScanGeometry(
        thin_axis=thin_axis,
        thin_axis_name=_AXIS_NAMES[thin_axis],
        acquisition_plane=plane,
        baseline_rotation_degrees=_BASELINE_ROTATION_BY_PLANE[plane],
        shape_vox=tuple(int(v) for v in shape.tolist()),
        spacing_mm=tuple(float(v) for v in spacing.tolist()),
        extent_mm=tuple(float(v) for v in extent.tolist()),
        inference_reason=reason,
    )


def _canonical_method_name(method: str) -> str:
    name = str(method)
    aliases = {
        "naive_volume": "legacy_loop",
        "optimized_local": "legacy_loop",
        "optimized_patch": "optimized_fused",
        "fused_vectorized": "optimized_fused",
    }
    canonical = aliases.get(name, name)
    valid = {"legacy_loop", "optimized_fused"}
    if canonical not in valid:
        raise ValueError(f"Unknown sampling method '{method}'. Expected one of {sorted(valid)}")
    return canonical


def resolve_nifti_path(series_path: str) -> str:
    """Resolve NIfTI path deterministically by largest file size."""
    path = Path(series_path)
    if path.is_file() and path.suffix in {".nii", ".gz"}:
        return str(path)

    candidates: list[str] = []
    candidates.extend(glob.glob(os.path.join(series_path, "*.nii.gz")))
    candidates.extend(glob.glob(os.path.join(series_path, "*.nii")))

    if not candidates:
        parent = os.path.dirname(series_path)
        candidates.extend(glob.glob(os.path.join(parent, "*.nii.gz")))
        candidates.extend(glob.glob(os.path.join(parent, "*.nii")))

    if not candidates:
        raise NiftiResolutionError(f"Could not resolve NIfTI for series_path={series_path}")

    candidates.sort(key=lambda f: os.path.getsize(f), reverse=True)
    return candidates[0]


def compute_robust_stats(data: np.ndarray) -> tuple[float, float, float, float]:
    """Return median/std and robust lower/upper bounds."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 1.0, 0.0, 1.0
    p_low, p_high = np.percentile(arr, [0.5, 99.5])
    clipped = np.clip(arr, p_low, p_high)
    median = float(np.median(clipped))
    std = float(np.std(clipped))
    if std <= 1e-6:
        std = 1.0
    if p_high <= p_low:
        p_high = p_low + 1.0
    return median, std, float(p_low), float(p_high)


def _euler_xyz_to_matrix(degrees_xyz: tuple[float, float, float]) -> np.ndarray:
    x, y, z = [math.radians(v) for v in degrees_xyz]
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def _matrix_to_euler_xyz_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return one XYZ-Euler representation for a rotation matrix.

    This is used for debugging/metadata only and is not used for training targets.
    """
    rot = np.asarray(matrix, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (3, 3), got {rot.shape}")

    sy = float(np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rot[2, 1], rot[2, 2])
        y = math.atan2(-rot[2, 0], sy)
        z = math.atan2(rot[1, 0], rot[0, 0])
    else:
        x = math.atan2(-rot[1, 2], rot[1, 1])
        y = math.atan2(-rot[2, 0], sy)
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def rotate_volume_about_center(
    volume: np.ndarray,
    center_vox: np.ndarray,
    rotation_matrix: np.ndarray,
    *,
    spacing_mm: np.ndarray | tuple[float, float, float] | None = None,
    interpolation_order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """Rotate a 3D volume around an in-volume voxel center.

    If ``spacing_mm`` is provided, rotation is applied in physical (mm) space
    and mapped back to voxel space so anisotropic scans are not warped.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")
    center = np.asarray(center_vox, dtype=np.float64).reshape(-1)
    rot = np.asarray(rotation_matrix, dtype=np.float64)
    if center.size != 3 or rot.shape != (3, 3):
        raise ValueError(
            f"Invalid center/rotation shape: center={tuple(center.shape)} rotation={tuple(rot.shape)}"
        )
    rot_inv = np.linalg.inv(rot)
    if spacing_mm is None:
        matrix = rot_inv
    else:
        spacing = np.asarray(spacing_mm, dtype=np.float64).reshape(-1)
        if spacing.size != 3 or np.any(spacing <= 0.0):
            raise ValueError(f"Invalid spacing_mm: {spacing_mm}")
        scale = np.diag(spacing)
        inv_scale = np.diag(1.0 / spacing)
        # x_in_vox = S^-1 R^-1 S x_out_vox + offset
        matrix = inv_scale @ rot_inv @ scale
    offset = center - matrix @ center
    return ndimage.affine_transform(
        vol,
        matrix,
        offset=offset,
        order=int(interpolation_order),
        mode=mode,
    ).astype(np.float32, copy=False)


def rotated_relative_points_to_voxel(
    relative_points_pt: np.ndarray,
    prism_center_vox: np.ndarray,
    spacing_mm: np.ndarray,
    *,
    shape_vox: tuple[int, int, int] | np.ndarray | None = None,
) -> np.ndarray:
    """Convert rotated relative point coordinates (mm) back to voxel indices."""
    rel = np.asarray(relative_points_pt, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] != 3:
        raise ValueError(f"Expected relative_points_pt shape (N, 3), got {rel.shape}")
    center_vox = np.asarray(prism_center_vox, dtype=np.float32).reshape(-1)
    spacing = np.asarray(spacing_mm, dtype=np.float32).reshape(-1)
    if center_vox.size != 3 or spacing.size != 3:
        raise ValueError(
            f"Invalid center/spacing shape: center={tuple(center_vox.shape)} spacing={tuple(spacing.shape)}"
        )
    center_pt = center_vox * spacing
    points_pt = center_pt[np.newaxis, :] + rel
    points_vox = np.rint(points_pt / spacing).astype(np.int64)
    if shape_vox is not None:
        upper = np.asarray(shape_vox, dtype=np.int64).reshape(-1)
        if upper.size != 3:
            raise ValueError(f"Invalid shape_vox shape: {tuple(upper.shape)}")
        points_vox = np.clip(points_vox, 0, upper - 1)
    return points_vox


@dataclass
class NiftiScan:
    data: np.ndarray
    affine: np.ndarray
    spacing: np.ndarray
    modality: str
    base_patch_mm: float
    robust_median: float
    robust_std: float
    robust_low: float
    robust_high: float
    target_patch_size: int = 16

    @property
    def geometry(self) -> ScanGeometry:
        return infer_scan_geometry(self.data.shape, self.spacing)

    @property
    def patch_mm(self) -> np.ndarray:
        mm = np.array([self.base_patch_mm, self.base_patch_mm, self.base_patch_mm], dtype=np.float32)
        # Sampling frame is always canonical RAS: axial baseline means z is thin.
        mm[_RAS_AXIAL_AXIS] = self.base_patch_mm / 16.0
        return mm

    @property
    def patch_shape_vox(self) -> np.ndarray:
        return np.maximum(np.ceil(self.patch_mm / self.spacing).astype(np.int64), 1)

    def _sample_center(self, rng: np.random.Generator, sampling_radius_mm: float) -> np.ndarray:
        shape = np.asarray(self.data.shape, dtype=np.int64)
        radius_vox = np.ceil(sampling_radius_mm / self.spacing).astype(np.int64)
        half_patch = np.ceil(self.patch_shape_vox / 2.0).astype(np.int64)
        min_idx = half_patch + radius_vox
        max_idx = shape - half_patch - radius_vox - 1
        if np.any(max_idx < min_idx):
            raise SmallScanError(
                f"scan too small for patch extraction: shape={tuple(shape.tolist())} patch={tuple(self.patch_shape_vox.tolist())}"
            )
        return np.array([rng.integers(low=int(lo), high=int(hi) + 1) for lo, hi in zip(min_idx, max_idx)], dtype=np.int64)

    def _extract_patch(self, center_vox: np.ndarray) -> np.ndarray:
        shape = self.patch_shape_vox
        starts = center_vox - (shape // 2)
        ends = starts + shape

        src_starts = np.maximum(starts, 0)
        src_ends = np.minimum(ends, np.asarray(self.data.shape, dtype=np.int64))

        patch = np.zeros(tuple(int(v) for v in shape.tolist()), dtype=np.float32)
        dst_starts = src_starts - starts
        dst_ends = dst_starts + (src_ends - src_starts)

        patch[
            slice(int(dst_starts[0]), int(dst_ends[0])),
            slice(int(dst_starts[1]), int(dst_ends[1])),
            slice(int(dst_starts[2]), int(dst_ends[2])),
        ] = self.data[
            slice(int(src_starts[0]), int(src_ends[0])),
            slice(int(src_starts[1]), int(src_ends[1])),
            slice(int(src_starts[2]), int(src_ends[2])),
        ]

        # Standard contract expects (N, 16, 16) style data with singleton depth squeezed.
        if patch.shape[0] == 1:
            patch = patch[0]
        elif patch.shape[1] == 1:
            patch = patch[:, 0, :]
        elif patch.shape[2] == 1:
            patch = patch[:, :, 0]
        return patch.astype(np.float32, copy=False)

    @staticmethod
    def _patch_to_2d(patch: np.ndarray) -> np.ndarray:
        arr = np.asarray(patch, dtype=np.float32)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            center_idx = int(arr.shape[_RAS_AXIAL_AXIS] // 2)
            return np.take(arr, indices=center_idx, axis=_RAS_AXIAL_AXIS).astype(np.float32, copy=False)
        raise SmallScanError(f"Unexpected patch rank: {arr.ndim}")

    def _resize_patch(self, patch_2d: np.ndarray) -> np.ndarray:
        return self._resize_patches_batch(patch_2d)[0]

    def _resolve_target_patch_size(self, target_patch_size: int | None = None) -> int:
        size = int(self.target_patch_size if target_patch_size is None else target_patch_size)
        if size <= 0:
            raise ValueError(f"target_patch_size must be > 0, got {size}")
        return size

    def _resize_patches_batch(
        self,
        patches_2d: np.ndarray,
        *,
        target_patch_size: int | None = None,
    ) -> np.ndarray:
        out_size = self._resolve_target_patch_size(target_patch_size)
        arr = np.asarray(patches_2d, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(1).float()
        t = F.interpolate(
            t,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )
        return t.squeeze(1).cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _patches_to_2d(patches_3d: np.ndarray) -> np.ndarray:
        arr = np.asarray(patches_3d, dtype=np.float32)
        if arr.ndim == 3:
            return NiftiScan._patch_to_2d(arr)
        if arr.ndim != 4:
            raise SmallScanError(f"Unexpected patch batch rank: {arr.ndim}")

        center_idx = int(arr.shape[_RAS_AXIAL_AXIS + 1] // 2)
        return arr[:, :, :, center_idx]

    def _sampling_offset_vectors_vox(
        self,
        patch_shape: np.ndarray,
        *,
        rotation_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        offsets = [np.arange(int(d), dtype=np.float32) - (float(d) - 1.0) / 2.0 for d in patch_shape]
        mesh = np.meshgrid(*offsets, indexing="ij")
        offset_vectors = np.stack([m.reshape(-1) for m in mesh], axis=1).astype(np.float32, copy=False)
        if rotation_matrix is None:
            return offset_vectors

        spacing = np.asarray(self.spacing, dtype=np.float32)
        inv_rot = np.asarray(rotation_matrix, dtype=np.float32).T
        offset_mm = offset_vectors * spacing[np.newaxis, :]
        src_offset_mm = (inv_rot @ offset_mm.T).T
        return (src_offset_mm / np.maximum(spacing[np.newaxis, :], 1e-6)).astype(np.float32, copy=False)

    def _extract_patches_legacy_loop(
        self,
        centers_vox: np.ndarray,
        *,
        target_patch_size: int | None = None,
        rotation_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        if rotation_matrix is None:
            return np.stack(
                [
                    self._resize_patches_batch(
                        self._patch_to_2d(self._extract_patch(c)),
                        target_patch_size=target_patch_size,
                    )[0]
                    for c in centers_vox
                ],
                axis=0,
            )

        patch_shape = self.patch_shape_vox.astype(np.int64)
        offset_vectors = self._sampling_offset_vectors_vox(patch_shape, rotation_matrix=rotation_matrix)
        centers = np.asarray(centers_vox, dtype=np.float32)
        patches: list[np.ndarray] = []
        for center in centers:
            src_vox = center[np.newaxis, :] + offset_vectors
            sampled = ndimage.map_coordinates(
                self.data,
                [src_vox[:, 0], src_vox[:, 1], src_vox[:, 2]],
                order=1,
                mode="constant",
                cval=0.0,
            )
            patch_3d = sampled.reshape(int(patch_shape[0]), int(patch_shape[1]), int(patch_shape[2]))
            patch_2d = self._patch_to_2d(patch_3d)
            patches.append(self._resize_patches_batch(patch_2d, target_patch_size=target_patch_size)[0])
        return np.stack(
            patches,
            axis=0,
        )

    def _extract_patches_optimized_fused(
        self,
        centers_vox: np.ndarray,
        *,
        target_patch_size: int | None = None,
        rotation_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        patch_shape = self.patch_shape_vox.astype(np.int64)
        offset_vectors = self._sampling_offset_vectors_vox(patch_shape, rotation_matrix=rotation_matrix)

        centers = np.asarray(centers_vox, dtype=np.float32)
        all_src_vox = centers[:, np.newaxis, :] + offset_vectors[np.newaxis, :, :]
        flat_src_vox = all_src_vox.reshape(-1, 3)

        sampled = ndimage.map_coordinates(
            self.data,
            [flat_src_vox[:, 0], flat_src_vox[:, 1], flat_src_vox[:, 2]],
            order=1,
            mode="constant",
            cval=0.0,
        )
        patches_3d = sampled.reshape(centers.shape[0], int(patch_shape[0]), int(patch_shape[1]), int(patch_shape[2]))
        patches_2d = self._patches_to_2d(patches_3d)
        return self._resize_patches_batch(patches_2d, target_patch_size=target_patch_size)

    def train_sample(
        self,
        n_patches: int,
        *,
        seed: int | None = None,
        method: str = "optimized_fused",
        wc: float | None = None,
        ww: float | None = None,
        sampling_radius_mm: float | None = None,
        rotation_degrees: tuple[float, float, float] | None = None,
        native_hint_rotation_degrees: tuple[float, float, float] | None = None,
        rotation_augmentation_degrees: tuple[float, float, float] | None = None,
        apply_native_orientation_hint: bool = True,
        rotation_augmentation_max_degrees: float = 10.0,
        subset_center_vox: np.ndarray | None = None,
        patch_centers_vox: np.ndarray | None = None,
        target_patch_size: int | None = None,
    ) -> dict[str, Any]:
        method_name = _canonical_method_name(method)
        resolved_target_patch_size = self._resolve_target_patch_size(target_patch_size)
        rng = np.random.default_rng(seed)

        half_patch = np.ceil(self.patch_shape_vox / 2.0).astype(np.int64)
        shape = np.asarray(self.data.shape, dtype=np.int64)
        max_radius_vox_by_axis = ((shape - (2 * half_patch) - 1) // 2).astype(np.int64)
        max_radius_vox = int(np.min(max_radius_vox_by_axis))
        if max_radius_vox <= 0:
            raise SmallScanError(
                f"scan too small for patch extraction: shape={tuple(shape.tolist())} patch={tuple(self.patch_shape_vox.tolist())}"
            )
        max_radius_mm = float(max_radius_vox * float(np.min(self.spacing)))
        target_radius = float(rng.uniform(20.0, 30.0)) if sampling_radius_mm is None else float(sampling_radius_mm)
        sampling_radius_mm = min(target_radius, max_radius_mm * 0.9)
        sampling_radius_mm = max(sampling_radius_mm, 0.0)
        if subset_center_vox is None:
            prism_center = self._sample_center(rng, sampling_radius_mm)
        else:
            prism_center = np.asarray(subset_center_vox, dtype=np.int64)
            if prism_center.shape != (3,):
                raise ValueError(f"subset_center_vox must have shape (3,), got {prism_center.shape}")
            if np.any(prism_center < 0) or np.any(prism_center >= np.asarray(self.data.shape, dtype=np.int64)):
                raise ValueError(
                    f"subset_center_vox out of bounds: center={tuple(int(x) for x in prism_center.tolist())} "
                    f"shape={tuple(int(x) for x in self.data.shape)}"
                )

        if patch_centers_vox is None:
            centers = []
            for _ in range(int(n_patches)):
                delta_mm = rng.uniform(-sampling_radius_mm, sampling_radius_mm, size=3)
                if float(np.linalg.norm(delta_mm)) > sampling_radius_mm:
                    delta_mm = delta_mm * (sampling_radius_mm / max(np.linalg.norm(delta_mm), 1e-6))
                center = prism_center + np.rint(delta_mm / self.spacing).astype(np.int64)
                center = np.clip(center, 0, np.asarray(self.data.shape, dtype=np.int64) - 1)
                centers.append(center)
            centers_arr = np.asarray(centers, dtype=np.int64)
        else:
            centers_arr = np.asarray(patch_centers_vox, dtype=np.int64)
            if centers_arr.ndim != 2 or centers_arr.shape[1] != 3:
                raise ValueError(f"patch_centers_vox must have shape (N, 3), got {centers_arr.shape}")
            if int(centers_arr.shape[0]) != int(n_patches):
                raise ValueError(
                    f"patch_centers_vox has {centers_arr.shape[0]} centers but n_patches={int(n_patches)}"
                )
            centers_arr = np.clip(centers_arr, 0, np.asarray(self.data.shape, dtype=np.int64) - 1)

        geometry = self.geometry
        if native_hint_rotation_degrees is None:
            hint_tuple = tuple(float(v) for v in geometry.baseline_rotation_degrees)
        else:
            if len(native_hint_rotation_degrees) != 3:
                raise ValueError("native_hint_rotation_degrees must be a tuple of length 3")
            hint_tuple = tuple(float(v) for v in native_hint_rotation_degrees)

        if rotation_degrees is not None and rotation_augmentation_degrees is not None:
            raise ValueError("Provide either rotation_degrees or rotation_augmentation_degrees, not both")

        if rotation_degrees is not None:
            if len(rotation_degrees) != 3:
                raise ValueError("rotation_degrees must be a tuple of length 3")
            # Absolute override in canonical RAS space.
            rotation_control_tuple = tuple(float(v) for v in rotation_degrees)
            rotation_augmentation_tuple = rotation_control_tuple
            rotation_matrix = _euler_xyz_to_matrix(rotation_control_tuple)
        else:
            if rotation_augmentation_degrees is None:
                if seed is None:
                    rot_rng = np.random.default_rng()
                else:
                    rot_rng = np.random.default_rng(int(seed) + 1_000_003)
                max_abs_aug = max(float(rotation_augmentation_max_degrees), 0.0)
                rotation_augmentation_tuple = tuple(
                    float(rot_rng.uniform(-max_abs_aug, max_abs_aug)) for _ in range(3)
                )
            else:
                if len(rotation_augmentation_degrees) != 3:
                    raise ValueError("rotation_augmentation_degrees must be a tuple of length 3")
                rotation_augmentation_tuple = tuple(float(v) for v in rotation_augmentation_degrees)

            rotation_control_tuple = rotation_augmentation_tuple
            aug_matrix = _euler_xyz_to_matrix(rotation_augmentation_tuple)
            if bool(apply_native_orientation_hint):
                hint_matrix = _euler_xyz_to_matrix(hint_tuple)
                # Global-RAS augmentation axes, then reorient with native hint.
                rotation_matrix = hint_matrix @ aug_matrix
            else:
                rotation_matrix = aug_matrix

        rotation_effective_tuple = _matrix_to_euler_xyz_degrees(rotation_matrix)

        if method_name == "optimized_fused":
            raw_patches = self._extract_patches_optimized_fused(
                centers_arr,
                target_patch_size=resolved_target_patch_size,
                rotation_matrix=rotation_matrix,
            )
        else:
            raw_patches = self._extract_patches_legacy_loop(
                centers_arr,
                target_patch_size=resolved_target_patch_size,
                rotation_matrix=rotation_matrix,
            )

        if wc is None or ww is None:
            wc = float(rng.uniform(self.robust_median - self.robust_std, self.robust_median + self.robust_std))
            ww = float(rng.uniform(2.0 * self.robust_std, 6.0 * self.robust_std))
        ww = max(float(ww), 1e-3)

        w_min = float(wc) - 0.5 * ww
        w_max = float(wc) + 0.5 * ww
        clipped = np.clip(raw_patches, w_min, w_max)
        normalized = ((clipped - w_min) / max(w_max - w_min, 1e-6)) * 2.0 - 1.0

        prism_center_pt = (prism_center.astype(np.float32) * self.spacing.astype(np.float32)).astype(np.float32)
        patch_centers_pt = (centers_arr.astype(np.float32) * self.spacing.astype(np.float32)).astype(np.float32)
        relative_patch_centers_pt = patch_centers_pt - prism_center_pt
        relative_patch_centers_pt_rotated = (rotation_matrix @ relative_patch_centers_pt.T).T.astype(
            np.float32,
            copy=False,
        )

        return {
            "method": method_name,
            "raw_patches": raw_patches,
            "normalized_patches": normalized.astype(np.float32, copy=False),
            "target_patch_size": int(resolved_target_patch_size),
            "patch_centers_pt": patch_centers_pt,
            "patch_centers_vox": centers_arr.astype(np.int64, copy=False),
            "relative_patch_centers_pt": relative_patch_centers_pt,
            "relative_patch_centers_pt_rotated": relative_patch_centers_pt_rotated,
            "prism_center_pt": prism_center_pt,
            "prism_center_vox": prism_center.astype(np.int64, copy=False),
            "rotation_hint_degrees": np.asarray(hint_tuple, dtype=np.float32),
            "rotation_augmentation_degrees": np.asarray(rotation_augmentation_tuple, dtype=np.float32),
            "rotation_degrees": np.asarray(rotation_control_tuple, dtype=np.float32),
            "rotation_effective_degrees": np.asarray(rotation_effective_tuple, dtype=np.float32),
            "rotation_matrix_ras": rotation_matrix,
            "native_thin_axis": int(geometry.thin_axis),
            "native_thin_axis_name": str(geometry.thin_axis_name),
            "native_acquisition_plane": str(geometry.acquisition_plane),
            "native_baseline_rotation_degrees": np.asarray(geometry.baseline_rotation_degrees, dtype=np.float32),
            "native_inference_reason": str(geometry.inference_reason),
            "patch_content_rotated": True,
            "wc": float(wc),
            "ww": float(ww),
            "w_min": float(w_min),
            "w_max": float(w_max),
            "sampling_radius_mm": float(sampling_radius_mm),
        }


def load_nifti_scan(record: ScanRecord | dict[str, Any], base_patch_mm: float) -> tuple[NiftiScan, str]:
    """Load a scan object from a record; resolve NIfTI path lazily if needed."""
    if isinstance(record, ScanRecord):
        rec = record
        modality = rec.modality
        series_path = rec.series_path
        nifti_path = rec.nifti_path
    else:
        modality = str(record.get("modality", "CT"))
        series_path = str(record.get("series_path", ""))
        nifti_path = str(record.get("nifti_path", ""))

    effective_path = nifti_path if nifti_path else resolve_nifti_path(series_path)
    try:
        raw = nib.load(effective_path)
        try:
            img = nib.as_closest_canonical(raw)
        except Exception:
            img = raw
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim != 3:
            raise NiftiLoadError(f"Expected 3D volume, got shape={data.shape}")
        spacing = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
        if np.any(spacing <= 0):
            raise NiftiLoadError(f"Invalid spacing for {effective_path}: {spacing}")
        median, std, p_low, p_high = compute_robust_stats(data)
        scan = NiftiScan(
            data=data,
            affine=np.asarray(img.affine, dtype=np.float32),
            spacing=spacing,
            modality=modality.upper(),
            base_patch_mm=float(base_patch_mm),
            robust_median=median,
            robust_std=std,
            robust_low=p_low,
            robust_high=p_high,
        )
        return scan, effective_path
    except (NiftiResolutionError, NiftiLoadError):
        raise
    except Exception as exc:
        raise NiftiLoadError(f"Failed loading NIfTI '{effective_path}': {exc}") from exc
