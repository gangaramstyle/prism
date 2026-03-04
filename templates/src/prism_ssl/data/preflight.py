"""Lazy NIfTI resolution/loading utilities used by streaming dataset workers."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
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
    shape_vox: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    extent_mm: tuple[float, float, float]
    inference_reason: str


_AXIS_NAMES = ("x", "y", "z")
_PLANE_BY_THIN_AXIS = {0: "sagittal", 1: "coronal", 2: "axial"}


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
        shape_vox=tuple(int(v) for v in shape.tolist()),
        spacing_mm=tuple(float(v) for v in spacing.tolist()),
        extent_mm=tuple(float(v) for v in extent.tolist()),
        inference_reason=reason,
    )


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


def _affine_linear_translation(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(affine, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected affine shape (4, 4), got {arr.shape}")
    linear = np.asarray(arr[:3, :3], dtype=np.float32)
    translation = np.asarray(arr[:3, 3], dtype=np.float32)
    return linear, translation


def voxel_points_to_world(points_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert voxel coordinates (N, 3) to world coordinates (N, 3) with affine."""
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points_vox shape (N, 3), got {pts.shape}")
    linear, translation = _affine_linear_translation(affine)
    return (pts @ linear.T + translation[np.newaxis, :]).astype(np.float32, copy=False)


def world_points_to_voxel(points_world: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert world coordinates (N, 3) to voxel coordinates (N, 3) with affine."""
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points_world shape (N, 3), got {pts.shape}")
    linear, translation = _affine_linear_translation(affine)
    linear_inv = np.linalg.inv(np.asarray(linear, dtype=np.float64))
    centered = pts.astype(np.float64) - translation.astype(np.float64)[np.newaxis, :]
    return (centered @ linear_inv.T).astype(np.float32, copy=False)


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
    affine_linear: np.ndarray = field(init=False, repr=False)
    affine_translation: np.ndarray = field(init=False, repr=False)
    affine_linear_inv: np.ndarray = field(init=False, repr=False)
    voxel_axis_mm: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        self.affine = np.asarray(self.affine, dtype=np.float32)
        self.spacing = np.asarray(self.spacing, dtype=np.float32)
        if self.data.ndim != 3:
            raise ValueError(f"NiftiScan data must be 3D, got shape={self.data.shape}")
        if self.affine.shape != (4, 4):
            raise ValueError(f"NiftiScan affine must be shape (4,4), got {self.affine.shape}")
        if self.spacing.shape != (3,) or np.any(self.spacing <= 0):
            raise ValueError(f"NiftiScan spacing must be positive shape (3,), got {self.spacing}")
        linear, translation = _affine_linear_translation(self.affine)
        linear = np.asarray(linear, dtype=np.float32)
        axis_norm = np.linalg.norm(linear.astype(np.float64), axis=0).astype(np.float32)
        spacing = np.asarray(self.spacing, dtype=np.float32)
        if np.any(axis_norm <= 1e-6):
            linear = np.diag(spacing.astype(np.float32))
        elif not np.allclose(axis_norm, spacing, rtol=1e-3, atol=1e-3):
            direction = linear / np.clip(axis_norm[np.newaxis, :], 1e-6, None)
            linear = direction * spacing[np.newaxis, :]
        self.affine_linear = np.asarray(linear, dtype=np.float32)
        self.affine_translation = np.asarray(translation, dtype=np.float32)
        self.affine_linear_inv = np.linalg.inv(self.affine_linear.astype(np.float64)).astype(np.float32, copy=False)
        axis_mm = np.linalg.norm(self.affine_linear.astype(np.float64), axis=0).astype(np.float32)
        self.voxel_axis_mm = np.clip(axis_mm, 1e-6, None)

    @property
    def geometry(self) -> ScanGeometry:
        return infer_scan_geometry(self.data.shape, self.voxel_axis_mm)

    @property
    def patch_mm(self) -> np.ndarray:
        mm = np.full(3, self.base_patch_mm, dtype=np.float32)
        thin = self.geometry.thin_axis
        mm[thin] = self.voxel_axis_mm[thin]
        return mm

    @property
    def patch_shape_vox(self) -> np.ndarray:
        return np.maximum(np.ceil(self.patch_mm / self.voxel_axis_mm).astype(np.int64), 1)

    def _sample_center(self, rng: np.random.Generator, sampling_radius_mm: float) -> np.ndarray:
        shape = np.asarray(self.data.shape, dtype=np.int64)
        half_patch = np.ceil(self.patch_shape_vox / 2.0).astype(np.int64)
        thin = self.geometry.thin_axis

        # Only require radius margin on in-plane axes; thin axis just needs patch margin
        radius_vox = np.ceil(sampling_radius_mm / self.voxel_axis_mm).astype(np.int64)
        radius_vox[thin] = 0

        min_idx = half_patch + radius_vox
        max_idx = shape - half_patch - radius_vox - 1
        if np.any(max_idx < min_idx):
            raise SmallScanError(
                f"scan too small for patch extraction: shape={tuple(shape.tolist())} patch={tuple(self.patch_shape_vox.tolist())}"
            )
        return np.array([rng.integers(low=int(lo), high=int(hi) + 1) for lo, hi in zip(min_idx, max_idx)], dtype=np.int64)

    def _extract_patch(self, center_vox: np.ndarray) -> np.ndarray:
        """Extract a 3D patch and squeeze the thin axis to get a 2D slice."""
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

        # Squeeze the thin axis to get a 2D native-plane slice.
        thin = self.geometry.thin_axis
        patch_2d = np.take(patch, indices=patch.shape[thin] // 2, axis=thin)
        return patch_2d.astype(np.float32, copy=False)

    def _resize_patches_batch(
        self,
        patches_2d: np.ndarray,
        *,
        target_patch_size: int | None = None,
    ) -> np.ndarray:
        out_size = int(self.target_patch_size if target_patch_size is None else target_patch_size)
        if out_size <= 0:
            raise ValueError(f"target_patch_size must be > 0, got {out_size}")
        arr = np.asarray(patches_2d, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(1).float()
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return t.squeeze(1).cpu().numpy().astype(np.float32, copy=False)

    def _extract_patches(self, centers_vox: np.ndarray, *, target_patch_size: int | None = None) -> np.ndarray:
        """Extract 2D native-plane patches via direct array slicing."""
        patches_2d = np.stack([self._extract_patch(c) for c in centers_vox], axis=0)
        return self._resize_patches_batch(patches_2d, target_patch_size=target_patch_size)

    def train_sample(
        self,
        n_patches: int,
        *,
        seed: int | None = None,
        wc: float | None = None,
        ww: float | None = None,
        sampling_radius_mm: float | None = None,
        subset_center_vox: np.ndarray | None = None,
        patch_centers_vox: np.ndarray | None = None,
        target_patch_size: int | None = None,
    ) -> dict[str, Any]:
        resolved_target_patch_size = int(self.target_patch_size if target_patch_size is None else target_patch_size)
        rng = np.random.default_rng(seed)

        half_patch = np.ceil(self.patch_shape_vox / 2.0).astype(np.int64)
        shape = np.asarray(self.data.shape, dtype=np.int64)
        thin = self.geometry.thin_axis
        in_plane = [i for i in range(3) if i != thin]

        # Max radius is constrained by in-plane axes only (patches are 1 voxel on thin axis)
        max_radius_mm_by_axis = ((shape - (2 * half_patch) - 1) / 2).astype(np.float64) * self.voxel_axis_mm
        max_radius_mm = float(np.min(max_radius_mm_by_axis[in_plane]))
        if max_radius_mm <= 0:
            raise SmallScanError(
                f"scan too small for patch extraction: shape={tuple(shape.tolist())} patch={tuple(self.patch_shape_vox.tolist())}"
            )
        target_radius = float(rng.uniform(20.0, 30.0)) if sampling_radius_mm is None else float(sampling_radius_mm)
        sampling_radius_mm = min(target_radius, max_radius_mm * 0.9)
        sampling_radius_mm = max(sampling_radius_mm, 0.0)

        if subset_center_vox is None:
            prism_center = self._sample_center(rng, sampling_radius_mm)
        else:
            prism_center = np.asarray(subset_center_vox, dtype=np.int64)
            if prism_center.shape != (3,):
                raise ValueError(f"subset_center_vox must have shape (3,), got {prism_center.shape}")
            if np.any(prism_center < 0) or np.any(prism_center >= shape):
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
                delta_vox = (self.affine_linear_inv @ np.asarray(delta_mm, dtype=np.float32)).astype(np.float32, copy=False)
                center = prism_center + np.rint(delta_vox).astype(np.int64)
                center = np.clip(center, 0, shape - 1)
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
            centers_arr = np.clip(centers_arr, 0, shape - 1)

        raw_patches = self._extract_patches(centers_arr, target_patch_size=resolved_target_patch_size)

        if wc is None or ww is None:
            wc = float(rng.uniform(self.robust_median - self.robust_std, self.robust_median + self.robust_std))
            ww = float(rng.uniform(2.0 * self.robust_std, 6.0 * self.robust_std))
        ww = max(float(ww), 1e-3)

        w_min = float(wc) - 0.5 * ww
        w_max = float(wc) + 0.5 * ww
        clipped = np.clip(raw_patches, w_min, w_max)
        normalized = ((clipped - w_min) / max(w_max - w_min, 1e-6)) * 2.0 - 1.0

        prism_center_pt = voxel_points_to_world(prism_center.astype(np.float32), self.affine)[0]
        patch_centers_pt = voxel_points_to_world(centers_arr.astype(np.float32), self.affine)
        relative_patch_centers_pt = patch_centers_pt - prism_center_pt

        geometry = self.geometry
        return {
            "normalized_patches": normalized.astype(np.float32, copy=False),
            "raw_patches": raw_patches,
            "target_patch_size": int(resolved_target_patch_size),
            "patch_centers_pt": patch_centers_pt,
            "patch_centers_vox": centers_arr.astype(np.int64, copy=False),
            "relative_patch_centers_pt": relative_patch_centers_pt,
            "prism_center_pt": prism_center_pt,
            "prism_center_vox": prism_center.astype(np.int64, copy=False),
            "native_thin_axis": int(geometry.thin_axis),
            "native_thin_axis_name": str(geometry.thin_axis_name),
            "native_acquisition_plane": str(geometry.acquisition_plane),
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
