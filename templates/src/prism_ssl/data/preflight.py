"""Lazy NIfTI resolution/loading utilities used by streaming dataset workers."""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image

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
_TOTALSEG_TOTAL_CT_SUFFIX = "_e1_ts_total_ct.nii.gz"
_TOTALSEG_BODY_MAX_CANDIDATES = 4096
_TOTALSEG_BODY_GRID_STEPS = (4, 2, 1)


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


def resolve_totalseg_total_ct_path(series_path: str) -> str:
    """Resolve a matching TotalSegmentator total-CT path when the PMBB layout is present."""
    normalized = os.path.normpath(str(series_path))
    marker = f"{os.sep}subjects{os.sep}"
    if marker not in normalized:
        return ""
    prefix, suffix = normalized.split(marker, 1)
    series_name = Path(normalized).name
    candidate = Path(prefix) / "processing" / "totalsegmentator" / suffix / f"{series_name}{_TOTALSEG_TOTAL_CT_SUFFIX}"
    return str(candidate) if candidate.is_file() else ""


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
    body_center_candidates_vox: np.ndarray | None = field(default=None, repr=False)
    body_center_candidates_pt: np.ndarray | None = field(default=None, repr=False)
    body_sampling_source: str = ""
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
        if self.body_center_candidates_vox is None:
            self.body_center_candidates_vox = np.empty((0, 3), dtype=np.int32)
        else:
            body_centers = np.asarray(self.body_center_candidates_vox, dtype=np.int32)
            self.body_center_candidates_vox = body_centers.reshape(-1, 3) if body_centers.size else np.empty((0, 3), dtype=np.int32)
        if self.body_center_candidates_pt is None:
            if len(self.body_center_candidates_vox) > 0:
                self.body_center_candidates_pt = voxel_points_to_world(self.body_center_candidates_vox, self.affine)
            else:
                self.body_center_candidates_pt = np.empty((0, 3), dtype=np.float32)
        else:
            body_points = np.asarray(self.body_center_candidates_pt, dtype=np.float32)
            self.body_center_candidates_pt = body_points.reshape(-1, 3) if body_points.size else np.empty((0, 3), dtype=np.float32)

    @property
    def geometry(self) -> ScanGeometry:
        return infer_scan_geometry(self.data.shape, self.voxel_axis_mm)

    @property
    def patch_shape_vox(self) -> np.ndarray:
        """Physical voxel footprint for base_patch_mm (used for overlay box drawing)."""
        return self._mm_patch_vox_shape(self.base_patch_mm)

    def _mm_patch_vox_shape(self, patch_mm: float) -> np.ndarray:
        """Compute 3D voxel shape covering patch_mm in-plane, 1 voxel on thin axis."""
        thin = self.geometry.thin_axis
        n_vox = np.maximum(np.ceil(patch_mm / self.voxel_axis_mm).astype(np.int64), 1)
        n_vox[thin] = 1
        return n_vox

    def _sample_center(self, rng: np.random.Generator, patch_vox: np.ndarray) -> np.ndarray:
        """Pick a random voxel inside the volume with just enough margin for one patch."""
        min_idx, max_idx = self._center_bounds_for_full_patch(patch_vox)
        return np.array([rng.integers(low=int(lo), high=int(hi) + 1) for lo, hi in zip(min_idx, max_idx)], dtype=np.int64)

    def sample_prism_center_vox(self, rng: np.random.Generator, patch_vox: np.ndarray | None = None) -> np.ndarray:
        """Sample a valid prism center, preferring TotalSegmentator body voxels when available."""
        if len(self.body_center_candidates_vox) > 0:
            idx = int(rng.integers(low=0, high=len(self.body_center_candidates_vox)))
            return np.asarray(self.body_center_candidates_vox[idx], dtype=np.int64).copy()
        resolved_patch_vox = self.patch_shape_vox if patch_vox is None else np.asarray(patch_vox, dtype=np.int64)
        return self._sample_center(rng, resolved_patch_vox)

    def sample_center_near_vox(
        self,
        rng: np.random.Generator,
        anchor_center_vox: np.ndarray,
        radius_mm: float,
        patch_vox: np.ndarray | None = None,
        max_attempts: int = 32,
    ) -> np.ndarray:
        """Sample a center near an anchor, preferring nearby body voxels when available."""
        resolved_patch_vox = self.patch_shape_vox if patch_vox is None else np.asarray(patch_vox, dtype=np.int64)
        min_idx, max_idx = self._center_bounds_for_full_patch(resolved_patch_vox)
        anchor = np.asarray(anchor_center_vox, dtype=np.int64).reshape(3)
        anchor = np.clip(anchor, min_idx, max_idx)
        radius_mm = max(float(radius_mm), 0.0)

        if len(self.body_center_candidates_vox) > 0 and len(self.body_center_candidates_pt) > 0 and radius_mm > 0.0:
            anchor_pt = voxel_points_to_world(anchor.astype(np.float32), self.affine)[0]
            dist_mm = np.linalg.norm(self.body_center_candidates_pt - anchor_pt[np.newaxis, :], axis=1)
            nearby = np.flatnonzero((dist_mm > 1e-3) & (dist_mm <= radius_mm))
            if len(nearby) > 0:
                idx = int(nearby[int(rng.integers(low=0, high=len(nearby)))])
                return np.asarray(self.body_center_candidates_vox[idx], dtype=np.int64).copy()

        for _ in range(max(1, int(max_attempts))):
            direction = rng.normal(loc=0.0, scale=1.0, size=3)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                continue
            direction = direction / norm
            radius = radius_mm * float(rng.random() ** (1.0 / 3.0))
            delta_mm = direction * radius
            delta_vox = (self.affine_linear_inv @ np.asarray(delta_mm, dtype=np.float32)).astype(np.float32, copy=False)
            center = anchor + np.rint(delta_vox).astype(np.int64)
            center = np.clip(center, min_idx, max_idx)
            if np.any(center != anchor):
                return center
        return self.sample_prism_center_vox(rng, patch_vox=resolved_patch_vox)

    def _center_bounds_for_full_patch(self, patch_vox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-axis center bounds that keep a full patch inside the scan."""
        shape = np.asarray(self.data.shape, dtype=np.int64)
        half_patch = (patch_vox // 2).astype(np.int64)
        min_idx = half_patch
        max_idx = shape - patch_vox + half_patch
        if np.any(max_idx < min_idx):
            raise SmallScanError(
                f"scan too small for patch extraction: shape={tuple(shape.tolist())} patch={tuple(patch_vox.tolist())}"
            )
        return min_idx, max_idx

    def _patch_has_overlap(self, center_vox: np.ndarray, patch_vox: np.ndarray) -> bool:
        """Check if a patch centered here fits fully inside the volume."""
        starts = center_vox - (patch_vox // 2)
        ends = starts + patch_vox
        vol_shape = np.asarray(self.data.shape, dtype=np.int64)
        return bool(np.all(starts >= 0) and np.all(ends <= vol_shape))

    def _extract_patches(self, centers_vox: np.ndarray, patch_vox: np.ndarray, output_size: int) -> np.ndarray:
        """Extract 2D native-plane patches via direct slicing, resize to output_size x output_size."""
        thin = self.geometry.thin_axis
        in_plane = [i for i in range(3) if i != thin]
        ax_r, ax_c = in_plane[0], in_plane[1]
        half = patch_vox // 2
        vol_shape = np.asarray(self.data.shape, dtype=np.int64)
        h_in = int(patch_vox[ax_r])
        w_in = int(patch_vox[ax_c])
        need_resize = (h_in != output_size or w_in != output_size)

        n = len(centers_vox)
        out = np.zeros((n, output_size, output_size), dtype=np.float32)

        for i, center in enumerate(centers_vox):
            starts = center - half
            ends = starts + patch_vox
            src_s = np.maximum(starts, 0)
            src_e = np.minimum(ends, vol_shape)
            if np.any(src_e <= src_s):
                continue
            dst_s = src_s - starts
            dst_e = dst_s + (src_e - src_s)

            # Slice the thin axis directly (always index 0 within the 1-thick dim)
            thin_idx = int(np.clip(center[thin], 0, vol_shape[thin] - 1))
            if thin == 0:
                plane = self.data[thin_idx, src_s[1]:src_e[1], src_s[2]:src_e[2]]
            elif thin == 1:
                plane = self.data[src_s[0]:src_e[0], thin_idx, src_s[2]:src_e[2]]
            else:
                plane = self.data[src_s[0]:src_e[0], src_s[1]:src_e[1], thin_idx]

            if need_resize:
                # Place into full-size native patch, then resize
                native = np.zeros((h_in, w_in), dtype=np.float32)
                native[dst_s[ax_r]:dst_e[ax_r], dst_s[ax_c]:dst_e[ax_c]] = plane
                out[i] = np.array(
                    Image.fromarray(native, mode="F").resize((output_size, output_size), Image.BILINEAR),
                    dtype=np.float32,
                )
            else:
                out[i, dst_s[ax_r]:dst_e[ax_r], dst_s[ax_c]:dst_e[ax_c]] = plane

        return out

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
        resolved_patch_size = int(self.target_patch_size if target_patch_size is None else target_patch_size)
        patch_vox = self._mm_patch_vox_shape(self.base_patch_mm)
        rng = np.random.default_rng(seed)

        min_center_idx, max_center_idx = self._center_bounds_for_full_patch(patch_vox)

        # No artificial radius clamping — use the requested radius as-is
        sampling_radius_mm = float(rng.uniform(20.0, 30.0)) if sampling_radius_mm is None else float(sampling_radius_mm)
        sampling_radius_mm = max(sampling_radius_mm, 0.0)

        if subset_center_vox is None:
            prism_center = self.sample_prism_center_vox(rng, patch_vox=patch_vox)
        else:
            prism_center = np.asarray(subset_center_vox, dtype=np.int64)
            if prism_center.shape != (3,):
                raise ValueError(f"subset_center_vox must have shape (3,), got {prism_center.shape}")
            prism_center = np.clip(prism_center, min_center_idx, max_center_idx)

        if patch_centers_vox is None:
            centers = []
            max_attempts = int(n_patches) * 4
            attempts = 0
            while len(centers) < int(n_patches) and attempts < max_attempts:
                delta_mm = rng.uniform(-sampling_radius_mm, sampling_radius_mm, size=3)
                if float(np.linalg.norm(delta_mm)) > sampling_radius_mm:
                    delta_mm = delta_mm * (sampling_radius_mm / max(np.linalg.norm(delta_mm), 1e-6))
                delta_vox = (self.affine_linear_inv @ np.asarray(delta_mm, dtype=np.float32)).astype(np.float32, copy=False)
                center = prism_center + np.rint(delta_vox).astype(np.int64)
                center = np.clip(center, min_center_idx, max_center_idx)
                attempts += 1
                if self._patch_has_overlap(center, patch_vox):
                    centers.append(center)
            # If we couldn't fill all patches, duplicate last valid or use prism center
            while len(centers) < int(n_patches):
                centers.append(centers[-1] if centers else prism_center.copy())
            centers_arr = np.asarray(centers, dtype=np.int64)
        else:
            centers_arr = np.asarray(patch_centers_vox, dtype=np.int64)
            if centers_arr.ndim != 2 or centers_arr.shape[1] != 3:
                raise ValueError(f"patch_centers_vox must have shape (N, 3), got {centers_arr.shape}")
            if int(centers_arr.shape[0]) != int(n_patches):
                raise ValueError(
                    f"patch_centers_vox has {centers_arr.shape[0]} centers but n_patches={int(n_patches)}"
                )

        raw_patches = self._extract_patches(centers_arr, patch_vox, resolved_patch_size)

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
            "target_patch_size": int(resolved_patch_size),
            "patch_vox_shape": tuple(int(v) for v in patch_vox.tolist()),
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
            "body_sampling_source": str(self.body_sampling_source),
        }


def _stable_seed_from_text(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _resolve_body_center_candidates(
    *,
    series_path: str,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    patch_vox: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    """Build a small set of valid body-center candidates from a TotalSegmentator volume."""
    ts_path = resolve_totalseg_total_ct_path(series_path)
    if not ts_path:
        return None, ""

    try:
        raw = nib.load(ts_path)
        try:
            img = nib.as_closest_canonical(raw)
        except Exception:
            img = raw
        seg = np.asarray(img.dataobj)
        if seg.ndim == 4:
            seg = seg[..., 0]
        if seg.ndim != 3:
            return None, ts_path
        if tuple(int(v) for v in seg.shape) != tuple(int(v) for v in reference_shape):
            return None, ts_path
        if not np.allclose(np.asarray(img.affine, dtype=np.float32), np.asarray(reference_affine, dtype=np.float32), atol=1e-2, rtol=1e-3):
            return None, ts_path

        patch = np.asarray(patch_vox, dtype=np.int64).reshape(3)
        shape = np.asarray(reference_shape, dtype=np.int64).reshape(3)
        half_patch = (patch // 2).astype(np.int64)
        min_idx = half_patch
        max_idx = shape - patch + half_patch
        if np.any(max_idx < min_idx):
            return None, ts_path

        lower = tuple(int(v) for v in min_idx.tolist())
        upper = tuple(int(v) + 1 for v in max_idx.tolist())
        cropped = seg[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        if cropped.size == 0:
            return None, ts_path

        centers: np.ndarray | None = None
        for step in _TOTALSEG_BODY_GRID_STEPS:
            view = np.asarray(cropped[::step, ::step, ::step] > 0, dtype=bool)
            coords = np.argwhere(view)
            if coords.size == 0:
                continue
            centers = min_idx[np.newaxis, :] + coords.astype(np.int64, copy=False) * int(step)
            break
        if centers is None or len(centers) == 0:
            return None, ts_path

        if len(centers) > _TOTALSEG_BODY_MAX_CANDIDATES:
            rng = np.random.default_rng(_stable_seed_from_text(series_path))
            choice = rng.choice(len(centers), size=_TOTALSEG_BODY_MAX_CANDIDATES, replace=False)
            centers = centers[np.asarray(choice, dtype=np.int64)]
        return np.asarray(centers, dtype=np.int32), ts_path
    except Exception:
        return None, ts_path


def load_nifti_scan(
    record: ScanRecord | dict[str, Any],
    base_patch_mm: float,
    *,
    use_totalseg_body_centers: bool = True,
) -> tuple[NiftiScan, str]:
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
        patch_vox = np.maximum(np.ceil(float(base_patch_mm) / np.clip(spacing, 1e-6, None)).astype(np.int64), 1)
        geometry = infer_scan_geometry(data.shape, spacing)
        patch_vox[int(geometry.thin_axis)] = 1
        body_center_candidates_vox, body_sampling_source = (
            _resolve_body_center_candidates(
                series_path=series_path,
                reference_shape=tuple(int(v) for v in data.shape),
                reference_affine=np.asarray(img.affine, dtype=np.float32),
                patch_vox=patch_vox,
            )
            if use_totalseg_body_centers
            else (None, "")
        )
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
            body_center_candidates_vox=body_center_candidates_vox,
            body_sampling_source=body_sampling_source,
        )
        return scan, effective_path
    except (NiftiResolutionError, NiftiLoadError):
        raise
    except Exception as exc:
        raise NiftiLoadError(f"Failed loading NIfTI '{effective_path}': {exc}") from exc
