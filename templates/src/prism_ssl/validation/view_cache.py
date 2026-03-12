"""CT semantic view validation cache helpers."""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np
import polars as pl
import torch

from prism_ssl.config import RunConfig
from prism_ssl.config.schema import ScanRecord
from prism_ssl.data import (
    apply_window_to_raw_patches,
    compute_patch_robust_stats,
    load_catalog,
    load_nifti_scan,
    normalize_series_description,
)
from prism_ssl.data.catalog import build_scan_id, protocol_key_from_row, series_id_from_row
from prism_ssl.data.filters import filter_modalities, filter_nonempty_series_path
from prism_ssl.data.preflight import resolve_totalseg_total_ct_path
from prism_ssl.utils.hashing import stable_int_hash

_CACHE_VERSION = 1
_SUMMARY_JSON = "summary.json"
_BUILD_CONFIG_JSON = "build_config.json"
_ELIGIBLE_SCANS_PARQUET = "eligible_scans.parquet"
_SCANS_PARQUET = "scans.parquet"
_VIEWS_PARQUET = "views.parquet"
_SHARDS_DIR = "shards"

_MAX_TARGET_CENTER_CANDIDATES_PER_SCAN = 1024
_MIN_TARGET_VOXEL_COUNT = 128
_TARGET_METADATA_SLACK_BYTES = 8_192

TOTALSEG_LABEL_NAME_TO_ID: dict[str, int] = {
    "spleen": 1,
    "kidney_right": 2,
    "stomach": 6,
    "pancreas": 7,
    "lung_upper_lobe_left": 10,
    "lung_lower_lobe_left": 11,
    "lung_upper_lobe_right": 12,
    "lung_middle_lobe_right": 13,
    "lung_lower_lobe_right": 14,
    "esophagus": 15,
    "urinary_bladder": 21,
    "heart": 51,
}
TOTALSEG_ID_TO_LABEL_NAME: dict[int, str] = {label_id: name for name, label_id in TOTALSEG_LABEL_NAME_TO_ID.items()}


@dataclass(frozen=True)
class SemanticTargetSpec:
    key: str
    source_label_names: tuple[str, ...]


@dataclass(frozen=True)
class ValidationBuilderSpec:
    n_scans: int
    views_per_scan: int
    max_cache_gb: float
    shard_size: int
    seed: int
    n_patches: int
    source_patch_mm_min: float
    source_patch_mm_max: float
    source_patch_mm_distribution: str
    max_window_retries: int = 8
    max_view_attempts: int = 16
    low_variation_std_threshold: float = 0.05
    max_saturation_fraction: float = 0.90
    include_raw_patches: bool = True
    include_normalized_patches: bool = True


SEMANTIC_TARGETS: tuple[SemanticTargetSpec, ...] = (
    SemanticTargetSpec("heart", ("heart",)),
    SemanticTargetSpec(
        "lung",
        (
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
        ),
    ),
    SemanticTargetSpec("esophagus", ("esophagus",)),
    SemanticTargetSpec("stomach", ("stomach",)),
    SemanticTargetSpec("kidney_right", ("kidney_right",)),
    SemanticTargetSpec("spleen", ("spleen",)),
    SemanticTargetSpec("bladder", ("urinary_bladder",)),
    SemanticTargetSpec("pancreas", ("pancreas",)),
)


def _target_label_ids() -> dict[str, tuple[int, ...]]:
    mapping: dict[str, tuple[int, ...]] = {}
    for spec in SEMANTIC_TARGETS:
        ids: list[int] = []
        for name in spec.source_label_names:
            if name not in TOTALSEG_LABEL_NAME_TO_ID:
                raise KeyError(f"Missing TotalSegmentator label mapping for '{name}'")
            ids.append(int(TOTALSEG_LABEL_NAME_TO_ID[name]))
        mapping[spec.key] = tuple(ids)
    return mapping


def estimate_ct_view_cache_bytes_per_view(
    n_patches: int,
    *,
    include_raw_patches: bool = True,
    include_normalized_patches: bool = True,
    patch_dtype: torch.dtype = torch.float16,
    position_dtype: torch.dtype = torch.float32,
    voxel_dtype: torch.dtype = torch.int32,
    metadata_slack_bytes: int = _TARGET_METADATA_SLACK_BYTES,
) -> int:
    patch_bytes = int(n_patches) * 16 * 16 * torch.tensor([], dtype=patch_dtype).element_size()
    tensor_bytes = 0
    if include_normalized_patches:
        tensor_bytes += patch_bytes
    if include_raw_patches:
        tensor_bytes += patch_bytes
    tensor_bytes += int(n_patches) * 3 * torch.tensor([], dtype=position_dtype).element_size()
    tensor_bytes += int(n_patches) * 3 * torch.tensor([], dtype=position_dtype).element_size()
    tensor_bytes += int(n_patches) * 3 * torch.tensor([], dtype=voxel_dtype).element_size()
    return int(tensor_bytes + int(metadata_slack_bytes))


def max_ct_view_cache_views_for_budget(
    n_patches: int,
    *,
    max_cache_gb: float,
    include_raw_patches: bool = True,
    include_normalized_patches: bool = True,
) -> int:
    budget_bytes = int(max(float(max_cache_gb), 0.0) * (1024**3))
    if budget_bytes <= 0:
        return 0
    per_view = estimate_ct_view_cache_bytes_per_view(
        n_patches,
        include_raw_patches=include_raw_patches,
        include_normalized_patches=include_normalized_patches,
    )
    return max(1, budget_bytes // max(per_view, 1))


def _stable_seed(*parts: object) -> int:
    return stable_int_hash("|".join(str(part) for part in parts)) & 0xFFFFFFFF


def _cache_summary_path(output_dir: Path) -> Path:
    return output_dir / _SUMMARY_JSON


def _cache_build_config_path(output_dir: Path) -> Path:
    return output_dir / _BUILD_CONFIG_JSON


def _cache_eligible_scans_path(output_dir: Path) -> Path:
    return output_dir / _ELIGIBLE_SCANS_PARQUET


def _cache_scans_path(output_dir: Path) -> Path:
    return output_dir / _SCANS_PARQUET


def _cache_views_path(output_dir: Path) -> Path:
    return output_dir / _VIEWS_PARQUET


def _cache_shards_dir(output_dir: Path) -> Path:
    return output_dir / _SHARDS_DIR


def _cache_shard_path(output_dir: Path, shard_index: int) -> Path:
    return _cache_shards_dir(output_dir) / f"shard_{int(shard_index):03d}.pt"


def _load_catalog_df(catalog: str | Path | pl.DataFrame) -> pl.DataFrame:
    if isinstance(catalog, pl.DataFrame):
        return catalog
    return load_catalog(str(catalog))


def _ct_catalog_rows(catalog: str | Path | pl.DataFrame) -> list[dict[str, Any]]:
    df = filter_nonempty_series_path(filter_modalities(_load_catalog_df(catalog), ("CT",)))
    if len(df) == 0:
        return []
    return df.to_dicts()


def _scan_record_from_row(row: Mapping[str, Any]) -> ScanRecord:
    return ScanRecord(
        scan_id=build_scan_id(dict(row)),
        series_id=series_id_from_row(dict(row)),
        modality=str(row.get("modality", "CT")).upper(),
        series_path=str(row.get("series_path", "")),
        series_description=str(row.get("series_description", "")),
    )


def _load_totalseg_segmentation(series_path: str, reference_shape: tuple[int, int, int], reference_affine: np.ndarray) -> tuple[np.ndarray | None, str]:
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
        if not np.allclose(
            np.asarray(img.affine, dtype=np.float32),
            np.asarray(reference_affine, dtype=np.float32),
            atol=1e-2,
            rtol=1e-3,
        ):
            return None, ts_path
        return np.asarray(seg, dtype=np.int16), ts_path
    except Exception:
        return None, ts_path


def _target_center_candidates(
    seg: np.ndarray,
    target_ids: Sequence[int],
    patch_vox: np.ndarray,
    *,
    seed: int,
    max_candidates: int = _MAX_TARGET_CENTER_CANDIDATES_PER_SCAN,
    min_voxel_count: int = _MIN_TARGET_VOXEL_COUNT,
) -> tuple[np.ndarray, int]:
    mask = np.isin(seg, np.asarray(list(target_ids), dtype=np.int16))
    coords = np.argwhere(mask)
    if len(coords) < int(min_voxel_count):
        return np.empty((0, 3), dtype=np.int32), int(len(coords))

    patch = np.asarray(patch_vox, dtype=np.int64).reshape(3)
    shape = np.asarray(seg.shape, dtype=np.int64).reshape(3)
    half_patch = (patch // 2).astype(np.int64)
    min_idx = half_patch
    max_idx = shape - patch + half_patch
    if np.any(max_idx < min_idx):
        return np.empty((0, 3), dtype=np.int32), 0

    valid = np.all((coords >= min_idx[np.newaxis, :]) & (coords <= max_idx[np.newaxis, :]), axis=1)
    coords = coords[valid]
    usable_count = int(len(coords))
    if usable_count < int(min_voxel_count):
        return np.empty((0, 3), dtype=np.int32), usable_count
    if usable_count > int(max_candidates):
        rng = np.random.default_rng(int(seed))
        choice = np.asarray(rng.choice(usable_count, size=int(max_candidates), replace=False), dtype=np.int64)
        coords = coords[choice]
    return np.asarray(coords, dtype=np.int32), usable_count


def _semantic_fraction_payload(seg: np.ndarray, patch_centers_vox: np.ndarray, target_label_ids: Mapping[str, Sequence[int]]) -> tuple[str, str, str, float]:
    centers = np.asarray(patch_centers_vox, dtype=np.int64)
    labels = seg[centers[:, 0], centers[:, 1], centers[:, 2]].astype(np.int32, copy=False)
    total = max(int(labels.size), 1)
    fractions: dict[str, float] = {}
    remaining = np.ones(labels.shape[0], dtype=bool)
    dominant_key = "other"
    dominant_exact_name = "background"
    dominant_exact_fraction = 0.0

    if labels.size > 0:
        unique_ids, counts = np.unique(labels, return_counts=True)
        top_idx = int(np.argmax(counts))
        dominant_exact_name = TOTALSEG_ID_TO_LABEL_NAME.get(int(unique_ids[top_idx]), "background")
        dominant_exact_fraction = float(counts[top_idx] / total)

    best_fraction = -1.0
    for target in SEMANTIC_TARGETS:
        ids = np.asarray(list(target_label_ids[target.key]), dtype=np.int32)
        hit = np.isin(labels, ids)
        frac = float(hit.sum() / total)
        fractions[target.key] = frac
        remaining &= ~hit
        if frac > best_fraction:
            best_fraction = frac
            dominant_key = target.key
    fractions["other"] = float(remaining.sum() / total)
    return json.dumps(fractions, sort_keys=True), str(dominant_key), str(dominant_exact_name), float(dominant_exact_fraction)


def _is_informative_view(
    normalized_patches: np.ndarray,
    *,
    std_threshold: float,
    max_saturation_fraction: float,
) -> tuple[bool, float]:
    arr = np.asarray(normalized_patches, dtype=np.float32)
    std_value = float(np.std(arr))
    low_fraction = float(np.mean(arr <= -0.98))
    high_fraction = float(np.mean(arr >= 0.98))
    informative = (
        std_value >= float(std_threshold)
        and low_fraction <= float(max_saturation_fraction)
        and high_fraction <= float(max_saturation_fraction)
    )
    return bool(informative), std_value


def _sample_source_patch_mm(spec: ValidationBuilderSpec, rng: np.random.Generator) -> float:
    lo = max(float(spec.source_patch_mm_min), 1e-3)
    hi = max(float(spec.source_patch_mm_max), lo)
    distribution = str(spec.source_patch_mm_distribution).strip().lower()
    if distribution == "log_uniform":
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def _eligible_scan_row(row: Mapping[str, Any], spec: ValidationBuilderSpec, target_label_ids: Mapping[str, Sequence[int]]) -> dict[str, Any] | None:
    record = _scan_record_from_row(row)
    try:
        scan, nifti_path = load_nifti_scan(
            record,
            base_patch_mm=float(spec.source_patch_mm_max),
            source_patch_mm_min=float(spec.source_patch_mm_min),
            source_patch_mm_max=float(spec.source_patch_mm_max),
            source_patch_mm_distribution=str(spec.source_patch_mm_distribution),
            use_totalseg_body_centers=False,
        )
    except Exception:
        return None

    seg, ts_path = _load_totalseg_segmentation(record.series_path, tuple(int(v) for v in scan.data.shape), scan.affine)
    if seg is None or not ts_path:
        return None

    patch_vox = scan.mm_patch_vox_shape(float(spec.source_patch_mm_max))
    counts: dict[str, int] = {}
    available: list[str] = []
    for target in SEMANTIC_TARGETS:
        _, usable_count = _target_center_candidates(
            seg,
            target_label_ids[target.key],
            patch_vox,
            seed=_stable_seed(spec.seed, record.scan_id, target.key, "eligible"),
        )
        counts[target.key] = int(usable_count)
        if usable_count >= _MIN_TARGET_VOXEL_COUNT:
            available.append(target.key)
    if not available:
        return None

    return {
        "scan_id": record.scan_id,
        "series_id": record.series_id,
        "protocol_key": protocol_key_from_row(dict(row)),
        "series_description": str(row.get("series_description", "")),
        "series_path": record.series_path,
        "nifti_path": str(nifti_path),
        "totalseg_path": str(ts_path),
        "shape_vox": [int(v) for v in scan.data.shape],
        "spacing_mm": [float(v) for v in scan.spacing.tolist()],
        "available_semantic_keys": list(available),
        "semantic_voxel_counts_json": json.dumps(counts, sort_keys=True),
        "n_available_semantic_keys": int(len(available)),
    }


def _select_validation_scans(eligible_rows: Sequence[Mapping[str, Any]], *, target_scans: int, seed: int) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in eligible_rows]
    if len(remaining) < int(target_scans):
        raise ValueError(f"Requested {target_scans} scans but only {len(remaining)} eligible scans are available")

    selected: list[dict[str, Any]] = []
    counts = {target.key: 0 for target in SEMANTIC_TARGETS}
    while len(selected) < int(target_scans):
        best_idx = -1
        best_score: tuple[float, int, int] | None = None
        for idx, row in enumerate(remaining):
            keys = [str(k) for k in row.get("available_semantic_keys", [])]
            primary = float(sum(1.0 / (1.0 + counts.get(key, 0)) for key in keys))
            secondary = int(len(keys))
            tertiary = -int(_stable_seed(seed, row["scan_id"], "select"))
            score = (primary, secondary, tertiary)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        for key in chosen.get("available_semantic_keys", []):
            counts[str(key)] += 1
    return selected


def _target_schedule(available_keys: Sequence[str], views_per_scan: int) -> list[str]:
    available = {str(k) for k in available_keys}
    canonical = [target.key for target in SEMANTIC_TARGETS if target.key in available]
    if not canonical:
        return []
    schedule: list[str] = []
    rounds = max(1, math.ceil(int(views_per_scan) / max(len(SEMANTIC_TARGETS), 1)))
    for _ in range(rounds):
        schedule.extend(canonical)
    if len(schedule) < int(views_per_scan):
        idx = 0
        while len(schedule) < int(views_per_scan):
            schedule.append(canonical[idx % len(canonical)])
            idx += 1
    return schedule[: int(views_per_scan)]


def _build_scan_target_banks(
    scan_id: str,
    seg: np.ndarray,
    scan_patch_vox: np.ndarray,
    target_label_ids: Mapping[str, Sequence[int]],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    banks: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for target in SEMANTIC_TARGETS:
        centers, usable_count = _target_center_candidates(
            seg,
            target_label_ids[target.key],
            scan_patch_vox,
            seed=_stable_seed(seed, scan_id, target.key, "bank"),
        )
        counts[target.key] = int(usable_count)
        if len(centers) > 0:
            banks[target.key] = centers
    return banks, counts


def _build_single_semantic_view(
    *,
    scan: Any,
    seg: np.ndarray,
    scan_row: Mapping[str, Any],
    scan_index: int,
    view_index: int,
    target_key: str,
    target_centers: np.ndarray,
    spec: ValidationBuilderSpec,
    target_label_ids: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    for view_attempt in range(int(spec.max_view_attempts)):
        rng = np.random.default_rng(_stable_seed(spec.seed, scan_row["scan_id"], view_index, target_key, view_attempt))
        center_idx = int(rng.integers(low=0, high=len(target_centers)))
        prism_center = np.asarray(target_centers[center_idx], dtype=np.int64)
        source_patch_mm = _sample_source_patch_mm(spec, rng)
        raw_view = scan.sample_view_raw(
            int(spec.n_patches),
            seed=_stable_seed(spec.seed, scan_row["scan_id"], view_index, target_key, view_attempt, "raw"),
            source_patch_mm=float(source_patch_mm),
            subset_center_vox=prism_center,
            sampled_body_center=False,
        )
        raw_patches = np.asarray(raw_view["raw_patches"], dtype=np.float32)
        raw_median, raw_std, _, _ = compute_patch_robust_stats(raw_patches)
        for window_retry in range(int(spec.max_window_retries)):
            window_rng = np.random.default_rng(
                _stable_seed(spec.seed, scan_row["scan_id"], view_index, target_key, view_attempt, "window", window_retry)
            )
            wc = float(window_rng.uniform(raw_median - raw_std, raw_median + raw_std))
            ww = float(window_rng.uniform(2.0 * raw_std, 6.0 * raw_std))
            normalized, w_min, w_max = apply_window_to_raw_patches(raw_patches, wc=wc, ww=ww)
            informative, normalized_std = _is_informative_view(
                normalized,
                std_threshold=float(spec.low_variation_std_threshold),
                max_saturation_fraction=float(spec.max_saturation_fraction),
            )
            if not informative:
                continue
            semantic_fraction_json, dominant_semantic_key, dominant_exact_label_name, dominant_exact_fraction = _semantic_fraction_payload(
                seg,
                np.asarray(raw_view["patch_centers_vox"], dtype=np.int64),
                target_label_ids,
            )
            return {
                "view_row": {
                    "view_index": int(view_index),
                    "scan_index": int(scan_index),
                    "scan_id": str(scan_row["scan_id"]),
                    "series_id": str(scan_row["series_id"]),
                    "protocol_key": str(scan_row["protocol_key"]),
                    "series_description": str(scan_row["series_description"]),
                    "semantic_target_key": str(target_key),
                    "dominant_semantic_key": str(dominant_semantic_key),
                    "dominant_totalseg_label_name": str(dominant_exact_label_name),
                    "dominant_totalseg_label_fraction": float(dominant_exact_fraction),
                    "semantic_fraction_json": str(semantic_fraction_json),
                    "prism_center_vox": [int(v) for v in np.asarray(raw_view["prism_center_vox"], dtype=np.int64).tolist()],
                    "prism_center_pt": [float(v) for v in np.asarray(raw_view["prism_center_pt"], dtype=np.float32).tolist()],
                    "patch_vox_shape": [int(v) for v in np.asarray(raw_view["patch_vox_shape"], dtype=np.int64).tolist()],
                    "source_patch_mm": float(raw_view["source_patch_mm"]),
                    "sampling_radius_mm": float(raw_view["sampling_radius_mm"]),
                    "wc": float(wc),
                    "ww": float(ww),
                    "w_min": float(w_min),
                    "w_max": float(w_max),
                    "raw_patch_median": float(raw_median),
                    "raw_patch_std": float(raw_std),
                    "normalized_patch_std": float(normalized_std),
                    "window_retry_count": int(window_retry),
                    "view_attempt_count": int(view_attempt),
                    "sampled_body_center": bool(raw_view.get("sampled_body_center", False)),
                    "native_acquisition_plane": str(raw_view["native_acquisition_plane"]),
                    "native_thin_axis_name": str(raw_view["native_thin_axis_name"]),
                },
                "tensor_payload": {
                    "normalized_patches": np.asarray(normalized[..., np.newaxis], dtype=np.float16),
                    "raw_patches": np.asarray(raw_patches[..., np.newaxis], dtype=np.float16),
                    "relative_patch_centers_pt": np.asarray(raw_view["relative_patch_centers_pt"], dtype=np.float32),
                    "patch_centers_pt": np.asarray(raw_view["patch_centers_pt"], dtype=np.float32),
                    "patch_centers_vox": np.asarray(raw_view["patch_centers_vox"], dtype=np.int32),
                },
            }
    raise RuntimeError(
        f"Failed to build informative semantic view for scan={scan_row['scan_id']} target={target_key} after {spec.max_view_attempts} attempts"
    )


def _flush_shard_chunk(
    output_dir: Path,
    *,
    shard_index: int,
    rows: Sequence[Mapping[str, Any]],
    tensors: Sequence[Mapping[str, np.ndarray]],
) -> int:
    payload = {
        "view_index": torch.tensor([int(row["view_index"]) for row in rows], dtype=torch.int64),
        "normalized_patches": torch.from_numpy(np.stack([np.asarray(item["normalized_patches"], dtype=np.float16) for item in tensors], axis=0)),
        "raw_patches": torch.from_numpy(np.stack([np.asarray(item["raw_patches"], dtype=np.float16) for item in tensors], axis=0)),
        "relative_patch_centers_pt": torch.from_numpy(np.stack([np.asarray(item["relative_patch_centers_pt"], dtype=np.float32) for item in tensors], axis=0)),
        "patch_centers_pt": torch.from_numpy(np.stack([np.asarray(item["patch_centers_pt"], dtype=np.float32) for item in tensors], axis=0)),
        "patch_centers_vox": torch.from_numpy(np.stack([np.asarray(item["patch_centers_vox"], dtype=np.int32) for item in tensors], axis=0)),
    }
    shard_path = _cache_shard_path(output_dir, shard_index)
    torch.save(payload, shard_path)
    return int(shard_path.stat().st_size)


def build_ct_view_validation_cache(
    catalog: str | Path | pl.DataFrame,
    config: RunConfig,
    output_dir: str | Path,
    *,
    target_scans: int = 128,
    views_per_scan: int = 16,
    seed: int = 42,
    max_cache_gb: float = 5.0,
    shard_size: int = 256,
    overwrite: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    spec = ValidationBuilderSpec(
        n_scans=int(target_scans),
        views_per_scan=int(views_per_scan),
        max_cache_gb=float(max_cache_gb),
        shard_size=int(shard_size),
        seed=int(seed),
        n_patches=int(config.data.n_patches),
        source_patch_mm_min=float(config.data.source_patch_mm_min),
        source_patch_mm_max=float(config.data.source_patch_mm_max),
        source_patch_mm_distribution=str(config.data.source_patch_mm_distribution),
    )
    target_label_ids = _target_label_ids()
    output_root = Path(output_dir).expanduser().resolve()
    summary_path = _cache_summary_path(output_root)
    if summary_path.exists() and not overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    if overwrite and output_root.exists():
        for stale in (
            summary_path,
            _cache_build_config_path(output_root),
            _cache_eligible_scans_path(output_root),
            _cache_scans_path(output_root),
            _cache_views_path(output_root),
        ):
            if stale.exists():
                stale.unlink()
        shutil.rmtree(_cache_shards_dir(output_root), ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    _cache_shards_dir(output_root).mkdir(parents=True, exist_ok=True)

    requested_views = int(spec.n_scans * spec.views_per_scan)
    max_views = max_ct_view_cache_views_for_budget(
        int(spec.n_patches),
        max_cache_gb=float(spec.max_cache_gb),
        include_raw_patches=bool(spec.include_raw_patches),
        include_normalized_patches=bool(spec.include_normalized_patches),
    )
    estimated_bytes_per_view = estimate_ct_view_cache_bytes_per_view(
        int(spec.n_patches),
        include_raw_patches=bool(spec.include_raw_patches),
        include_normalized_patches=bool(spec.include_normalized_patches),
    )
    if requested_views > max_views:
        raise ValueError(
            f"Requested {requested_views} cached views but budget {spec.max_cache_gb:.2f} GiB only fits about {max_views}"
        )

    if progress is not None:
        progress(
            {
                "stage": "eligible_scans",
                "status": "start",
                "requested_scans": int(spec.n_scans),
                "requested_views": int(requested_views),
                "estimated_bytes_per_view": int(estimated_bytes_per_view),
            }
        )

    eligible_rows: list[dict[str, Any]] = []
    for row in _ct_catalog_rows(catalog):
        eligible = _eligible_scan_row(row, spec, target_label_ids)
        if eligible is not None:
            eligible_rows.append(eligible)

    eligible_df = pl.DataFrame(eligible_rows).sort("scan_id") if eligible_rows else pl.DataFrame([])
    eligible_df.write_parquet(_cache_eligible_scans_path(output_root))
    if len(eligible_df) < int(spec.n_scans):
        raise RuntimeError(f"Only {len(eligible_df)} eligible CT scans with TotalSegmentator targets found; need {spec.n_scans}")

    selected_rows = _select_validation_scans(eligible_rows, target_scans=int(spec.n_scans), seed=int(spec.seed))

    if progress is not None:
        progress(
            {
                "stage": "selected_scans",
                "status": "complete",
                "eligible_scans": int(len(eligible_rows)),
                "selected_scans": int(len(selected_rows)),
            }
        )

    _cache_build_config_path(output_root).write_text(
        json.dumps(
            {
                "spec": asdict(spec),
                "semantic_targets": [asdict(target) for target in SEMANTIC_TARGETS],
                "totalseg_label_name_to_id": TOTALSEG_LABEL_NAME_TO_ID,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_scan_rows: list[dict[str, Any]] = []
    all_view_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    chunk_tensors: list[dict[str, np.ndarray]] = []
    total_bytes_written = 0
    selected_scan_count = 0

    for scan_index, scan_row in enumerate(selected_rows):
        record = ScanRecord(
            scan_id=str(scan_row["scan_id"]),
            series_id=str(scan_row["series_id"]),
            modality="CT",
            series_path=str(scan_row["series_path"]),
            series_description=str(scan_row["series_description"]),
            nifti_path=str(scan_row["nifti_path"]),
        )
        scan, _ = load_nifti_scan(
            record,
            base_patch_mm=float(spec.source_patch_mm_max),
            source_patch_mm_min=float(spec.source_patch_mm_min),
            source_patch_mm_max=float(spec.source_patch_mm_max),
            source_patch_mm_distribution=str(spec.source_patch_mm_distribution),
            use_totalseg_body_centers=False,
        )
        seg, ts_path = _load_totalseg_segmentation(record.series_path, tuple(int(v) for v in scan.data.shape), scan.affine)
        if seg is None or not ts_path:
            raise RuntimeError(f"Selected scan lost TotalSegmentator segmentation: {scan_row['scan_id']}")

        banks, usable_counts = _build_scan_target_banks(
            str(scan_row["scan_id"]),
            seg,
            scan.mm_patch_vox_shape(float(spec.source_patch_mm_max)),
            target_label_ids,
            seed=int(spec.seed),
        )
        available_targets = [target.key for target in SEMANTIC_TARGETS if target.key in banks]
        schedule = _target_schedule(available_targets, int(spec.views_per_scan))
        if len(schedule) != int(spec.views_per_scan):
            raise RuntimeError(f"Could not derive target schedule for scan={scan_row['scan_id']}")

        scan_view_rows: list[dict[str, Any]] = []
        scan_semantic_counts = {target.key: 0 for target in SEMANTIC_TARGETS}
        for local_view_index, target_key in enumerate(schedule):
            built = _build_single_semantic_view(
                scan=scan,
                seg=seg,
                scan_row=scan_row,
                scan_index=int(scan_index),
                view_index=int(scan_index * spec.views_per_scan + local_view_index),
                target_key=str(target_key),
                target_centers=banks[str(target_key)],
                spec=spec,
                target_label_ids=target_label_ids,
            )
            view_row = dict(built["view_row"])
            current_global_index = int(len(all_view_rows) + len(chunk_rows))
            view_row["shard_index"] = int(current_global_index // int(spec.shard_size))
            view_row["view_index_in_shard"] = int(current_global_index % int(spec.shard_size))
            chunk_rows.append(view_row)
            chunk_tensors.append(dict(built["tensor_payload"]))
            scan_view_rows.append(view_row)
            scan_semantic_counts[str(target_key)] += 1

            if len(chunk_rows) >= int(spec.shard_size):
                shard_index = int(len(all_view_rows) // int(spec.shard_size))
                total_bytes_written += _flush_shard_chunk(output_root, shard_index=shard_index, rows=chunk_rows, tensors=chunk_tensors)
                all_view_rows.extend(chunk_rows)
                chunk_rows = []
                chunk_tensors = []
                gc.collect()

        if len(scan_view_rows) != int(spec.views_per_scan):
            raise RuntimeError(f"Scan {scan_row['scan_id']} produced {len(scan_view_rows)} views; expected {spec.views_per_scan}")

        all_scan_rows.append(
            {
                "scan_index": int(scan_index),
                "scan_id": str(scan_row["scan_id"]),
                "series_id": str(scan_row["series_id"]),
                "protocol_key": str(scan_row["protocol_key"]),
                "series_description": str(scan_row["series_description"]),
                "series_path": str(scan_row["series_path"]),
                "nifti_path": str(scan_row["nifti_path"]),
                "totalseg_path": str(ts_path),
                "shape_vox": [int(v) for v in scan.data.shape],
                "spacing_mm": [float(v) for v in scan.spacing.tolist()],
                "selected_view_count": int(len(scan_view_rows)),
                "selected_semantic_counts_json": json.dumps(scan_semantic_counts, sort_keys=True),
            }
        )
        selected_scan_count += 1
        if progress is not None:
            progress(
                {
                    "stage": "materialize",
                    "status": "scan_complete",
                    "scan_index": int(scan_index),
                    "selected_scans": int(selected_scan_count),
                    "target_scans": int(spec.n_scans),
                    "available_targets": list(available_targets),
                    "usable_counts": dict(usable_counts),
                }
            )

    if chunk_rows:
        shard_index = int(len(all_view_rows) // int(spec.shard_size))
        total_bytes_written += _flush_shard_chunk(output_root, shard_index=shard_index, rows=chunk_rows, tensors=chunk_tensors)
        all_view_rows.extend(chunk_rows)
        chunk_rows = []
        chunk_tensors = []
        gc.collect()

    scans_df = pl.DataFrame(all_scan_rows).sort("scan_index")
    views_df = pl.DataFrame(all_view_rows).sort("view_index")
    scans_df.write_parquet(_cache_scans_path(output_root))
    views_df.write_parquet(_cache_views_path(output_root))

    summary = {
        "cache_version": _CACHE_VERSION,
        "cache_type": "ct_semantic_view_validation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(output_root),
        "catalog_path": str(catalog) if isinstance(catalog, (str, Path)) else "<dataframe>",
        "target_scans_requested": int(spec.n_scans),
        "views_per_scan": int(spec.views_per_scan),
        "target_views_requested": int(requested_views),
        "n_scans": int(len(all_scan_rows)),
        "n_views": int(len(all_view_rows)),
        "n_shards": int(math.ceil(len(all_view_rows) / max(int(spec.shard_size), 1))),
        "shard_size": int(spec.shard_size),
        "max_cache_gb": float(spec.max_cache_gb),
        "estimated_bytes_per_view": int(estimated_bytes_per_view),
        "estimated_total_bytes": int(estimated_bytes_per_view * len(all_view_rows)),
        "actual_tensor_bytes": int(total_bytes_written),
        "n_patches": int(spec.n_patches),
        "source_patch_mm_min": float(spec.source_patch_mm_min),
        "source_patch_mm_max": float(spec.source_patch_mm_max),
        "source_patch_mm_distribution": str(spec.source_patch_mm_distribution),
        "semantic_target_keys": [target.key for target in SEMANTIC_TARGETS],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_ct_view_validation_cache(
    cache_dir: str | Path,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cache_root = Path(cache_dir).expanduser().resolve()
    summary_path = _cache_summary_path(cache_root)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Validation cache summary not found: {summary_path}")
    shard_dir = _cache_shards_dir(cache_root)
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No validation cache shards found under {shard_dir}")

    if progress is not None:
        progress({"stage": "load", "status": "start", "cache_dir": str(cache_root), "n_shards": int(len(shard_paths))})

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eligible_df = pl.read_parquet(_cache_eligible_scans_path(cache_root))
    scans_df = pl.read_parquet(_cache_scans_path(cache_root)).sort("scan_index")
    views_df = pl.read_parquet(_cache_views_path(cache_root)).sort("view_index")

    payloads = [torch.load(path, map_location="cpu") for path in shard_paths]
    cache = {
        "summary": summary,
        "eligible_scans_df": eligible_df,
        "scans_df": scans_df,
        "views_df": views_df,
        "view_index": torch.cat([payload["view_index"] for payload in payloads], dim=0),
        "normalized_patches": torch.cat([payload["normalized_patches"] for payload in payloads], dim=0),
        "raw_patches": torch.cat([payload["raw_patches"] for payload in payloads], dim=0),
        "relative_patch_centers_pt": torch.cat([payload["relative_patch_centers_pt"] for payload in payloads], dim=0),
        "patch_centers_pt": torch.cat([payload["patch_centers_pt"] for payload in payloads], dim=0),
        "patch_centers_vox": torch.cat([payload["patch_centers_vox"] for payload in payloads], dim=0),
    }
    if progress is not None:
        progress({"stage": "load", "status": "complete", "cache_dir": str(cache_root), "n_views": int(cache["normalized_patches"].shape[0])})
    return cache


def build_scan_view_index(cache: Mapping[str, Any]) -> dict[str, list[int]]:
    scan_index: dict[str, list[int]] = {}
    for row in cache["views_df"].to_dicts():
        scan_id = str(row["scan_id"])
        scan_index.setdefault(scan_id, []).append(int(row["view_index"]))
    return scan_index


def build_ordered_within_scan_view_pairs(
    cache: Mapping[str, Any],
    *,
    include_self: bool = True,
) -> pl.DataFrame:
    views_df = cache["views_df"].sort(["scan_index", "view_index"])
    rows: list[dict[str, Any]] = []
    for _, group in views_df.group_by("scan_index", maintain_order=True):
        group_rows = group.to_dicts()
        for row_a in group_rows:
            pt_a = np.asarray(row_a["prism_center_pt"], dtype=np.float32)
            for row_b in group_rows:
                if not include_self and int(row_a["view_index"]) == int(row_b["view_index"]):
                    continue
                pt_b = np.asarray(row_b["prism_center_pt"], dtype=np.float32)
                center_delta = pt_b - pt_a
                rows.append(
                    {
                        "scan_index": int(row_a["scan_index"]),
                        "view_index_a": int(row_a["view_index"]),
                        "view_index_b": int(row_b["view_index"]),
                        "center_delta_mm": [float(v) for v in center_delta.tolist()],
                        "center_distance_mm": float(np.linalg.norm(center_delta)),
                        "window_delta": [
                            float(float(row_b["wc"]) - float(row_a["wc"])),
                            float(float(row_b["ww"]) - float(row_a["ww"])),
                        ],
                        "series_id_a": str(row_a["series_id"]),
                        "series_id_b": str(row_b["series_id"]),
                        "protocol_key_a": str(row_a["protocol_key"]),
                        "protocol_key_b": str(row_b["protocol_key"]),
                    }
                )
    return pl.DataFrame(rows).sort(["scan_index", "view_index_a", "view_index_b"])


def build_view_tensor_batch(
    cache: Mapping[str, Any],
    view_indices: Sequence[int],
) -> dict[str, torch.Tensor]:
    idx = torch.tensor([int(i) for i in view_indices], dtype=torch.long)
    views_df = cache["views_df"]
    selected_df = views_df.filter(pl.col("view_index").is_in([int(i) for i in view_indices])).sort("view_index")
    if selected_df.height != len(view_indices):
        raise ValueError("Requested view indices are not all present in the cache")
    return {
        "view_index": cache["view_index"][idx],
        "normalized_patches": cache["normalized_patches"][idx],
        "raw_patches": cache["raw_patches"][idx],
        "relative_patch_centers_pt": cache["relative_patch_centers_pt"][idx],
        "patch_centers_pt": cache["patch_centers_pt"][idx],
        "patch_centers_vox": cache["patch_centers_vox"][idx],
        "series_id": [str(v) for v in selected_df["series_id"].to_list()],
        "protocol_key": [str(v) for v in selected_df["protocol_key"].to_list()],
        "source_patch_mm": torch.tensor([float(v) for v in selected_df["source_patch_mm"].to_list()], dtype=torch.float32),
        "wc": torch.tensor([float(v) for v in selected_df["wc"].to_list()], dtype=torch.float32),
        "ww": torch.tensor([float(v) for v in selected_df["ww"].to_list()], dtype=torch.float32),
    }


__all__ = [
    "SemanticTargetSpec",
    "ValidationBuilderSpec",
    "SEMANTIC_TARGETS",
    "TOTALSEG_LABEL_NAME_TO_ID",
    "estimate_ct_view_cache_bytes_per_view",
    "max_ct_view_cache_views_for_budget",
    "build_ct_view_validation_cache",
    "load_ct_view_validation_cache",
    "build_scan_view_index",
    "build_ordered_within_scan_view_pairs",
    "build_view_tensor_batch",
]
