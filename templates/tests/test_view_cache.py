from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl

from prism_ssl.config.schema import RunConfig
from prism_ssl.validation import (
    TOTALSEG_LABEL_NAME_TO_ID,
    build_ct_view_validation_cache,
    build_ordered_within_scan_view_pairs,
    build_scan_view_index,
    build_view_tensor_batch,
    estimate_ct_view_cache_bytes_per_view,
    load_ct_view_validation_cache,
    max_ct_view_cache_views_for_budget,
)


def _make_scan_dirs(tmp_path: Path, name: str) -> tuple[Path, Path]:
    series_dir = tmp_path / "subjects" / "patient_a" / "study_a" / name
    ts_dir = tmp_path / "processing" / "totalsegmentator" / "patient_a" / "study_a" / name
    series_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)
    return series_dir, ts_dir


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4, dtype=np.float32)), str(path))


def _add_cube(seg: np.ndarray, center: tuple[int, int, int], radius: int, label_id: int) -> None:
    x0, y0, z0 = center
    seg[max(0, x0 - radius): x0 + radius, max(0, y0 - radius): y0 + radius, max(0, z0 - radius): z0 + radius] = int(label_id)


def _catalog_row(tmp_path: Path, name: str, description: str, labels: list[tuple[str, tuple[int, int, int]]], *, with_ts: bool = True) -> dict[str, object]:
    series_dir, ts_dir = _make_scan_dirs(tmp_path, name)
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFFFFFF)
    data = rng.normal(loc=25.0, scale=30.0, size=(48, 48, 16)).astype(np.float32)
    seg = np.zeros((48, 48, 16), dtype=np.int16)

    for label_name, center in labels:
        label_id = TOTALSEG_LABEL_NAME_TO_ID[label_name]
        _add_cube(seg, center, radius=3, label_id=label_id)
        cx, cy, cz = center
        data[max(0, cx - 4): cx + 4, max(0, cy - 4): cy + 4, max(0, cz - 1): cz + 1] += 150.0

    _write_nifti(series_dir / f"{name}.nii.gz", data)
    if with_ts:
        nib.save(
            nib.Nifti1Image(seg.astype(np.int16), affine=np.eye(4, dtype=np.float32)),
            str(ts_dir / f"{name}_e1_ts_total_ct.nii.gz"),
        )

    return {
        "pmbb_id": f"p_{name}",
        "modality": "CT",
        "series_path": str(series_dir),
        "series_description": description,
    }


def _test_config() -> RunConfig:
    config = RunConfig()
    config.data.n_patches = 8
    config.data.source_patch_mm_min = 16.0
    config.data.source_patch_mm_max = 16.0
    config.data.source_patch_mm_distribution = "uniform"
    return config


def test_cache_size_estimators_are_positive() -> None:
    per_view = estimate_ct_view_cache_bytes_per_view(8)
    max_views = max_ct_view_cache_views_for_budget(8, max_cache_gb=0.01)
    assert per_view > 0
    assert max_views > 0


def test_totalseg_mapping_contains_required_labels() -> None:
    for name in (
        "heart",
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
        "esophagus",
        "stomach",
        "kidney_right",
        "spleen",
        "urinary_bladder",
        "pancreas",
    ):
        assert name in TOTALSEG_LABEL_NAME_TO_ID


def test_ct_view_cache_build_and_load(tmp_path: Path) -> None:
    rows = [
        _catalog_row(
            tmp_path,
            "series_heart_lung",
            "AX BONE",
            [("heart", (20, 20, 8)), ("lung_upper_lobe_left", (12, 12, 8)), ("lung_lower_lobe_right", (28, 28, 8))],
        ),
        _catalog_row(
            tmp_path,
            "series_abdomen",
            "AX SOFT TISSUE",
            [("spleen", (14, 30, 8)), ("stomach", (24, 16, 8)), ("pancreas", (22, 24, 8))],
        ),
        _catalog_row(
            tmp_path,
            "series_pelvis",
            "AX IV CONTRAST",
            [("urinary_bladder", (18, 18, 8)), ("kidney_right", (30, 24, 8)), ("esophagus", (10, 22, 8))],
        ),
        _catalog_row(
            tmp_path,
            "series_missing_ts",
            "AX MISSING TS",
            [("heart", (20, 20, 8))],
            with_ts=False,
        ),
    ]
    catalog = pl.DataFrame(rows)
    output_dir = tmp_path / "cache_out"

    summary = build_ct_view_validation_cache(
        catalog,
        _test_config(),
        output_dir,
        target_scans=2,
        views_per_scan=4,
        seed=7,
        max_cache_gb=0.5,
        shard_size=3,
        overwrite=True,
    )

    assert summary["n_scans"] == 2
    assert summary["n_views"] == 8
    assert summary["actual_tensor_bytes"] > 0
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "build_config.json").is_file()
    assert (output_dir / "eligible_scans.parquet").is_file()
    assert (output_dir / "scans.parquet").is_file()
    assert (output_dir / "views.parquet").is_file()
    assert len(sorted((output_dir / "shards").glob("shard_*.pt"))) >= 1

    eligible = pl.read_parquet(output_dir / "eligible_scans.parquet")
    assert eligible.height == 2

    cache = load_ct_view_validation_cache(output_dir)
    assert cache["scans_df"].height == 2
    assert cache["views_df"].height == 8
    assert tuple(cache["normalized_patches"].shape[1:]) == (8, 16, 16, 1)
    assert tuple(cache["raw_patches"].shape[1:]) == (8, 16, 16, 1)
    assert tuple(cache["relative_patch_centers_pt"].shape[1:]) == (8, 3)

    scan_view_index = build_scan_view_index(cache)
    assert len(scan_view_index) == 2
    assert all(len(indices) == 4 for indices in scan_view_index.values())

    pair_df = build_ordered_within_scan_view_pairs(cache, include_self=True)
    assert pair_df.height == 2 * 16
    no_self_df = build_ordered_within_scan_view_pairs(cache, include_self=False)
    assert no_self_df.height == 2 * 12

    first_pair = pair_df.sort(["scan_index", "view_index_a", "view_index_b"]).row(0, named=True)
    views_df = cache["views_df"].sort("view_index")
    row_a = views_df.filter(pl.col("view_index") == int(first_pair["view_index_a"])).row(0, named=True)
    row_b = views_df.filter(pl.col("view_index") == int(first_pair["view_index_b"])).row(0, named=True)
    center_delta = np.asarray(row_b["prism_center_pt"], dtype=np.float32) - np.asarray(row_a["prism_center_pt"], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(first_pair["center_delta_mm"], dtype=np.float32), center_delta, atol=1e-5)
    assert abs(float(first_pair["center_distance_mm"]) - float(np.linalg.norm(center_delta))) < 1e-5

    batch = build_view_tensor_batch(cache, [0, 1])
    assert tuple(batch["normalized_patches"].shape) == (2, 8, 16, 16, 1)
    assert tuple(batch["raw_patches"].shape) == (2, 8, 16, 16, 1)
    assert tuple(batch["relative_patch_centers_pt"].shape) == (2, 8, 3)
    assert len(batch["series_id"]) == 2
    assert len(batch["protocol_key"]) == 2

    semantic_counts = json.loads(cache["scans_df"].row(0, named=True)["selected_semantic_counts_json"])
    assert sum(int(v) for v in semantic_counts.values()) == 4
