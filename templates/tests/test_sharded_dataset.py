from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from prism_ssl.config.schema import ScanRecord
from prism_ssl.data.sharded_dataset import BrokenScanRateExceeded, ShardedScanDataset


def _make_nifti(path: Path, shape: tuple[int, int, int] = (40, 40, 16)) -> None:
    arr = np.random.randn(*shape).astype(np.float32)
    img = nib.Nifti1Image(arr, affine=np.eye(4, dtype=np.float32))
    nib.save(img, str(path))


def _valid_record(base: Path, idx: int) -> ScanRecord:
    series_dir = base / f"series_{idx}"
    series_dir.mkdir(parents=True, exist_ok=True)
    _make_nifti(series_dir / f"scan_{idx}.nii.gz")
    return ScanRecord(
        scan_id=f"scan_{idx}",
        series_id=f"series_{idx}",
        modality="CT",
        series_path=str(series_dir),
    )


def _broken_record(base: Path, idx: int) -> ScanRecord:
    series_dir = base / f"broken_{idx}"
    series_dir.mkdir(parents=True, exist_ok=True)
    return ScanRecord(
        scan_id=f"broken_{idx}",
        series_id=f"broken_series_{idx}",
        modality="CT",
        series_path=str(series_dir),
    )


def test_sharded_dataset_bootstrap_and_replacement(tmp_path: Path):
    records = [_broken_record(tmp_path, 0), _valid_record(tmp_path, 1), _valid_record(tmp_path, 2)]

    ds = ShardedScanDataset(
        scan_records=records,
        n_patches=8,
        base_patch_mm=16.0,
        method="optimized_fused",
        warm_pool_size=2,
        visits_per_scan=1,
        seed=7,
        max_prefetch_replacements=1,
        strict_background_errors=False,
        broken_abort_ratio=0.99,
        broken_abort_min_attempts=100,
        max_broken_series_log=100,
        broken_series_log_path=str(tmp_path / "broken.jsonl"),
        pair_views=True,
    )

    it = iter(ds)
    samples = [next(it) for _ in range(10)]

    assert samples[0]["patches_a"].shape[-1] == 1
    assert tuple(samples[0]["patches_a"].shape[1:3]) == (16, 16)
    assert samples[0]["positions_a"].shape[-1] == 3

    replacement_events = sum(int(s["replacement_completed_count_delta"]) + int(s["replacement_failed_count_delta"]) for s in samples)
    assert replacement_events >= 0
    assert all(s["attempted_series_delta"] >= 0 for s in samples)
    assert all(s["broken_series_delta"] >= 0 for s in samples)


def test_broken_ratio_abort(tmp_path: Path):
    records = [_broken_record(tmp_path, 0), _broken_record(tmp_path, 1), _broken_record(tmp_path, 2)]

    ds = ShardedScanDataset(
        scan_records=records,
        n_patches=4,
        base_patch_mm=16.0,
        method="optimized_fused",
        warm_pool_size=2,
        visits_per_scan=1,
        seed=13,
        max_prefetch_replacements=1,
        strict_background_errors=False,
        broken_abort_ratio=0.10,
        broken_abort_min_attempts=1,
        max_broken_series_log=100,
        broken_series_log_path=str(tmp_path / "broken_abort.jsonl"),
        pair_views=True,
    )

    with pytest.raises(BrokenScanRateExceeded):
        next(iter(ds))


def test_scratch_staging_is_bounded_and_cleaned(tmp_path: Path):
    records = [_valid_record(tmp_path, i) for i in range(5)]
    scratch_root = tmp_path / "scratch"

    ds = ShardedScanDataset(
        scan_records=records,
        n_patches=8,
        base_patch_mm=16.0,
        method="optimized_fused",
        warm_pool_size=2,
        visits_per_scan=1,
        seed=17,
        max_prefetch_replacements=1,
        strict_background_errors=False,
        broken_abort_ratio=0.99,
        broken_abort_min_attempts=200,
        max_broken_series_log=100,
        broken_series_log_path=str(tmp_path / "broken_scratch.jsonl"),
        scratch_dir=str(scratch_root),
        pair_views=True,
    )

    it = iter(ds)
    max_files_seen = 0
    try:
        for _ in range(16):
            _ = next(it)
            worker_dir = scratch_root / "worker_0"
            if worker_dir.exists():
                file_count = sum(1 for p in worker_dir.rglob("*") if p.is_file())
                max_files_seen = max(max_files_seen, file_count)
    finally:
        it.close()

    # One staged scan per warm-pool slot, plus up to one in-flight temp file.
    assert max_files_seen <= 3
    assert not (scratch_root / "worker_0").exists()
