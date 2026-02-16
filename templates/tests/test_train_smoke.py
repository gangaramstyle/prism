from __future__ import annotations

import csv
import gzip
from pathlib import Path

import nibabel as nib
import numpy as np

from prism_ssl.config.schema import (
    CheckpointConfig,
    DataConfig,
    LossConfig,
    QuotaConfig,
    RunConfig,
    RunMetadataConfig,
    RuntimeConfig,
    TrainConfig,
    WandbConfig,
)
from prism_ssl.train.checkpoint import select_resume_checkpoint
from prism_ssl.train.train_loop import run_training


def _make_nifti(path: Path, shape: tuple[int, int, int] = (48, 48, 12)) -> None:
    arr = np.random.randn(*shape).astype(np.float32)
    img = nib.Nifti1Image(arr, affine=np.eye(4, dtype=np.float32))
    nib.save(img, str(path))


def _write_catalog(catalog_path: Path, series_paths: list[Path]) -> None:
    fieldnames = [
        "pmbb_id",
        "modality",
        "body_part",
        "series_description",
        "manufacturer",
        "exam_type",
        "series_size_mb",
        "study_total_size_mb",
        "n_nifti_in_study",
        "series_path",
    ]
    with gzip.open(catalog_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, sp in enumerate(series_paths):
            writer.writerow(
                {
                    "pmbb_id": f"PMBB{i:08d}",
                    "modality": "CT" if i % 2 == 0 else "MR",
                    "body_part": "CHEST",
                    "series_description": f"SERIES_{i}",
                    "manufacturer": "TEST",
                    "exam_type": "TEST",
                    "series_size_mb": "1.0",
                    "study_total_size_mb": "1.0",
                    "n_nifti_in_study": "1",
                    "series_path": str(sp),
                }
            )


def test_train_smoke_cpu_120_steps(tmp_path: Path):
    series_paths = []
    for i in range(4):
        series_dir = tmp_path / f"series_{i}"
        series_dir.mkdir(parents=True, exist_ok=True)
        _make_nifti(series_dir / f"scan_{i}.nii.gz")
        series_paths.append(series_dir)

    catalog_path = tmp_path / "catalog.csv.gz"
    _write_catalog(catalog_path, series_paths)

    cfg = RunConfig(
        run=RunMetadataConfig(name="smoke", seed=123),
        wandb=WandbConfig(mode="disabled", project="", entity="", tags=[]),
        data=DataConfig(
            catalog_path=str(catalog_path),
            n_scans=4,
            n_patches=8,
            patch_mm=16.0,
            workers=0,
            warm_pool_size=2,
            visits_per_scan=2,
            max_prefetch_replacements=1,
            strict_background_errors=False,
            broken_abort_ratio=0.10,
            broken_abort_min_attempts=200,
            max_broken_series_log=100,
        ),
        train=TrainConfig(
            batch_size=2,
            max_steps=120,
            log_every=20,
            precision="fp32",
            lr=1e-4,
            weight_decay=1e-2,
            seed=123,
            device="cpu",
            allow_failures=False,
        ),
        loss=LossConfig(
            w_distance=1.0,
            w_rotation=1.0,
            w_window=1.0,
            w_supcon_target=0.2,
            supcon_temperature=0.1,
            supcon_warmup_steps=10,
            supcon_ramp_steps=20,
            normalize_targets=True,
        ),
        checkpoint=CheckpointConfig(
            artifact_every_steps=0,
            max_local_checkpoints=1,
            local_ckpt_dir=str(tmp_path / "checkpoints"),
            no_resume=False,
        ),
        quota=QuotaConfig(home_soft_limit_gb=10_000.0, home_hard_limit_gb=20_000.0),
        runtime=RuntimeConfig(tmp_run_dir=str(tmp_path / "tmp_run"), summary_output=str(tmp_path / "summary.json")),
    )

    result = run_training(cfg)

    assert result["status"] == "ok"
    assert result["final_step"] == 120
    assert result["throughput_effective_patches_per_sec"] > 0.0
    assert result["attempted_series"] > 0
    assert Path(result["local_ckpt_path"]).exists()


def test_resume_selection_prefers_local(tmp_path: Path):
    local_ckpt = tmp_path / "last.ckpt"
    artifact_ckpt = tmp_path / "artifact.ckpt"
    local_ckpt.touch()
    artifact_ckpt.touch()

    chosen = select_resume_checkpoint(local_ckpt, artifact_ckpt)
    assert chosen == local_ckpt
