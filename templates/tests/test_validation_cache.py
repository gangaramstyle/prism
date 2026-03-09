from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import torch

from prism_ssl.validation import (
    build_eval_batch_from_ct_validation_cache,
    estimate_ct_validation_sample_bytes,
    infer_contrast_bucket,
    infer_series_family,
    load_ct_validation_cache,
    max_ct_validation_studies_for_budget,
)


def test_contrast_and_series_family_inference() -> None:
    assert infer_contrast_bucket({"series_description": "CT ABDOMEN W/ CONTRAST"}) == "contrast"
    assert infer_contrast_bucket({"series_description": "CT CHEST WITHOUT CONTRAST"}) == "non_contrast"
    assert infer_contrast_bucket({"series_description": "CT LOCALIZER"}) == "unknown"

    assert infer_series_family({"series_description": "  ct chest venous  "}) == "CT CHEST VENOUS"
    assert infer_series_family({"series_path": "/tmp/patient/study/series_xyz"}) == "SERIES_XYZ"


def test_budget_estimate_and_capacity_scale_with_patch_count() -> None:
    small = estimate_ct_validation_sample_bytes(128)
    large = estimate_ct_validation_sample_bytes(384)
    assert small > 0
    assert large > small

    capacity = max_ct_validation_studies_for_budget(384, max_cache_gb=2.0)
    assert capacity > 0
    assert capacity < 10_000


def test_validation_cache_roundtrip_and_slice(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ct_cache"
    cache_dir.mkdir()

    summary = {
        "cache_version": 1,
        "cache_type": "ct_study4_validation",
        "n_studies": 2,
        "n_views": 8,
        "n_shards": 1,
        "n_patches": 3,
    }
    (cache_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    samples_df = pl.DataFrame(
        [
            {
                "sample_index": 0,
                "study_id": "study_a",
                "series_id_x": "series_ax",
                "series_id_y": "series_ay",
                "cross_valid": True,
                "cross_mode": "paired_world",
                "shard_index": 0,
            },
            {
                "sample_index": 1,
                "study_id": "study_b",
                "series_id_x": "series_bx",
                "series_id_y": "series_by",
                "cross_valid": False,
                "cross_mode": "unpaired",
                "shard_index": 0,
            },
        ]
    )
    samples_df.write_parquet(cache_dir / "samples.parquet")

    view_rows = []
    for sample_index in range(2):
        for view_index, view_name in enumerate(("a", "ap", "b", "bp")):
            view_rows.append(
                {
                    "sample_index": sample_index,
                    "view_index": view_index,
                    "view_name": view_name,
                    "study_id": f"study_{sample_index}",
                    "series_id": f"series_{sample_index}_{view_name}",
                    "series_path": f"/tmp/series_{sample_index}_{view_name}",
                    "series_description": f"Series {sample_index} {view_name}",
                    "series_label_text": f"Series {sample_index}",
                    "anatomy_label": 7 if sample_index == 0 else 0,
                    "cross_valid": bool(sample_index == 0),
                    "cross_mode": "paired_world" if sample_index == 0 else "unpaired",
                    "modality": "CT",
                    "contrast_bucket": "contrast" if sample_index == 0 else "unknown",
                    "series_family": f"SERIES {sample_index}",
                    "anatomy_label_source": "totalseg" if sample_index == 0 else "none",
                    "has_anatomy_label": bool(sample_index == 0),
                    "native_acquisition_plane": "axial",
                    "shard_index": 0,
                }
            )
    pl.DataFrame(view_rows).write_parquet(cache_dir / "views.parquet")

    torch.save(
        {
            "sample_index": torch.tensor([0, 1], dtype=torch.int64),
            "patches_views": torch.arange(2 * 4 * 3 * 16 * 16, dtype=torch.float16).reshape(2, 4, 3, 16, 16),
            "positions_views": torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3),
            "cross_valid": torch.tensor([True, False], dtype=torch.bool),
        },
        cache_dir / "shard_000.pt",
    )

    events: list[dict[str, object]] = []
    cache = load_ct_validation_cache(cache_dir, progress=events.append)
    batch = build_eval_batch_from_ct_validation_cache(cache, start=1, stop=2)

    assert cache["summary"]["cache_dir"] == str(cache_dir.resolve())
    assert cache["patches_views"].shape == (2, 4, 3, 16, 16)
    assert cache["positions_views"].shape == (2, 4, 3, 3)
    assert cache["cross_valid"].tolist() == [True, False]
    assert batch["patches_views"].shape == (1, 4, 3, 16, 16)
    assert batch["positions_views"].shape == (1, 4, 3, 3)
    assert batch["cross_valid"].tolist() == [False]
    assert [row["sample_index"] for row in batch["sample_rows"]] == [1]
    assert [row["view_name"] for row in batch["view_rows"]] == ["a", "ap", "b", "bp"]
    assert [event["stage"] for event in events] == ["metadata", "metadata", "shards", "finalize"]
