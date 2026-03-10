"""CT-only cached validation helpers for fast checkpoint probing."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import polars as pl
import torch

from prism_ssl.config import RunConfig
from prism_ssl.eval.checkpoint_probe import build_eval_batch, iter_study4_examples

CONTRAST_POSITIVE_KEYWORDS: tuple[str, ...] = (
    "WITH CONTRAST",
    "W/ CONTRAST",
    "W CONTRAST",
    "POST CONTRAST",
    "POST-CONTRAST",
    "CONTRAST",
    "ENHANC",
    "CTA",
    "VENOUS",
    "ARTERIAL",
    "PORTAL",
    "DELAYED",
)

CONTRAST_NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "WITHOUT CONTRAST",
    "W/O CONTRAST",
    "WO CONTRAST",
    "NO CONTRAST",
    "NON CONTRAST",
    "NON-CONTRAST",
    "PRE CONTRAST",
    "PRE-CONTRAST",
    "PRECONTRAST",
)

_CACHE_VERSION = 1
_SAMPLES_PARQUET = "samples.parquet"
_VIEWS_PARQUET = "views.parquet"
_SUMMARY_JSON = "summary.json"
_SHARD_GLOB = "shard_*.pt"


def _normalize_text_parts(row: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("series_description", "")).strip(),
        str(row.get("exam_type", "")).strip(),
        str(row.get("body_part", "")).strip(),
    ]
    return " ".join(part for part in parts if part).upper()


def infer_contrast_bucket(row: Mapping[str, Any]) -> str:
    text = _normalize_text_parts(row)
    if any(keyword in text for keyword in CONTRAST_NEGATIVE_KEYWORDS):
        return "non_contrast"
    if any(keyword in text for keyword in CONTRAST_POSITIVE_KEYWORDS):
        return "contrast"
    return "unknown"


def infer_series_family(row: Mapping[str, Any]) -> str:
    desc = " ".join(str(row.get("series_description", "")).strip().upper().split())
    if desc:
        return desc
    return Path(str(row.get("series_path", ""))).name.upper() or "UNKNOWN"


def estimate_ct_validation_sample_bytes(
    n_patches: int,
    *,
    patch_dtype: torch.dtype = torch.float16,
    position_dtype: torch.dtype = torch.float32,
) -> int:
    """Estimate bytes required for one cached study4 sample."""
    patches_bytes = 4 * int(n_patches) * 16 * 16 * torch.tensor([], dtype=patch_dtype).element_size()
    positions_bytes = 4 * int(n_patches) * 3 * torch.tensor([], dtype=position_dtype).element_size()
    cross_valid_bytes = 1
    metadata_slack_bytes = 16_384
    return int(patches_bytes + positions_bytes + cross_valid_bytes + metadata_slack_bytes)


def max_ct_validation_studies_for_budget(
    n_patches: int,
    *,
    max_cache_gb: float,
    patch_dtype: torch.dtype = torch.float16,
    position_dtype: torch.dtype = torch.float32,
) -> int:
    budget_bytes = int(max(float(max_cache_gb), 0.0) * (1024**3))
    if budget_bytes <= 0:
        return 0
    per_sample_bytes = estimate_ct_validation_sample_bytes(
        n_patches,
        patch_dtype=patch_dtype,
        position_dtype=position_dtype,
    )
    return max(1, budget_bytes // max(per_sample_bytes, 1))


def _cache_summary_path(output_dir: Path) -> Path:
    return output_dir / _SUMMARY_JSON


def _cache_samples_path(output_dir: Path) -> Path:
    return output_dir / _SAMPLES_PARQUET


def _cache_views_path(output_dir: Path) -> Path:
    return output_dir / _VIEWS_PARQUET


def _cache_shard_path(output_dir: Path, shard_index: int) -> Path:
    return output_dir / f"shard_{int(shard_index):03d}.pt"


def _enrich_sample_rows(rows: Sequence[Mapping[str, Any]], *, shard_index: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched.append(
            {
                **dict(row),
                "shard_index": int(shard_index),
            }
        )
    return enriched


def _enrich_view_rows(rows: Sequence[Mapping[str, Any]], *, shard_index: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        base = dict(row)
        enriched.append(
            {
                **base,
                "shard_index": int(shard_index),
                "modality": "CT",
                "contrast_bucket": infer_contrast_bucket(base),
                "series_family": infer_series_family(base),
                "anatomy_label_source": "totalseg" if int(base.get("anatomy_label", 0)) > 0 else "none",
                "has_anatomy_label": bool(int(base.get("anatomy_label", 0)) > 0),
                "native_acquisition_plane": str(base.get("native_acquisition_plane", "unknown")),
            }
        )
    return enriched


def build_ct_validation_cache(
    catalog: str | Path | pl.DataFrame,
    config: RunConfig,
    output_dir: str | Path,
    *,
    target_studies: int,
    seed: int,
    max_cache_gb: float = 2.0,
    shard_size: int = 128,
    overwrite: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Materialize a CT-only study4 validation cache under a size budget."""
    if str(config.data.sample_unit).strip().lower() != "study4":
        raise ValueError(f"CT validation cache requires sample_unit='study4', got '{config.data.sample_unit}'")

    output_root = Path(output_dir).expanduser().resolve()
    summary_path = _cache_summary_path(output_root)
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Validation cache already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale_path in (
            summary_path,
            _cache_samples_path(output_root),
            _cache_views_path(output_root),
            *sorted(output_root.glob(_SHARD_GLOB)),
        ):
            if stale_path.exists():
                stale_path.unlink()

    max_budget_studies = max_ct_validation_studies_for_budget(
        int(config.data.n_patches),
        max_cache_gb=float(max_cache_gb),
    )
    effective_target = min(max(int(target_studies), 0), max_budget_studies)
    if effective_target <= 0:
        raise ValueError(
            f"Requested target_studies={target_studies} does not fit within max_cache_gb={max_cache_gb:.2f}"
        )

    if progress is not None:
        progress(
            {
                "stage": "sampling",
                "target_studies_requested": int(target_studies),
                "target_studies_budgeted": int(effective_target),
                "max_budget_studies": int(max_budget_studies),
                "status": "start",
            }
        )

    samples_rows: list[dict[str, Any]] = []
    views_rows: list[dict[str, Any]] = []
    total_bytes_written = 0
    shard_paths: list[str] = []
    build_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    written_studies = 0
    current_chunk: list[dict[str, Any]] = []
    shard_width = max(int(shard_size), 1)

    def _flush_chunk(chunk: Sequence[Mapping[str, Any]], *, shard_index: int, sample_offset: int) -> None:
        nonlocal total_bytes_written, written_studies
        batch = build_eval_batch(chunk, sample_offset=sample_offset)
        shard_payload = {
            "sample_index": torch.tensor([int(row["sample_index"]) for row in batch["sample_rows"]], dtype=torch.int64),
            "patches_views": batch["patches_views"].to(dtype=torch.float16, device="cpu"),
            "positions_views": batch["positions_views"].to(dtype=torch.float32, device="cpu"),
            "cross_valid": batch["cross_valid"].to(dtype=torch.bool, device="cpu"),
        }
        shard_path = _cache_shard_path(output_root, shard_index)
        torch.save(shard_payload, shard_path)
        total_bytes_written += int(shard_path.stat().st_size)
        shard_paths.append(str(shard_path))

        samples_rows.extend(_enrich_sample_rows(batch["sample_rows"], shard_index=shard_index))
        views_rows.extend(_enrich_view_rows(batch["view_rows"], shard_index=shard_index))
        written_studies += int(len(chunk))

        if progress is not None:
            progress(
                {
                    "stage": "writing",
                    "status": "shard_complete",
                    "shard_index": int(shard_index),
                    "studies_written": int(written_studies),
                    "total_studies": int(effective_target),
                    "bytes_written": int(total_bytes_written),
                }
            )

    for example in iter_study4_examples(
        catalog,
        config,
        n_studies=effective_target,
        seed=int(seed),
        modality_filter=("CT",),
        progress=(lambda payload: progress({"stage": "sampling", **payload}) if progress is not None else None),
    ):
        current_chunk.append(example)
        if len(current_chunk) >= shard_width:
            _flush_chunk(current_chunk, shard_index=len(shard_paths), sample_offset=written_studies)
            current_chunk = []

    if current_chunk:
        _flush_chunk(current_chunk, shard_index=len(shard_paths), sample_offset=written_studies)

    if written_studies == 0:
        raise RuntimeError("No CT study4 examples could be sampled for validation cache")

    samples_df = pl.DataFrame(samples_rows).sort("sample_index")
    views_df = pl.DataFrame(views_rows).sort(["sample_index", "view_index"])
    samples_df.write_parquet(_cache_samples_path(output_root))
    views_df.write_parquet(_cache_views_path(output_root))

    estimated_bytes_per_sample = estimate_ct_validation_sample_bytes(int(config.data.n_patches))
    summary = {
        "cache_version": _CACHE_VERSION,
        "cache_type": "ct_study4_validation",
        "created_at": build_started_at,
        "output_dir": str(output_root),
        "catalog_path": str(catalog) if isinstance(catalog, (str, Path)) else "<dataframe>",
        "target_studies_requested": int(target_studies),
        "target_studies_budgeted": int(effective_target),
        "n_studies": int(written_studies),
        "n_views": int(len(views_rows)),
        "n_shards": int(len(shard_paths)),
        "shard_size": int(max(int(shard_size), 1)),
        "max_cache_gb": float(max_cache_gb),
        "estimated_bytes_per_sample": int(estimated_bytes_per_sample),
        "estimated_total_bytes": int(estimated_bytes_per_sample * written_studies),
        "actual_tensor_bytes": int(total_bytes_written),
        "n_patches": int(config.data.n_patches),
        "patch_mm": float(config.data.patch_mm),
        "sample_unit": str(config.data.sample_unit),
        "modality_filter": ["CT"],
        "totalseg_labeled_view_count": int(sum(int(bool(row["has_anatomy_label"])) for row in views_rows)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_ct_validation_cache(
    cache_dir: str | Path,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Load a CT validation cache into CPU memory for fast notebook reuse."""
    cache_root = Path(cache_dir).expanduser().resolve()
    summary_path = _cache_summary_path(cache_root)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Validation cache summary not found: {summary_path}")

    shard_paths = sorted(cache_root.glob(_SHARD_GLOB))
    if not shard_paths:
        raise FileNotFoundError(f"No validation cache shards found under: {cache_root}")

    if progress is not None:
        progress(
            {
                "stage": "metadata",
                "status": "start",
                "cache_dir": str(cache_root),
                "n_shards": int(len(shard_paths)),
            }
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples_df = pl.read_parquet(_cache_samples_path(cache_root)).sort("sample_index")
    views_df = pl.read_parquet(_cache_views_path(cache_root)).sort(["sample_index", "view_index"])
    summary["cache_dir"] = str(cache_root)
    summary["n_shards_loaded"] = int(len(shard_paths))
    if progress is not None:
        progress(
            {
                "stage": "metadata",
                "status": "complete",
                "cache_dir": str(cache_root),
                "n_shards": int(len(shard_paths)),
                "n_studies": int(samples_df.height),
            }
        )

    shard_payloads = []
    for shard_index, shard_path in enumerate(shard_paths, start=1):
        shard_payloads.append(torch.load(shard_path, map_location="cpu"))
        if progress is not None:
            progress(
                {
                    "stage": "shards",
                    "status": "loaded",
                    "cache_dir": str(cache_root),
                    "n_shards": int(len(shard_paths)),
                    "loaded_shards": int(shard_index),
                }
            )

    patches_views = torch.cat([payload["patches_views"] for payload in shard_payloads], dim=0)
    positions_views = torch.cat([payload["positions_views"] for payload in shard_payloads], dim=0)
    cross_valid = torch.cat([payload["cross_valid"] for payload in shard_payloads], dim=0)
    if progress is not None:
        progress(
            {
                "stage": "finalize",
                "status": "complete",
                "cache_dir": str(cache_root),
                "n_shards": int(len(shard_paths)),
                "n_studies": int(patches_views.shape[0]),
                "n_views": int(patches_views.shape[0] * 4),
            }
        )

    return {
        "summary": summary,
        "sample_df": samples_df,
        "view_df": views_df,
        "sample_rows": samples_df.to_dicts(),
        "view_rows": views_df.to_dicts(),
        "patches_views": patches_views,
        "positions_views": positions_views,
        "cross_valid": cross_valid,
    }


def build_eval_batch_from_ct_validation_cache(
    cache: Mapping[str, Any],
    *,
    start: int,
    stop: int,
) -> dict[str, Any]:
    """Slice a loaded CT validation cache into a notebook-ready eval batch."""
    begin = max(int(start), 0)
    end = min(int(stop), int(cache["patches_views"].shape[0]))
    if end <= begin:
        raise ValueError(f"Empty validation cache slice requested: start={start} stop={stop}")

    sample_rows = list(cache["sample_rows"][begin:end])
    view_rows = list(cache["view_rows"][begin * 4 : end * 4])
    return {
        "patches_views": cache["patches_views"][begin:end],
        "positions_views": cache["positions_views"][begin:end],
        "cross_valid": cache["cross_valid"][begin:end],
        "sample_rows": sample_rows,
        "view_rows": view_rows,
    }


__all__ = [
    "infer_contrast_bucket",
    "infer_series_family",
    "estimate_ct_validation_sample_bytes",
    "max_ct_validation_studies_for_budget",
    "build_ct_validation_cache",
    "load_ct_validation_cache",
    "build_eval_batch_from_ct_validation_cache",
]
