"""Catalog loading and deterministic candidate sampling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from prism_ssl.config.schema import ScanRecord
from prism_ssl.data.filters import filter_modalities, filter_nonempty_series_path
from prism_ssl.utils.hashing import stable_int_hash


def load_catalog(path: str) -> pl.DataFrame:
    """Load catalog from csv/csv.gz/parquet."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pl.read_parquet(p)
    return pl.read_csv(p)


def build_scan_id(row: dict[str, Any]) -> str:
    """Stable scan identifier derived from PMBB fields."""
    pmbb_id = str(row.get("pmbb_id", "unknown"))
    series_path = str(row.get("series_path", ""))
    series_description = str(row.get("series_description", ""))
    raw = f"{pmbb_id}|{series_path}|{series_description}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{pmbb_id}_{digest}"


def _series_id_from_row(row: dict[str, Any]) -> str:
    series_path = str(row.get("series_path", ""))
    return f"series_{stable_int_hash(series_path) & 0xFFFFFFFF:08x}"


def sample_scan_candidates(
    df: pl.DataFrame,
    n_scans: int,
    seed: int,
    modality_filter: tuple[str, ...] = ("CT", "MR"),
) -> list[ScanRecord]:
    """Deterministically select candidate scan records (without validation prepass)."""
    filtered = filter_nonempty_series_path(filter_modalities(df, modality_filter))
    if len(filtered) == 0:
        return []

    filtered = filtered.with_columns(
        pl.concat_str(
            [
                pl.col("pmbb_id").cast(pl.Utf8),
                pl.col("series_path").cast(pl.Utf8),
                pl.col("series_description").cast(pl.Utf8).fill_null(""),
            ],
            separator="|",
        ).alias("_sample_key")
    )

    # Stable deterministic ordering controlled by seed.
    ordered = filtered.with_columns(pl.col("_sample_key").hash(seed=seed).alias("_h")).sort("_h")
    if n_scans > 0:
        ordered = ordered.head(min(n_scans, len(ordered)))

    records: list[ScanRecord] = []
    for row in ordered.to_dicts():
        scan_id = build_scan_id(row)
        records.append(
            ScanRecord(
                scan_id=scan_id,
                series_id=_series_id_from_row(row),
                modality=str(row.get("modality", "CT")).upper(),
                series_path=str(row["series_path"]),
                nifti_path="",  # resolved lazily by dataset workers
            )
        )
    return records
