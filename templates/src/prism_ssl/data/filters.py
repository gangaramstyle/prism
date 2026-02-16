"""Catalog filtering helpers."""

from __future__ import annotations

import polars as pl


def filter_modalities(df: pl.DataFrame, modalities: tuple[str, ...]) -> pl.DataFrame:
    mods = [m.upper() for m in modalities]
    return df.filter(pl.col("modality").cast(pl.Utf8).str.to_uppercase().is_in(mods))


def filter_nonempty_series_path(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("series_path").is_not_null() & (pl.col("series_path") != ""))
