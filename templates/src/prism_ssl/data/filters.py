"""Catalog filtering helpers."""

from __future__ import annotations

import polars as pl


def filter_modalities(df: pl.DataFrame, modalities: tuple[str, ...]) -> pl.DataFrame:
    mods = [m.upper() for m in modalities]
    return df.filter(pl.col("modality").cast(pl.Utf8).str.to_uppercase().is_in(mods))


def filter_nonempty_series_path(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("series_path").is_not_null() & (pl.col("series_path") != ""))


TIME_SERIES_KEYWORDS: tuple[str, ...] = (
    "DIFFUSION",
    "DWI",
    "DTI",
    "ADC",
    "CINE",
    "CARDIAC",
    "PERFUSION",
    "DYNAMIC",
    "FMRI",
    "BOLD",
    "ASL",
    "MULTIPHASE",
    "MULTI PHASE",
    "TIME SERIES",
    "TIMESERIES",
    "4D",
    "GATED",
    "TRIGGERED",
)


def likely_time_series_expr(
    *,
    series_description_col: str = "series_description",
    exam_type_col: str = "exam_type",
) -> pl.Expr:
    expr = pl.lit(False)
    if series_description_col:
        desc = pl.col(series_description_col).cast(pl.Utf8).fill_null("").str.to_uppercase()
        for keyword in TIME_SERIES_KEYWORDS:
            expr = expr | desc.str.contains(keyword, literal=True)
    if exam_type_col:
        exam = pl.col(exam_type_col).cast(pl.Utf8).fill_null("").str.to_uppercase()
        for keyword in TIME_SERIES_KEYWORDS:
            expr = expr | exam.str.contains(keyword, literal=True)
    return expr


def filter_likely_non_time_series(
    df: pl.DataFrame,
    *,
    series_description_col: str = "series_description",
    exam_type_col: str = "exam_type",
) -> pl.DataFrame:
    if series_description_col not in df.columns and exam_type_col not in df.columns:
        return df
    desc_col = series_description_col if series_description_col in df.columns else ""
    exam_col = exam_type_col if exam_type_col in df.columns else ""
    return df.filter(~likely_time_series_expr(series_description_col=desc_col, exam_type_col=exam_col))
