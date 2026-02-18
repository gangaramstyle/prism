#!/usr/bin/env python3
"""Summarize a manifest with key cohort distribution counts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism_ssl.data.catalog import load_catalog
from prism_ssl.data.filters import likely_time_series_expr


CONTRAST_POSITIVE_KEYWORDS: tuple[str, ...] = (
    "WITH CONTRAST",
    "W/ CONTRAST",
    "W CONTRAST",
    "POST CONTRAST",
    "POST-CONTRAST",
    "CONTRAST",
    "ENHANC",
    "GAD",
    "+C",
    "ARTERIAL",
    "VENOUS",
    "PORTAL",
    "DELAYED",
    "CTA",
    "MRA",
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


def _contains_any_expr(text: pl.Expr, keywords: tuple[str, ...]) -> pl.Expr:
    expr = pl.lit(False)
    for keyword in keywords:
        expr = expr | text.str.contains(keyword, literal=True)
    return expr


def _build_text_expr(df: pl.DataFrame) -> pl.Expr:
    parts: list[pl.Expr] = []
    for col in ("series_description", "exam_type", "body_part"):
        if col in df.columns:
            parts.append(pl.col(col).cast(pl.Utf8).fill_null(""))
    if not parts:
        return pl.lit("").cast(pl.Utf8)
    return pl.concat_str(parts, separator=" ").str.to_uppercase()


def _top_counts(df: pl.DataFrame, column: str, top_k: int) -> pl.DataFrame:
    if column not in df.columns:
        return pl.DataFrame({"value": [], "count": []}, schema={"value": pl.Utf8, "count": pl.UInt32})
    return (
        df.select(pl.col(column).cast(pl.Utf8).fill_null("<null>").alias("value"))
        .group_by("value")
        .len()
        .rename({"len": "count"})
        .sort("count", descending=True)
        .head(top_k)
    )


def _print_section(title: str, table: pl.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if len(table) == 0:
        print("(none)")
    else:
        print(table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize manifest category counts")
    p.add_argument("--manifest-path", required=True, type=str)
    p.add_argument("--top-k", default=25, type=int)
    p.add_argument("--output-dir", default="", type=str)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    df = load_catalog(str(manifest_path))

    desc_col = "series_description" if "series_description" in df.columns else ""
    exam_col = "exam_type" if "exam_type" in df.columns else ""
    text_upper = _build_text_expr(df)
    likely_ts = likely_time_series_expr(series_description_col=desc_col, exam_type_col=exam_col)
    contrast_non = _contains_any_expr(text_upper, CONTRAST_NEGATIVE_KEYWORDS)
    contrast_pos = _contains_any_expr(text_upper, CONTRAST_POSITIVE_KEYWORDS)

    df = df.with_columns(
        [
            likely_ts.alias("_likely_time_series"),
            pl.when(contrast_non)
            .then(pl.lit("non_contrast"))
            .when(contrast_pos)
            .then(pl.lit("contrast"))
            .otherwise(pl.lit("unknown"))
            .alias("_contrast_bucket"),
        ]
    )

    n_rows = int(len(df))
    likely_ts_count = int(df.select(pl.col("_likely_time_series").sum()).item())
    likely_ts_fraction = float(likely_ts_count / n_rows) if n_rows > 0 else 0.0

    summary = {
        "manifest_path": str(manifest_path),
        "rows": n_rows,
        "likely_time_series_count": likely_ts_count,
        "likely_time_series_fraction": likely_ts_fraction,
    }

    print(json.dumps(summary, indent=2))

    modality_counts = _top_counts(df, "modality", args.top_k)
    body_part_counts = _top_counts(df, "body_part", args.top_k)
    manufacturer_counts = _top_counts(df, "manufacturer", args.top_k)
    exam_type_counts = _top_counts(df, "exam_type", args.top_k)
    series_desc_counts = _top_counts(df, "series_description", args.top_k)
    contrast_counts = (
        df.select(pl.col("_contrast_bucket").alias("value"))
        .group_by("value")
        .len()
        .rename({"len": "count"})
        .sort("count", descending=True)
    )
    likely_ts_counts = (
        df.select(pl.col("_likely_time_series").cast(pl.Utf8).alias("value"))
        .group_by("value")
        .len()
        .rename({"len": "count"})
        .sort("count", descending=True)
    )

    _print_section("Modality Counts", modality_counts)
    _print_section("Body Part Counts", body_part_counts)
    _print_section("Contrast Bucket Counts", contrast_counts)
    _print_section("Likely Time-Series Flag Counts", likely_ts_counts)
    _print_section("Top Series Descriptions", series_desc_counts)
    _print_section("Top Exam Types", exam_type_counts)
    _print_section("Top Manufacturers", manufacturer_counts)

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        modality_counts.write_csv(out_dir / "counts_modality.csv")
        body_part_counts.write_csv(out_dir / "counts_body_part.csv")
        contrast_counts.write_csv(out_dir / "counts_contrast_bucket.csv")
        likely_ts_counts.write_csv(out_dir / "counts_likely_time_series.csv")
        series_desc_counts.write_csv(out_dir / "top_series_description.csv")
        exam_type_counts.write_csv(out_dir / "top_exam_type.csv")
        manufacturer_counts.write_csv(out_dir / "top_manufacturer.csv")
        print(f"\n[written] {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
