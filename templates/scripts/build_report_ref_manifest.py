#!/usr/bin/env python3
"""Build weak report-reference labels from PMBB reports.

Example output rows map report snippets like "series 5 image 23" to a matched
series directory and, when possible, a canonical RAS voxel coordinate for the
corresponding DICOM instance/slice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism_ssl.reports import build_report_reference_manifest


def _parse_modalities(value: str) -> tuple[str, ...]:
    out = tuple(v.strip().upper() for v in value.split(",") if v.strip())
    if not out:
        raise argparse.ArgumentTypeError("must provide at least one modality")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PMBB report-reference weak-label manifest")
    p.add_argument("--catalog-path", required=True, type=str)
    p.add_argument("--output-path", required=True, type=str)
    p.add_argument("--summary-path", default="", type=str)
    p.add_argument("--modalities", default="CT,MR", type=str)
    p.add_argument("--max-rows", default=0, type=int, help="Limit catalog rows before grouping by study.")
    p.add_argument("--max-reports-per-study", default=0, type=int, help="0 means all report txt files.")
    p.add_argument(
        "--matched-series-scope",
        default="catalog",
        choices=("catalog", "study"),
        help="catalog restricts matches to series present in the input manifest; study scans all series dirs.",
    )
    p.add_argument(
        "--include-unmapped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep rows for references that could not be mapped to a series/slice.",
    )
    p.add_argument("--num-shards", default=1, type=int, help="Total deterministic study shards.")
    p.add_argument("--shard-index", default=0, type=int, help="0-based shard index.")
    p.add_argument("--progress-every", default=250, type=int)
    return p.parse_args()


def _write_df(df, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.suffix.lower() == ".parquet":
        df.write_parquet(tmp_path)
    elif output_path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if output_path.suffix.lower() == ".tsv" else ","
        df.write_csv(tmp_path, separator=sep)
    else:
        raise ValueError(f"Unsupported output suffix {output_path.suffix!r}; use .parquet, .csv, or .tsv")
    tmp_path.replace(output_path)


def main() -> int:
    args = parse_args()
    modalities = _parse_modalities(args.modalities)
    output_path = Path(args.output_path).expanduser().resolve()
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )

    df, summary = build_report_reference_manifest(
        catalog_path=args.catalog_path,
        modalities=modalities,
        max_rows=int(args.max_rows),
        max_reports_per_study=int(args.max_reports_per_study),
        matched_series_scope=str(args.matched_series_scope),
        include_unmapped=bool(args.include_unmapped),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        progress_every=int(args.progress_every),
    )
    summary = {**summary, "output_path": str(output_path), "summary_path": str(summary_path)}

    _write_df(df, output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
