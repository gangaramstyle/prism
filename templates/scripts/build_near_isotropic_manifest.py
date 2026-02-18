#!/usr/bin/env python3
"""Build a near-isotropic manifest from a PMBB catalog.

This script inspects NIfTI header spacing (no full volume load) and writes a CSV
containing rows whose spacing ratio passes a near-isotropic threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism_ssl.data.catalog import load_catalog
from prism_ssl.data.filters import filter_likely_non_time_series, filter_modalities, filter_nonempty_series_path
from prism_ssl.data.preflight import infer_scan_geometry, resolve_nifti_path


def _parse_modalities(value: str) -> tuple[str, ...]:
    mods = tuple(m.strip().upper() for m in value.split(",") if m.strip())
    if not mods:
        raise ValueError("modalities must include at least one value")
    return mods


def _coerce_csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return value


def _inspect_spacing(series_path: str) -> tuple[str, np.ndarray, np.ndarray, int]:
    nifti_path = resolve_nifti_path(series_path)
    img = nib.load(nifti_path)
    shape_full = tuple(int(v) for v in img.shape)
    shape = np.asarray(shape_full[:3], dtype=np.int64)
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    if shape.size != 3:
        raise ValueError(f"Expected 3D shape, got {shape_full}")
    if spacing.size != 3 or np.any(spacing <= 0):
        raise ValueError(f"Invalid spacing: {spacing}")
    timepoints = int(shape_full[3]) if len(shape_full) > 3 else 1
    return nifti_path, shape, spacing, timepoints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build near-isotropic scan manifest")
    p.add_argument("--catalog-path", required=True, type=str)
    p.add_argument("--output-path", required=True, type=str)
    p.add_argument("--modalities", default="CT,MR", type=str)
    p.add_argument("--max-spacing-ratio", default=1.2, type=float)
    p.add_argument("--max-spacing-mm", default=0.0, type=float)
    p.add_argument("--max-rows", default=0, type=int)
    p.add_argument("--summary-path", default="", type=str)
    p.add_argument(
        "--exclude-time-series",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude likely time-series sequences (keyword and 4D checks).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    modalities = _parse_modalities(args.modalities)
    max_spacing_ratio = float(args.max_spacing_ratio)
    max_spacing_mm = float(args.max_spacing_mm)
    exclude_time_series = bool(args.exclude_time_series)

    if max_spacing_ratio <= 1.0:
        raise SystemExit(f"--max-spacing-ratio must be > 1.0, got {max_spacing_ratio}")
    if max_spacing_mm < 0.0:
        raise SystemExit(f"--max-spacing-mm must be >= 0.0, got {max_spacing_mm}")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    df = load_catalog(args.catalog_path)
    df = filter_nonempty_series_path(filter_modalities(df, modalities))
    if args.max_rows > 0:
        df = df.head(int(args.max_rows))
    before_ts_filter = len(df)
    if exclude_time_series:
        df = filter_likely_non_time_series(df)
    ts_keyword_filtered = before_ts_filter - len(df)

    base_columns = list(df.columns)
    extra_columns = [
        "nifti_path",
        "shape_x",
        "shape_y",
        "shape_z",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "spacing_ratio",
        "native_plane",
        "thin_axis",
        "thin_axis_name",
        "geometry_inference_reason",
    ]
    fieldnames = base_columns + [c for c in extra_columns if c not in base_columns]

    counters = {
        "total_rows_considered": 0,
        "rows_written": 0,
        "missing_or_bad_nifti": 0,
        "filtered_by_time_series_keywords": int(ts_keyword_filtered),
        "filtered_by_time_series_4d": 0,
        "filtered_by_spacing_ratio": 0,
        "filtered_by_max_spacing_mm": 0,
    }

    with tmp_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in df.iter_rows(named=True):
            counters["total_rows_considered"] += 1
            series_path = str(row.get("series_path", ""))
            try:
                nifti_path, shape, spacing, timepoints = _inspect_spacing(series_path)
            except Exception:
                counters["missing_or_bad_nifti"] += 1
                continue

            if exclude_time_series and timepoints > 1:
                counters["filtered_by_time_series_4d"] += 1
                continue

            spacing_ratio = float(np.max(spacing) / max(np.min(spacing), 1e-9))
            if spacing_ratio > max_spacing_ratio:
                counters["filtered_by_spacing_ratio"] += 1
                continue
            if max_spacing_mm > 0.0 and float(np.max(spacing)) > max_spacing_mm:
                counters["filtered_by_max_spacing_mm"] += 1
                continue

            geom = infer_scan_geometry(tuple(int(v) for v in shape.tolist()), tuple(float(v) for v in spacing.tolist()))
            out_row = dict(row)
            out_row.update(
                {
                    "nifti_path": nifti_path,
                    "shape_x": int(shape[0]),
                    "shape_y": int(shape[1]),
                    "shape_z": int(shape[2]),
                    "spacing_x_mm": float(spacing[0]),
                    "spacing_y_mm": float(spacing[1]),
                    "spacing_z_mm": float(spacing[2]),
                    "spacing_ratio": spacing_ratio,
                    "native_plane": geom.acquisition_plane,
                    "thin_axis": int(geom.thin_axis),
                    "thin_axis_name": geom.thin_axis_name,
                    "geometry_inference_reason": geom.inference_reason,
                }
            )
            writer.writerow({k: _coerce_csv_value(out_row.get(k, "")) for k in fieldnames})
            counters["rows_written"] += 1

    tmp_output_path.replace(output_path)

    summary_payload = {
        **counters,
        "catalog_path": str(Path(args.catalog_path).expanduser()),
        "output_path": str(output_path),
        "modalities": list(modalities),
        "max_spacing_ratio": max_spacing_ratio,
        "max_spacing_mm": max_spacing_mm,
        "exclude_time_series": exclude_time_series,
        "max_rows": int(args.max_rows),
        "retained_fraction": (
            float(counters["rows_written"]) / float(counters["total_rows_considered"])
            if counters["total_rows_considered"] > 0
            else 0.0
        ),
    }

    summary_path = Path(args.summary_path).expanduser().resolve() if args.summary_path else output_path.with_suffix(
        output_path.suffix + ".summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(json.dumps(summary_payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
