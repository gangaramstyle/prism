#!/usr/bin/env python3
"""Merge sharded near-isotropic manifest CSVs into one manifest."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sharded near-isotropic manifests")
    p.add_argument("--shard-dir", required=True, type=str)
    p.add_argument("--output-path", required=True, type=str)
    p.add_argument("--summary-path", default="", type=str)
    p.add_argument("--pattern", default="near_iso_shard_*_of_*.csv", type=str)
    p.add_argument("--require-num-shards", default=0, type=int)
    p.add_argument("--dedupe-on-series-path", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_counter_key(key: str) -> bool:
    return key in {"total_rows_considered", "rows_written", "missing_or_bad_nifti"} or key.startswith("filtered_")


def main() -> int:
    args = parse_args()
    shard_dir = Path(args.shard_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")

    shard_paths = [Path(p).resolve() for p in sorted(glob.glob(str(shard_dir / args.pattern)))]
    if not shard_paths:
        raise SystemExit(f"No shard CSV files found in {shard_dir} matching pattern {args.pattern!r}")
    if args.require_num_shards > 0 and len(shard_paths) != int(args.require_num_shards):
        raise SystemExit(
            f"Expected {int(args.require_num_shards)} shard files, found {len(shard_paths)} in {shard_dir}"
        )

    frames: list[pl.DataFrame] = []
    for shard_path in shard_paths:
        frames.append(pl.read_csv(shard_path))
    merged = pl.concat(frames, how="diagonal_relaxed")
    rows_before = len(merged)

    dedupe_applied = bool(args.dedupe_on_series_path) and ("series_path" in merged.columns)
    if dedupe_applied:
        merged = merged.unique(subset=["series_path"], keep="first")

    sort_cols = [c for c in ("modality", "body_part", "series_description", "series_path") if c in merged.columns]
    if sort_cols:
        merged = merged.sort(sort_cols)

    merged.write_csv(tmp_output)
    tmp_output.replace(output_path)

    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate_counters: dict[str, int] = {}
    per_shard_summaries = []
    for shard_path in shard_paths:
        summary_candidate = shard_path.with_suffix(shard_path.suffix + ".summary.json")
        # Support summaries/<name>.summary.json layout used by array job script.
        alt_summary = shard_path.parent / "summaries" / (shard_path.stem + ".summary.json")
        chosen = summary_candidate if summary_candidate.exists() else alt_summary
        if not chosen.exists():
            continue
        payload = _load_json(chosen)
        per_shard_summaries.append({"path": str(chosen), "payload": payload})
        for k, v in payload.items():
            if _is_counter_key(k) and isinstance(v, int):
                aggregate_counters[k] = aggregate_counters.get(k, 0) + v

    summary_payload = {
        "shard_dir": str(shard_dir),
        "output_path": str(output_path),
        "pattern": args.pattern,
        "shard_file_count": len(shard_paths),
        "rows_merged_before_dedupe": int(rows_before),
        "rows_merged_after_dedupe": int(len(merged)),
        "dedupe_on_series_path": bool(args.dedupe_on_series_path),
        "dedupe_applied": bool(dedupe_applied),
        "aggregate_counters_from_shard_summaries": aggregate_counters,
        "shard_files": [str(p) for p in shard_paths],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
