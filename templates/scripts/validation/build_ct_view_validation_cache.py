#!/usr/bin/env python3
"""Build a CT semantic view validation cache under a size budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


_bootstrap_imports()

from prism_ssl.config import load_run_config
from prism_ssl.validation import build_ct_view_validation_cache, load_ct_view_validation_cache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, help="Training config YAML used to inherit patch sampling settings.")
    parser.add_argument("--catalog-path", required=True, help="Catalog CSV/CSV.GZ/PARQUET used to source CT scans.")
    parser.add_argument(
        "--output-dir",
        default="~/prism_ssl_validation/ct_view_phase1",
        help="Destination directory for the cached validation tensors and metadata.",
    )
    parser.add_argument("--target-scans", type=int, default=128, help="Number of CT scans to cache.")
    parser.add_argument("--views-per-scan", type=int, default=16, help="Number of semantic views to cache per scan.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for scan and view selection.")
    parser.add_argument("--max-cache-gb", type=float, default=5.0, help="Maximum cache size budget in GiB.")
    parser.add_argument("--shard-size", type=int, default=256, help="Number of views per tensor shard.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache directory.")
    return parser.parse_args()


def _progress_line(event: dict[str, object]) -> str:
    stage = str(event.get("stage", "build"))
    status = str(event.get("status", ""))
    return f"[{stage}:{status}] {json.dumps(event, sort_keys=True)}"


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.overwrite):
        print(summary_path.read_text(encoding="utf-8"))
        return 0

    config = load_run_config(str(args.config_path))

    def _progress(event: dict[str, object]) -> None:
        print(_progress_line(event), flush=True)

    summary = build_ct_view_validation_cache(
        args.catalog_path,
        config,
        output_dir,
        target_scans=int(args.target_scans),
        views_per_scan=int(args.views_per_scan),
        seed=int(args.seed),
        max_cache_gb=float(args.max_cache_gb),
        shard_size=int(args.shard_size),
        overwrite=bool(args.overwrite),
        progress=_progress,
    )
    loaded = load_ct_view_validation_cache(output_dir)
    print(
        json.dumps(
            {
                "status": "built",
                **summary,
                "loaded_scans": int(loaded["scans_df"].height),
                "loaded_views": int(loaded["views_df"].height),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
