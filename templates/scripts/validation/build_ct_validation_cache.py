#!/usr/bin/env python3
"""Build a CT-only patch-level validation cache for checkpoint probing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


_bootstrap_imports()

from prism_ssl.config import load_run_config_from_flat
from prism_ssl.eval import load_checkpoint_payload
from prism_ssl.validation import build_ct_validation_cache, load_ct_validation_cache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a study4 training checkpoint; only the embedded config is used.",
    )
    parser.add_argument(
        "--catalog-path",
        required=True,
        help="Catalog CSV/CSV.GZ used to source CT studies.",
    )
    parser.add_argument(
        "--output-dir",
        default="~/prism_ssl_validation/ct_phase1",
        help="Destination directory for the cached validation shards and metadata.",
    )
    parser.add_argument(
        "--target-studies",
        type=int,
        default=256,
        help="Number of unique studies to cache.",
    )
    parser.add_argument(
        "--samples-per-study",
        type=int,
        default=1,
        help="Deterministic study4 samples to cache per unique study.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for study and series selection.",
    )
    parser.add_argument(
        "--max-cache-gb",
        type=float,
        default=2.0,
        help="Maximum cache size budget in GiB.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=128,
        help="Number of study4 samples per tensor shard.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing cache directory instead of reusing it.",
    )
    return parser.parse_args()


def _progress_line(event: dict[str, object]) -> str:
    stage = str(event.get("stage", "build"))
    status = str(event.get("status", ""))
    if stage == "sampling":
        accepted = int(event.get("accepted_examples", 0))
        visited = int(event.get("visited_studies", 0))
        total = int(event.get("total_candidates", 0))
        accepted_studies = int(event.get("accepted_studies", 0))
        target_samples = int(event.get("target_examples", event.get("target_samples_requested", 0)))
        target_studies = int(event.get("target_studies", event.get("target_studies_requested", 0)))
        samples_per_study = int(event.get("samples_per_study", 1))
        return (
            f"[sampling:{status}] samples={accepted}/{target_samples} "
            f"studies={accepted_studies}/{target_studies} "
            f"visited={visited}/{total} samples_per_study={samples_per_study}"
        )
    if stage == "writing":
        shard_index = int(event.get("shard_index", 0))
        samples_written = int(event.get("samples_written", 0))
        total_samples = int(event.get("total_samples", 0))
        unique_studies_written = int(event.get("unique_studies_written", 0))
        total_studies = int(event.get("total_studies", 0))
        bytes_written = int(event.get("bytes_written", 0))
        return (
            f"[writing:{status}] shard={shard_index} samples={samples_written}/{total_samples} "
            f"studies={unique_studies_written}/{total_studies} "
            f"bytes={bytes_written}"
        )
    return f"[{stage}:{status}] {json.dumps(event, sort_keys=True)}"


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not bool(args.overwrite):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "reused_existing_cache", **summary}, indent=2))
        return 0

    payload = load_checkpoint_payload(args.checkpoint_path, device="cpu")
    flat_config = payload.get("config")
    if not isinstance(flat_config, dict):
        raise ValueError("Checkpoint is missing the embedded flat config dictionary")
    config = load_run_config_from_flat(flat_config)

    def _progress(event: dict[str, object]) -> None:
        print(_progress_line(event), flush=True)

    summary = build_ct_validation_cache(
        args.catalog_path,
        config,
        output_dir,
        target_studies=int(args.target_studies),
        samples_per_study=int(args.samples_per_study),
        seed=int(args.seed),
        max_cache_gb=float(args.max_cache_gb),
        shard_size=int(args.shard_size),
        overwrite=bool(args.overwrite),
        progress=_progress,
    )
    loaded = load_ct_validation_cache(output_dir)
    print(
        json.dumps(
            {
                "status": "built",
                **summary,
                "loaded_samples": int(loaded["patches_views"].shape[0]),
                "loaded_unique_studies": int(loaded["sample_df"]["study_id"].n_unique()),
                "loaded_views": int(loaded["view_df"].height),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
