#!/bin/bash
#SBATCH --job-name=prism-reportrefs
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/prism-ssl/templates}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Run scripts/setup_betty_uv.sh first." >&2
  exit 1
fi

mkdir -p logs results/report_refs

CATALOG_PATH="${CATALOG_PATH:-results/manifests/pmbb_catalog_near_iso.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-results/report_refs/pmbb_report_refs.parquet}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_PATH}.summary.json}"
MODALITIES="${MODALITIES:-CT,MR}"
MAX_ROWS="${MAX_ROWS:-0}"
MAX_REPORTS_PER_STUDY="${MAX_REPORTS_PER_STUDY:-0}"
MATCHED_SERIES_SCOPE="${MATCHED_SERIES_SCOPE:-catalog}"
INCLUDE_UNMAPPED="${INCLUDE_UNMAPPED:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"
OVERWRITE="${OVERWRITE:-0}"

if [ -f "$OUTPUT_PATH" ] && [ "$OVERWRITE" != "1" ]; then
  echo "[skip] Output exists: $OUTPUT_PATH (set OVERWRITE=1 to rebuild)"
  exit 0
fi

UNMAPPED_FLAG="--include-unmapped"
if [ "$INCLUDE_UNMAPPED" = "0" ]; then
  UNMAPPED_FLAG="--no-include-unmapped"
fi

uv run python scripts/build_report_ref_manifest.py \
  --catalog-path "$CATALOG_PATH" \
  --output-path "$OUTPUT_PATH" \
  --summary-path "$SUMMARY_PATH" \
  --modalities "$MODALITIES" \
  --max-rows "$MAX_ROWS" \
  --max-reports-per-study "$MAX_REPORTS_PER_STUDY" \
  --matched-series-scope "$MATCHED_SERIES_SCOPE" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --progress-every "$PROGRESS_EVERY" \
  "$UNMAPPED_FLAG"

echo "[done] report-reference manifest: $OUTPUT_PATH"
echo "[done] summary: $SUMMARY_PATH"
