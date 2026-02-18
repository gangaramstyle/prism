#!/bin/bash
#SBATCH --job-name=prism-neariso-arr
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/prism-ssl/templates}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Run scripts/setup_betty_uv.sh first." >&2
  exit 1
fi

mkdir -p logs results/manifests

CATALOG_PATH="${CATALOG_PATH:-$HOME/nvreason/data/pmbb_catalog.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/manifests/pmbb_catalog_near_iso_shards}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUTPUT_DIR}/summaries}"
MODALITIES="${MODALITIES:-CT,MR}"
MAX_SPACING_RATIO="${MAX_SPACING_RATIO:-1.2}"
MAX_SPACING_MM="${MAX_SPACING_MM:-0}"
MAX_ROWS="${MAX_ROWS:-0}"
OVERWRITE="${OVERWRITE:-0}"
EXCLUDE_TIME_SERIES="${EXCLUDE_TIME_SERIES:-1}"

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-8}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
if [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
  echo "[skip] shard index ${SHARD_INDEX} outside num_shards=${NUM_SHARDS}"
  exit 0
fi

mkdir -p "$OUTPUT_DIR" "$SUMMARY_DIR"
OUTPUT_PATH="${OUTPUT_DIR}/near_iso_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.csv"
SUMMARY_PATH="${SUMMARY_DIR}/near_iso_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.summary.json"

if [ -f "$OUTPUT_PATH" ] && [ "$OVERWRITE" != "1" ]; then
  echo "[skip] Output exists: $OUTPUT_PATH (set OVERWRITE=1 to rebuild)"
  exit 0
fi

TS_FLAG="--exclude-time-series"
if [ "$EXCLUDE_TIME_SERIES" = "0" ]; then
  TS_FLAG="--no-exclude-time-series"
fi

uv run python scripts/build_near_isotropic_manifest.py \
  --catalog-path "$CATALOG_PATH" \
  --output-path "$OUTPUT_PATH" \
  --summary-path "$SUMMARY_PATH" \
  --modalities "$MODALITIES" \
  --max-spacing-ratio "$MAX_SPACING_RATIO" \
  --max-spacing-mm "$MAX_SPACING_MM" \
  --max-rows "$MAX_ROWS" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  "$TS_FLAG"

echo "[done] shard ${SHARD_INDEX}/${NUM_SHARDS}: $OUTPUT_PATH"
