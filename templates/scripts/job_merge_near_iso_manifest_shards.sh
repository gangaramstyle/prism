#!/bin/bash
#SBATCH --job-name=prism-neariso-merge
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

mkdir -p logs results/manifests

SHARD_DIR="${SHARD_DIR:-results/manifests/pmbb_catalog_near_iso_shards}"
OUTPUT_PATH="${OUTPUT_PATH:-results/manifests/pmbb_catalog_near_iso.csv}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_PATH}.summary.json}"
REQUIRE_NUM_SHARDS="${REQUIRE_NUM_SHARDS:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [ -f "$OUTPUT_PATH" ] && [ "$OVERWRITE" != "1" ]; then
  echo "[skip] Output exists: $OUTPUT_PATH (set OVERWRITE=1 to rebuild)"
  exit 0
fi

uv run python scripts/merge_near_isotropic_manifest_shards.py \
  --shard-dir "$SHARD_DIR" \
  --output-path "$OUTPUT_PATH" \
  --summary-path "$SUMMARY_PATH" \
  --require-num-shards "$REQUIRE_NUM_SHARDS" \
  --dedupe-on-series-path

echo "[done] merged manifest: $OUTPUT_PATH"
