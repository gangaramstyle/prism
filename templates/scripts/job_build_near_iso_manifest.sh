#!/bin/bash
#SBATCH --job-name=prism-neariso
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

mkdir -p logs results/manifests

CATALOG_PATH="${CATALOG_PATH:-$HOME/nvreason/data/pmbb_catalog.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-results/manifests/pmbb_catalog_near_iso.csv}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_PATH}.summary.json}"
MODALITIES="${MODALITIES:-CT,MR}"
MAX_SPACING_RATIO="${MAX_SPACING_RATIO:-1.2}"
MAX_SPACING_MM="${MAX_SPACING_MM:-0}"
MAX_ROWS="${MAX_ROWS:-0}"
OVERWRITE="${OVERWRITE:-0}"
EXCLUDE_TIME_SERIES="${EXCLUDE_TIME_SERIES:-1}"

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
  "$TS_FLAG"

echo "[done] near-isotropic manifest: $OUTPUT_PATH"
echo "[done] summary: $SUMMARY_PATH"
