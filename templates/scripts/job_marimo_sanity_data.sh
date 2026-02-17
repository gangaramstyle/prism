#!/bin/bash
#SBATCH --job-name=prism-marimo-data
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/prism-ssl/templates}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-notebooks/sanity_data_pipeline.py}"
MARIMO_PORT="${MARIMO_PORT:-2718}"
CATALOG_PATH="${CATALOG_PATH:-$HOME/nvreason/data/pmbb_catalog.csv}"

cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Run scripts/setup_betty_uv.sh first." >&2
  exit 1
fi

mkdir -p logs
export CATALOG_PATH

NODE_NAME="$(hostname -s)"
echo "============================================"
echo "Marimo notebook: $NOTEBOOK_PATH"
echo "Node: $NODE_NAME"
echo "Port: $MARIMO_PORT"
echo "Catalog: $CATALOG_PATH"
echo ""
echo "Tunnel from local machine:"
echo "  ssh -L ${MARIMO_PORT}:${NODE_NAME}:${MARIMO_PORT} ${USER}@login.betty.parcc.upenn.edu"
echo ""
echo "Open in browser:"
echo "  http://localhost:${MARIMO_PORT}"
echo "============================================"

uv run marimo edit "$NOTEBOOK_PATH" --mcp --no-token --port "$MARIMO_PORT" --host 0.0.0.0
