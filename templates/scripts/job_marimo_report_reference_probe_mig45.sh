#!/bin/bash
#SBATCH --job-name=prism-marimo-reportref
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/prism-ssl/templates}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-notebooks/report_reference_probe.py}"
MARIMO_PORT="${MARIMO_PORT:-2721}"
REPORT_REF_PATH="${REPORT_REF_PATH:-$HOME/prism-ssl/templates/results/report_refs/pmbb_report_refs.parquet}"
MARIMO_ENABLE_MCP="${MARIMO_ENABLE_MCP:-1}"

cd "$REPO_ROOT"
source scripts/cluster-runtime-env.sh
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Run scripts/setup_betty_uv.sh first." >&2
  exit 1
fi

TMP_BASE="/tmp/${USER}/prism_report_ref_probe/${SLURM_JOB_ID}"
export TMPDIR="${TMP_BASE}/tmp"
export PRISM_NOTEBOOK_TMP="${TMP_BASE}"
mkdir -p logs "$TMPDIR"
export REPORT_REF_PATH

UV_RUN_ARGS=(run --with "ipyniivue>=2.2.2")
if [[ "$MARIMO_ENABLE_MCP" == "1" ]]; then
  UV_RUN_ARGS+=(--with "marimo[mcp]>=0.23.8")
fi

MARIMO_ARGS=(edit "$NOTEBOOK_PATH" --headless --no-token --port "$MARIMO_PORT" --host 127.0.0.1)
if [[ "$MARIMO_ENABLE_MCP" == "1" ]] && uv "${UV_RUN_ARGS[@]}" marimo edit --help 2>/dev/null | grep -q -- "--mcp"; then
  MARIMO_ARGS+=(--mcp)
fi

NODE_NAME="$(hostname -s)"
echo "============================================"
echo "Marimo notebook: $NOTEBOOK_PATH"
echo "Node: $NODE_NAME"
echo "Port: $MARIMO_PORT"
echo "Report refs: $REPORT_REF_PATH"
echo "MCP requested: $MARIMO_ENABLE_MCP"
echo ""
echo "Tunnel from local machine:"
echo "  ssh -N -J ${USER}@login.betty.parcc.upenn.edu ${USER}@${NODE_NAME} -L ${MARIMO_PORT}:127.0.0.1:${MARIMO_PORT}"
echo ""
echo "Open in browser:"
echo "  http://127.0.0.1:${MARIMO_PORT}"
echo "============================================"

uv "${UV_RUN_ARGS[@]}" marimo "${MARIMO_ARGS[@]}"
