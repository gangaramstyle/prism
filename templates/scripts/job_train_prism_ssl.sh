#!/bin/bash
#SBATCH --job-name=prism-ssl-train
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/prism-ssl}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Run scripts/setup_betty_uv.sh first." >&2
  exit 1
fi
mkdir -p logs results/train

TMP_BASE="/tmp/${USER}/prism_ssl/${SLURM_JOB_ID}"
export WANDB_DIR="${TMP_BASE}/wandb"
export WANDB_CACHE_DIR="${TMP_BASE}/wandb_cache"
export WANDB_ARTIFACT_DIR="${TMP_BASE}/wandb_artifacts"
export WANDB_MODE="online"
export TMPDIR="${TMP_BASE}/tmp"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR"

cleanup_tmp() {
  rm -rf "$TMP_BASE" || true
}
trap cleanup_tmp EXIT

SUMMARY_PATH="results/train/prism_ssl_${SLURM_JOB_ID}.json"
CATALOG_PATH="${CATALOG_PATH:-data/pmbb_catalog.csv.gz}"

uv run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in job environment")
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

uv run python scripts/train_prism_ssl.py \
  --config configs/baseline.yaml \
  --catalog-path "$CATALOG_PATH" \
  --wandb-mode online \
  --tmp-run-dir "$TMP_BASE" \
  --summary-output "$SUMMARY_PATH"
