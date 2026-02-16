#!/bin/bash
#SBATCH --job-name=prism-ssl-ablate
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/prism-ssl}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Run scripts/setup_betty_uv.sh first." >&2
  exit 1
fi
mkdir -p logs results/ablations

TMP_BASE="/tmp/${USER}/prism_ssl/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

BATCHES=(32 64)
POOLS=(16 24)
SUPCON_WEIGHTS=(0.1 0.2 0.4)
RAMP_STEPS=(2000 5000)

TASK_ID=${SLURM_ARRAY_TASK_ID}
B_IDX=$(( TASK_ID / (2 * 3 * 2) ))
REM1=$(( TASK_ID % (2 * 3 * 2) ))
P_IDX=$(( REM1 / (3 * 2) ))
REM2=$(( REM1 % (3 * 2) ))
W_IDX=$(( REM2 / 2 ))
R_IDX=$(( REM2 % 2 ))

BATCH=${BATCHES[$B_IDX]}
POOL=${POOLS[$P_IDX]}
SUPCON=${SUPCON_WEIGHTS[$W_IDX]}
RAMP=${RAMP_STEPS[$R_IDX]}

RUN_NAME="core24_b${BATCH}_p${POOL}_s${SUPCON}_r${RAMP}_${SLURM_ARRAY_TASK_ID}"
SUMMARY_PATH="results/ablations/${RUN_NAME}.json"
CATALOG_PATH="${CATALOG_PATH:-data/pmbb_catalog.csv.gz}"

if [ -f "$SUMMARY_PATH" ]; then
  echo "[skip] Summary exists: $SUMMARY_PATH"
  exit 0
fi

uv run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in job environment")
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

uv run python scripts/train_prism_ssl.py \
  --config configs/baseline.yaml \
  --catalog-path "$CATALOG_PATH" \
  --batch-size "$BATCH" \
  --warm-pool-size "$POOL" \
  --loss-weight-supcon "$SUPCON" \
  --supcon-ramp-steps "$RAMP" \
  --wandb-run-name "$RUN_NAME" \
  --wandb-mode online \
  --tmp-run-dir "$TMP_BASE" \
  --summary-output "$SUMMARY_PATH"
