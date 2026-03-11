#!/bin/bash
# 36-run clean pair2 ablation sweep:
# 3 exact-series SupCon weights x 3 protocol-family SupCon weights x 2 patch-size weights x 2 source patch maxima.
#SBATCH --job-name=prism-ssl-ablate
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0-35
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

SUPCON_INSTANCE_WEIGHTS=(0.05 0.1 0.2)
SUPCON_PROTOCOL_WEIGHTS=(0.05 0.1 0.2)
PATCH_WEIGHTS=(0.1 0.25)
SOURCE_PATCH_MAXES=(48 64)
MODEL_NAME="${MODEL_NAME:-vit_l}"
N_PATCHES="${N_PATCHES:-256}"
WORKERS="${WORKERS:-8}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
I_IDX=$(( TASK_ID / (3 * 2 * 2) ))
REM1=$(( TASK_ID % (3 * 2 * 2) ))
P_IDX=$(( REM1 / (2 * 2) ))
REM2=$(( REM1 % (2 * 2) ))
W_IDX=$(( REM2 / 2 ))
M_IDX=$(( REM2 % 2 ))

SUPCON_INSTANCE=${SUPCON_INSTANCE_WEIGHTS[$I_IDX]}
SUPCON_PROTOCOL=${SUPCON_PROTOCOL_WEIGHTS[$P_IDX]}
PATCH_WEIGHT=${PATCH_WEIGHTS[$W_IDX]}
SOURCE_PATCH_MAX=${SOURCE_PATCH_MAXES[$M_IDX]}

RUN_NAME="clean_pair2_si${SUPCON_INSTANCE}_sp${SUPCON_PROTOCOL}_wp${PATCH_WEIGHT}_max${SOURCE_PATCH_MAX}_${SLURM_ARRAY_TASK_ID}"
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
  --model-name "$MODEL_NAME" \
  --n-patches "$N_PATCHES" \
  --workers "$WORKERS" \
  --loss-weight-supcon-instance "$SUPCON_INSTANCE" \
  --loss-weight-supcon-protocol "$SUPCON_PROTOCOL" \
  --loss-weight-patch-size "$PATCH_WEIGHT" \
  --source-patch-mm-max "$SOURCE_PATCH_MAX" \
  --wandb-run-name "$RUN_NAME" \
  --wandb-mode online \
  --tmp-run-dir "$TMP_BASE" \
  --local-ckpt-dir "$TMP_BASE/checkpoints/<run_id>" \
  --summary-output "$SUMMARY_PATH"
