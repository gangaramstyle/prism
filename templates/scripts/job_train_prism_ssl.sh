#!/bin/bash
#SBATCH --job-name=prism-ssl-train
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=73:00:00
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
CATALOG_PATH="${CATALOG_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-}"
N_PATCHES="${N_PATCHES:-}"
WORKERS="${WORKERS:-}"
N_SCANS="${N_SCANS:-}"
WARM_POOL_SIZE="${WARM_POOL_SIZE:-}"
VISITS_PER_SCAN="${VISITS_PER_SCAN:-}"

uv run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in job environment")
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

TRAIN_ARGS=(
  --config configs/baseline.yaml
  --wandb-mode online
  --tmp-run-dir "$TMP_BASE"
  --local-ckpt-dir "$TMP_BASE/checkpoints/<run_id>"
  --summary-output "$SUMMARY_PATH"
)

if [[ -n "$CATALOG_PATH" ]]; then
  TRAIN_ARGS+=(--catalog-path "$CATALOG_PATH")
fi
if [[ -n "$MODEL_NAME" ]]; then
  TRAIN_ARGS+=(--model-name "$MODEL_NAME")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  TRAIN_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$MAX_STEPS" ]]; then
  TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$LR" ]]; then
  TRAIN_ARGS+=(--lr "$LR")
fi
if [[ -n "$N_PATCHES" ]]; then
  TRAIN_ARGS+=(--n-patches "$N_PATCHES")
fi
if [[ -n "$WORKERS" ]]; then
  TRAIN_ARGS+=(--workers "$WORKERS")
fi
if [[ -n "$N_SCANS" ]]; then
  TRAIN_ARGS+=(--n-scans "$N_SCANS")
fi
if [[ -n "$WARM_POOL_SIZE" ]]; then
  TRAIN_ARGS+=(--warm-pool-size "$WARM_POOL_SIZE")
fi
if [[ -n "$VISITS_PER_SCAN" ]]; then
  TRAIN_ARGS+=(--visits-per-scan "$VISITS_PER_SCAN")
fi

uv run python scripts/train_prism_ssl.py "${TRAIN_ARGS[@]}"
