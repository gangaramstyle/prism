#!/usr/bin/env bash
# Source this inside cluster shells/jobs to keep caches local to project/tmp.
# Intentionally avoid `set -euo pipefail` because this file is sourced.

export PATH="$HOME/.local/bin:$PATH"

# Project-local uv/pip cache.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"

# W&B local cache/config; job scripts may override to /tmp per-job.
export WANDB_DIR="${WANDB_DIR:-$PWD}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb-config}"

# Shared model caches (prefer project storage on cluster).
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch-cache}"

echo "cluster-runtime-env loaded"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "TORCH_HOME=${TORCH_HOME}"
