#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.cluster.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

CLUSTER_HOST="${CLUSTER_HOST:-betty}"
CLUSTER_PROJECT="${CLUSTER_PROJECT:-~/prism-ssl/templates}"
SSH_OPTS="${SSH_OPTS:-}"

rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.uv-cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude 'data/' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude 'artifacts/' \
  --exclude 'wandb/' \
  --exclude 'checkpoints/' \
  --exclude 'tmp/' \
  --exclude '*.out' \
  --exclude '*.err' \
  --exclude '.DS_Store' \
  -e "ssh ${SSH_OPTS}" \
  "${CLUSTER_HOST}:${CLUSTER_PROJECT}/" "${REPO_ROOT}/"

echo "Synced ${CLUSTER_HOST}:${CLUSTER_PROJECT} -> local"
