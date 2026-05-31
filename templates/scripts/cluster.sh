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

if [[ $# -eq 0 ]]; then
  echo "Usage: scripts/cluster.sh '<remote command>'" >&2
  exit 1
fi

if [[ -n "${SSH_OPTS}" ]]; then
  read -r -a SSH_OPTS_ARR <<<"${SSH_OPTS}"
else
  SSH_OPTS_ARR=()
fi

REMOTE_CMD="$*"
ssh "${SSH_OPTS_ARR[@]}" "${CLUSTER_HOST}" "cd ${CLUSTER_PROJECT} && ${REMOTE_CMD}"
