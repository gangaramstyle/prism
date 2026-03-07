#!/bin/bash
set -euo pipefail

PARTITION="${PARTITION:-dgx-b200}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-128G}"
TIME_LIMIT="${TIME_LIMIT:-73:00:00}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
JOB_SCRIPT="${SCRIPT_DIR}/job_train_prism_ssl.sh"

cmd=(
  sbatch
  --partition="${PARTITION}"
  --gpus="${GPUS}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM}"
  --time="${TIME_LIMIT}"
  --export=ALL
  "${JOB_SCRIPT}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

echo "[submit] partition=${PARTITION} gpus=${GPUS} cpus=${CPUS_PER_TASK} mem=${MEM} time=${TIME_LIMIT}" >&2
exec "${cmd[@]}"
