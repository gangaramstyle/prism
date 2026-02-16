#!/bin/bash
# Minimal Betty bootstrap for Prism SSL using uv.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/prism-ssl/templates}"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv sync --extra dev

uv run python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
else:
    print("CUDA not visible on this node (expected on login nodes).")
PY

echo "Setup complete at $REPO_ROOT"
