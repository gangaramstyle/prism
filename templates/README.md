# prism-ssl

## Quick start
1. Create a virtual environment with `uv sync --extra dev`.
2. Run local tests with `uv run pytest -q tests`.
3. Run a short smoke train:
   `uv run python scripts/train_prism_ssl.py --config configs/baseline.yaml --wandb-mode disabled --summary-output results/train/local_smoke.json`
4. Submit on Betty with `sbatch scripts/job_train_prism_ssl.sh`.

## Betty (uv) bootstrap
1. Run `bash scripts/setup_betty_uv.sh` once after clone/pull.
2. Job scripts run a CUDA preflight (`torch.cuda.is_available()`) before training.

## Required runtime policies
- W&B online logging.
- `/tmp/$USER` for heavy transient files.
- Single durable local checkpoint.
- Home quota guardrails enabled.
- Broken scan policy: skip broken scans, abort run if broken ratio > 10% after 200 attempts.
