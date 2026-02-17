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

## Marimo sanity notebooks
1. Data pipeline sanity:
   `uv run marimo edit notebooks/sanity_data_pipeline.py`
2. Batch/label sanity:
   `uv run marimo edit notebooks/sanity_batch_labels.py`
3. On cluster, set `CATALOG_PATH` first if needed:
   `export CATALOG_PATH=~/nvreason/data/pmbb_catalog.csv`

### Betty runbook (recommended)
Run notebooks on a compute allocation (not login nodes).

1. Submit data-pipeline sanity notebook:
   `cd ~/prism-ssl/templates && CATALOG_PATH=~/nvreason/data/pmbb_catalog.csv sbatch scripts/job_marimo_sanity_data.sh`
2. Submit batch/label sanity notebook:
   `cd ~/prism-ssl/templates && CATALOG_PATH=~/nvreason/data/pmbb_catalog.csv sbatch scripts/job_marimo_sanity_batch.sh`
3. Get node for a job:
   `cd ~/prism-ssl/templates && squeue -j <jobid> -o "%.18i %.8T %N"`
4. Tunnel from local machine (replace `<node>`):
   - data notebook (port 2718): `ssh -L 2718:<node>:2718 gangaram@login.betty.parcc.upenn.edu`
   - batch notebook (port 2719): `ssh -L 2719:<node>:2719 gangaram@login.betty.parcc.upenn.edu`
5. Open in browser:
   - data notebook: `http://localhost:2718`
   - batch notebook: `http://localhost:2719`

## Required runtime policies
- W&B online logging.
- `/tmp/$USER` for heavy transient files.
- Single durable local checkpoint.
- Home quota guardrails enabled.
- Broken scan policy: skip broken scans, abort run if broken ratio > 10% after 200 attempts.
