---
description: Generate a SLURM launch script for a model/task
argument-hint: <model name> <gpu type> [partition]
allowed-tools: Read, Write, Glob
---

Generate a SLURM launch script based on the user's request: $ARGUMENTS

1. Look at existing launch scripts in `scripts/launch_*.sh` for patterns
2. Match the module loading, environment setup, and job array structure
3. Choose appropriate GPU type and partition:
   - **CBICA:** p100 (12GB), a40 (48GB), a100 (80GB) — specify with `--gpus-per-node=type:1`
   - **Betty:** dgx-b200 partition — specify with `--gpus=1`
4. Set memory to 2x GPU VRAM as a starting point
5. Set time limit conservatively (2-4 hours for inference, 8-24 for training)
6. Include idempotent output checking (skip if result exists)
7. Include HuggingFace cache environment variables

Write the script to `scripts/launch_<model_name>.sh`.
