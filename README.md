# PRISM Pipeline Blueprint

This folder is a copy-ready implementation blueprint for rebuilding the Prism SSL training pipeline in a fresh repository.

## What is included
- `docs/IMPLEMENTATION_GUIDE.md`: decision-complete technical guide for engineers.
- `templates/configs/`: baseline and ablation config templates.
- `templates/scripts/`: Betty SLURM script templates.
- `templates/notebooks/`: marimo exploration notebook template.
- `templates/src/`: source tree scaffold for the new repo.
- `templates/tests/`: test scaffold and checklist.

## Primary constraints captured
- Betty (DGX B200) single-GPU first.
- CT+MR with strict scan-quality filtering.
- W&B online logging from step 0.
- Home quota guardrails (~30 GB free).
- Heavy transient files in `/tmp/$USER`.
- Sharded warm-pool streaming for data loading.

## Intended use
1. Copy `PRISM/` into a new repository root.
2. Use `docs/IMPLEMENTATION_GUIDE.md` as the build contract.
3. Fill in templates under `templates/` during implementation.
