# prism-ssl

Pair2-only PRISM SSL training stack for CT and MR.

Current baseline:
- same-scan paired views only
- pair-relation head over `(delta_x, delta_y, delta_z, delta_wc, delta_ww)` sign targets
- dual SupCon heads on the shared series CLS
- self-MIM only
- variable source patch scale from `16 mm` to `64 mm`, resized back to `16x16`
- TotalSegmentator-guided body-center sampling enabled by default when available
- slow global-to-local curriculum for paired centers

## Quick start
1. Create the environment with `uv sync --extra dev`.
2. Run the test suite with `uv run pytest -q tests`.
3. Run a short local smoke command:
   `uv run python scripts/train_prism_ssl.py --config configs/baseline.yaml --wandb-mode disabled --summary-output results/train/local_smoke.json`
4. Submit the default Betty job with:
   `sbatch scripts/job_train_prism_ssl.sh`

## Baseline config
The maintained baseline is [`configs/baseline.yaml`](/Users/vineethgangaram/prism/templates/configs/baseline.yaml).

Important defaults:
- `data.source_patch_mm_min = 16.0`
- `data.source_patch_mm_max = 64.0`
- `data.source_patch_mm_distribution = log_uniform`
- `loss.w_supcon_instance_target = 0.1`
- `loss.w_supcon_protocol_target = 0.1`
- `loss.w_patch_size = 0.25`

## Training objectives
- `pair_relation`: binary sign prediction for `x/y/z` center deltas and `wc/ww` window deltas
- `supcon_instance`: supervised contrastive loss over exact `series_id`
- `supcon_protocol`: supervised contrastive loss over minimally normalized `modality::series_description`
- `mim`: masked patch reconstruction from the same view only
- `patch_size`: per-view regression to `log2(source_patch_mm)`

## Data path
1. Load the PMBB catalog and sample candidate scans.
2. Hash-shard scans across workers.
3. Keep a warm pool of loaded NIfTI scans per worker and replace slots asynchronously.
4. Sample view A and B from the same scan.
5. Sample source patch scale independently per view.
6. Extract native-plane patches covering that physical footprint, then resize back to `16x16`.
7. Collate pairwise tensors plus exact-series and protocol-family labels.
8. Train the pair-relation, dual-SupCon, self-MIM, and patch-size heads.

## Offline validation cache
The maintained validation artifact is an offline CT semantic view cache built with [`scripts/validation/build_ct_view_validation_cache.py`](/Users/vineethgangaram/prism/templates/scripts/validation/build_ct_view_validation_cache.py).

Defaults:
- CT only
- TotalSegmentator required
- `128` scans
- `16` cached single views per scan
- fixed semantic target set: heart, lung, esophagus, stomach, right kidney, spleen, bladder, pancreas
- shard payloads in `.pt` plus metadata in parquet
- full ordered within-scan comparison grid derived at load time (`16 x 16` per scan)

The cache stores enough view-level tensors and metadata to reconstruct the current training losses offline:
- pair-relation targets from derived within-scan view pairs
- exact-instance and protocol-family SupCon labels
- patch-size targets
- self-MIM inputs

The cache builder uses prism-local window retries to filter uninformative all-black or all-white views. This is validation-specific behavior and is not wired into the training loop.

Example build command:
`uv run python scripts/validation/build_ct_view_validation_cache.py --config-path configs/baseline.yaml --catalog-path data/pmbb_catalog.csv.gz`

## Betty notes
- W&B online is the default cluster mode.
- Heavy transient files go under `/tmp/$USER`.
- Broken scans are skipped, and the run aborts if the broken ratio exceeds the configured threshold after the minimum-attempt gate.

## Ablations
[`configs/ablation_core24.yaml`](/Users/vineethgangaram/prism/templates/configs/ablation_core24.yaml) now defines the clean pair2 sweep over:
- exact-series SupCon weight
- protocol-family SupCon weight
- patch-size weight
- maximum source patch size

The current Cartesian product is 36 runs, even though the historical filename still says `core24`.

## Notebook support
The maintained analysis notebook is [`notebooks/explore_runs.py`](/Users/vineethgangaram/prism/templates/notebooks/explore_runs.py).

Older exploratory sanity notebooks are still in the repo, but they are not part of the maintained training surface for this branch.
