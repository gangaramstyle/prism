# PRISM SSL v1 Implementation Guide

## 1. Purpose
This document is the implementation contract for building a new Prism SSL training repository from scratch.

It captures:
1. Lessons from `nvreason` profiling and loader experiments.
2. Design intent from previous end-to-end pipelines (`oneweek/r2`, `rsna25`).
3. Betty cluster constraints and home-disk budget constraints.
4. A concrete build plan for a train + ablation ready v1.

This guide is decision-complete for implementation.

## 2. Locked Product Decisions
1. Execution target for v1 is single-node, single-GPU Betty (DGX B200).
2. Data scope for v1 is CT+MR with strict scan-quality filtering.
3. Canonical baseline uses two-stage loss ramp:
4. Stage A: geometry/window objectives only.
5. Stage B: phase in SupCon to a configured target weight.
6. W&B online is the source of truth for run metrics and run history.
7. Marimo exploration notebook reads checkpoints + hyperparams from W&B + scan data stored locally.
8. All transient heavy files are placed under `/tmp/$USER`.
9. Home directory usage is bounded with soft and hard quota guardrails.

## 3. Non-Goals
1. Multi-node distributed training.
2. Final downstream supervised benchmark integration.
3. Broad model architecture search.
4. Full automation platform integration.

## 4. Repo Architecture

### 4.1 Target tree
```text
prism-ssl/
  pyproject.toml
  README.md
  docs/
    IMPLEMENTATION_GUIDE.md
  prism_ssl/
    config/
      schema.py
      defaults.py
    data/
      catalog.py
      filters.py
      sharded_dataset.py
      collate.py
      preflight.py
    model/
      backbone.py
      heads.py
      loss.py
      schedules.py
    train/
      train_loop.py
      checkpoint.py
      logging.py
      quota_guard.py
      metrics.py
    eval/
      proxy_metrics.py
      embedding_probe.py
    utils/
      hashing.py
      time.py
      fs.py
      seeds.py
  scripts/
    train_prism_ssl.py
    job_train_prism_ssl.sh
    job_ablate_core24.sh
  notebooks/
    explore_runs.py
  tests/
    test_hashing.py
    test_sharded_dataset.py
    test_loss_schedule.py
    test_supcon.py
    test_train_smoke.py
```

### 4.2 Data flow
1. Read PMBB catalog.
2. Resolve deterministic NIfTI paths.
3. Filter invalid scans via preflight checks.
4. Create sharded iterable dataset with warm pool.
5. Yield paired views plus supervision targets.
6. Compute model forward on both views.
7. Compute multi-head losses with staged weighting.
8. Log metrics and checkpoints.
9. Publish run state to W&B.

## 5. Core Contracts

### 5.1 Config schema
```python
from dataclasses import dataclass

@dataclass
class DataConfig:
    catalog_path: str
    n_scans: int
    modality_filter: tuple[str, ...]
    n_patches: int
    patch_mm: float
    workers: int
    warm_pool_size: int
    visits_per_scan: int
    max_prefetch_replacements: int
    use_local_scratch: bool

@dataclass
class LossConfig:
    w_distance: float
    w_rotation: float
    w_window: float
    w_supcon_target: float
    supcon_temperature: float
    supcon_warmup_steps: int
    supcon_ramp_steps: int
    normalize_targets: bool

@dataclass
class TrainConfig:
    batch_size: int
    max_steps: int
    log_every: int
    lr: float
    weight_decay: float
    precision: str
    seed: int
```

### 5.2 Scan record contract
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ScanRecord:
    scan_id: str
    series_id: str
    modality: str
    nifti_path: str
    series_path: str
```

### 5.3 Batch contract
```python
# Tensor shapes shown for batch size B and patches N.
PrismBatch = {
    "patches_a": FloatTensor[B, N, 16, 16, 1],
    "positions_a": FloatTensor[B, N, 3],
    "patches_b": FloatTensor[B, N, 16, 16, 1],
    "positions_b": FloatTensor[B, N, 3],
    "center_distance_mm": FloatTensor[B],
    "rotation_delta_deg": FloatTensor[B, 3],
    "window_delta": FloatTensor[B, 2],
    "series_label": LongTensor[B],
    "scan_id": list[str],
    "series_id": list[str],
}
```

## 6. Data Pipeline Implementation

### 6.1 Catalog and deterministic IDs
1. Load PMBB catalog using `polars`.
2. Keep CT and MR rows.
3. Resolve `nifti_path` deterministically.
4. Build globally unique `scan_id` using stable source fields.
5. Build `series_id` from `series_path` or deterministic fallback.

### 6.2 Preflight filtering
Run preflight before training:
1. File exists and readable.
2. NIfTI load succeeds.
3. Affine is invertible.
4. Voxel shape supports requested patch shape.
5. Robust stats can be computed.

Persist preflight report to JSON:
```json
{
  "kept": 4932,
  "rejected": 68,
  "reasons": {
    "affine_singular": 31,
    "shape_too_small": 24,
    "nifti_load_failed": 13
  }
}
```

### 6.3 Sharded warm pool
Use iterable dataset with deterministic worker assignment.

Assignment:
```python
def shard_worker(scan_id: str, n_workers: int) -> int:
    import hashlib
    k = int(hashlib.sha256(scan_id.encode()).hexdigest(), 16)
    return k % n_workers
```

Rules:
1. Fixed warm-pool slots per worker.
2. Each slot has current scan and optional replacement future.
3. Replacement is non-blocking.
4. Continue sampling from current slot until replacement completes.
5. On future failure, raise immediately if strict mode is on.
6. Scratch files are cleaned at worker shutdown, not per-eviction.

### 6.4 Pair-view target generation
For each sampled scan:
1. Draw view A and view B with independent seeds.
2. Extract tensors and metadata for each view.
3. Compute target deltas:
4. `center_distance_mm = ||center_b - center_a||`.
5. `rotation_delta_deg = rot_b - rot_a`.
6. `window_delta = [wc_b - wc_a, ww_b - ww_a]`.

## 7. Model and Losses

### 7.1 Model heads
Use shared encoder and four heads:
1. Distance regression vs classification head.
2. Rotation delta regression vs classification head.
3. Window center regression vs classification head.
4. Window width regression vs classification head.
5. Projection head for SupCon.

### 7.2 SupCon objective
Use normalized embeddings and same-series positives.

```python
def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temp: float) -> torch.Tensor:
    n = emb.shape[0]
    sim = (emb @ emb.T) / max(temp, 1e-6)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logits = sim.masked_fill(eye, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return emb.new_tensor(0.0)
    per_anchor = -(pos * logp).sum(dim=1) / pos_count.clamp(min=1)
    return per_anchor[valid].mean()
```

### 7.3 Two-stage loss ramp
Stage A:
1. `w_supcon = 0.0`.
2. optimize distance, rotation, window only.

Stage B:
1. Ramp `w_supcon` linearly over `supcon_ramp_steps`.
2. Hold at target weight after ramp.

```python
def supcon_weight(step: int, warmup: int, ramp: int, target: float) -> float:
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return target
    t = min(1.0, (step - warmup) / float(ramp))
    return target * t
```

## 8. Honest Throughput and Tail Metrics

Log at fixed interval and final summary:
1. `throughput_effective_patches_per_sec` from wall-clock measured time.
2. `patches_per_sec_step` per-step instantaneous throughput.
3. `total_step_time_ms` mean, p50, p95, p99, max.
4. `stall_steps_ge_2000ms` and `stall_steps_ge_10000ms`.
5. `replacement_completed_count`.
6. `replacement_failed_count`.
7. `replacement_wait_time_ms_total`.

Keep backward-compatible median fields if migrating from existing tooling.

## 9. Preventing Misleading Near-Zero Loss

Add explicit sanity instrumentation:
1. Log losses using scientific notation in stdout.
2. Log target means and standard deviations per head.
3. Log prediction means and standard deviations per head.
4. Log ratio `pred_std / target_std`.
5. Log SupCon positives-per-anchor stats.
6. Emit warning when any target std remains near-zero for prolonged window.
7. Keep both raw and normalized loss values.

This is mandatory for ablation readiness.

## 10. W&B and Storage Policy

### 10.1 W&B requirements
1. Online mode by default on cluster.
2. Log full config dictionary at run start.
3. Write all scalar metrics to W&B.
4. Upload artifact checkpoints every `N` steps.

### 10.2 Disk policy
1. Keep one local durable checkpoint (`last.ckpt`).
2. Temp artifact checkpoint is created under `/tmp` and deleted after upload.
3. W&B cache/artifact/temp dirs must all be under `/tmp/$USER/...`.
4. Add soft and hard home quota guardrails.

Suggested env in job script:
```bash
export WANDB_DIR="/tmp/$USER/prism_ssl/$SLURM_JOB_ID/wandb"
export WANDB_CACHE_DIR="/tmp/$USER/prism_ssl/$SLURM_JOB_ID/wandb_cache"
export WANDB_ARTIFACT_DIR="/tmp/$USER/prism_ssl/$SLURM_JOB_ID/wandb_artifacts"
export TMPDIR="/tmp/$USER/prism_ssl/$SLURM_JOB_ID/tmp"
```

### 10.3 Resume order
1. local `last.ckpt`.
2. W&B artifact `latest`.
3. start from step 0 if neither exists.

## 11. Betty Job Specs

Default single-GPU run:
1. `--partition=dgx-b200`.
2. `--gpus=1`.
3. `--cpus-per-task=16`.
4. `--mem=256G`.
5. `--time=24:00:00`.

Ensure trap-based cleanup of `/tmp/$USER/prism_ssl/$SLURM_JOB_ID`.

## 12. Core 24-Run Ablation Matrix

Factors:
1. Batch size: `32`, `64`.
2. Warm pool size: `16`, `24`.
3. SupCon target weight: `0.1`, `0.2`, `0.4`.
4. SupCon ramp steps: `2000`, `5000`.
5. Regressions vs Binary Classification per Axis

Total runs: `24`.

Fixed settings:
1. `n_patches=1024`.
2. `n_workers=8`.
3. `visits_per_scan=100`.
4. `storage_mode=sharded`.
5. `method=optimized_fused`.
6. `use_local_scratch=true`.

Ranking:
1. Primary: proxy-quality composite.
2. Secondary: effective throughput with stall penalties.

## 13. Marimo Exploration Notebook

Create `notebooks/explore_runs.py`.

Requirements:
1. Read run metadata and history from W&B API.
2. Optional `/tmp` cache with TTL.
3. Interactive filters for run tags and config keys.
4. Produce at least these views:
5. Loss decomposition over time.
6. Throughput and tail latency diagnostics.
7. Replacement/failure counters.
8. Proxy metrics table and ranking.

Required utility functions:
```python
def fetch_runs(entity: str, project: str, filters: dict) -> pl.DataFrame: ...
def fetch_history(run_ids: list[str], keys: list[str]) -> pl.DataFrame: ...
def compute_proxy_score(df_runs: pl.DataFrame, df_hist: pl.DataFrame) -> pl.DataFrame: ...
def summarize_best_configs(df: pl.DataFrame) -> pl.DataFrame: ...
```

## 14. Testing Plan

### 14.1 Unit tests
1. deterministic sharding assignment.
2. warm-pool replacement state transitions.
3. SupCon loss positive-mask behavior.
4. loss ramp schedule math.
5. quota-guard pruning behavior.

### 14.2 Integration tests
1. CPU smoke training for 100-200 steps.
2. background replacement failure path returns explicit error status.
3. checkpoint resume from local and artifact fallback.

### 14.3 Cluster smoke tests
1. one 1000-step run with canonical config.
2. one low-batch and one high-batch ablation run.
3. verify W&B metrics and artifact checkpoints.

## 15. Acceptance Criteria

A run is considered v1-ready when:
1. no detached thread exceptions are present in stderr.
2. no run reports `status=ok` when replacement failures occurred in strict mode.
3. throughput tails and stall counters explain observed effective throughput.
4. loss and target diagnostics are non-degenerate and interpretable.
5. home usage remains below hard quota limit throughout run.
6. 24-run ablation results are queryable and rankable in notebook.

## 16. Phased Execution Plan

### Phase 0: Repo bootstrap
1. scaffold source tree.
2. add config schema.
3. add scripts and notebook skeletons.

### Phase 1: Data path
1. implement catalog resolution.
2. implement preflight filters.
3. implement sharded warm-pool dataset.
4. add collate contract and tests.

### Phase 2: Model and training
1. implement encoder and heads.
2. implement loss bundle and schedule.
3. implement training loop and metrics.
4. integrate W&B and checkpoint policy.

### Phase 3: Validation and ablations
1. run smoke tests.
2. run 24-run ablation matrix.
3. analyze in marimo and publish baseline recommendation.

## 17. Known Risks and Mitigations
1. Loader failure from malformed scans.
2. Mitigation: preflight reject + strict replacement error surfacing.
3. Memory pressure at larger batches.
4. Mitigation: baseline with batch 32/64 and explicit VRAM logging.
5. Misleading low loss from degenerate targets.
6. Mitigation: mandatory target-distribution diagnostics.
7. Home quota lockout.
8. Mitigation: strict `/tmp` policy + single durable checkpoint + hard stop.

## 18. Engineer Handoff Checklist
1. run unit tests locally.
2. run CPU smoke.
3. run single Betty smoke.
4. confirm W&B logging and artifact uploads.
5. confirm notebook can rank runs from W&B.
6. sign off against acceptance criteria.

---

This document is intended to be copied directly into a new Prism SSL repository and treated as the implementation source of truth.
