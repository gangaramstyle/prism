# PRISM SSL Implementation Guide

## Purpose
This document describes the maintained training design in the cleaned pair2 branch.

The active recipe is intentionally narrow:
- pair2 only
- no cross-series reconstruction path
- no rotation targets
- dual SupCon on a shared series CLS
- variable source patch scale with patch-size prediction

## Current Training Recipe

### Data sampling
1. Load PMBB catalog rows into `ScanRecord`.
2. Keep `scan_id`, `series_id`, `modality`, `series_path`, and `series_description`.
3. Hash-shard scans across workers.
4. Maintain a warm pool of loaded scans per worker.
5. Prefer TotalSegmentator body centers when available so sampling stays inside anatomy-rich regions.
6. Draw paired views `A` and `B` from the same scan.
7. Use the existing global-to-local curriculum to gradually reduce the average distance between paired centers.

### Variable source patch scale
Each view independently samples a physical source patch size from:
- `source_patch_mm_min`
- `source_patch_mm_max`
- `source_patch_mm_distribution`

Default baseline:
- min `16 mm`
- max `64 mm`
- `log_uniform`

The sampled footprint is extracted in the native acquisition plane, then resized back to `16x16` pixels before embedding.

The thin axis remains one voxel.

### Labels carried by the batch
The pairwise batch contract is:

```python
PrismBatch = {
    "patches_a": FloatTensor[B, N, 16, 16, 1],
    "positions_a": FloatTensor[B, N, 3],
    "patches_b": FloatTensor[B, N, 16, 16, 1],
    "positions_b": FloatTensor[B, N, 3],
    "center_delta_mm": FloatTensor[B, 3],
    "center_distance_mm": FloatTensor[B],
    "window_delta": FloatTensor[B, 2],
    "series_instance_label": LongTensor[B],
    "series_protocol_label": LongTensor[B],
    "source_patch_mm_a": FloatTensor[B],
    "source_patch_mm_b": FloatTensor[B],
    "scan_id": list[str],
    "series_id": list[str],
    "protocol_key": list[str],
}
```

`protocol_key` is defined as:

```python
protocol_key = f"{modality.upper()}::{normalize_series_description(series_description)}"
```

Normalization is intentionally minimal:
- trim whitespace
- uppercase
- collapse runs of non-alphanumeric characters to `_`
- strip leading and trailing `_`
- fallback to `UNKNOWN`

## Model Contract

### Encoder streams
The encoder keeps two CLS streams per view:
- `view_cls_token`: used for pair relation and patch-size prediction
- `series_cls_token`: used for both SupCon heads

### Output heads
The maintained output surface is pair2-only:

```python
PrismModelOutput(
    pair_relation_logits,
    proj_instance_a,
    proj_instance_b,
    proj_protocol_a,
    proj_protocol_b,
    patch_size_pred_a,
    patch_size_pred_b,
    mim_pred_a,
    mim_pred_b,
    mim_target_a,
    mim_target_b,
)
```

There are no cross-series or auxiliary-memory reconstruction outputs in the active code path.

## Objectives

### Pair relation
`pair_relation_logits` has shape `[B, 5]` and predicts the sign of:
- `delta_x_mm`
- `delta_y_mm`
- `delta_z_mm`
- `delta_wc`
- `delta_ww`

The first three axes use the existing ambiguity mask:
- if `abs(delta_mm) < 1.0`, that axis is excluded from loss and accuracy

This objective is still controlled by `loss.w_distance` for config compatibility, but in logs and docs it should be treated as pair relation, not metric distance regression.

### Dual SupCon
Both SupCon heads operate on the same shared `series_cls` stream.

Head 1:
- exact series instance
- labels come from `series_id`

Head 2:
- protocol family
- labels come from `protocol_key`

Both use the same supervised-contrastive loss form and the same warmup/ramp schedule, but they have separate target weights:
- `loss.w_supcon_instance_target`
- `loss.w_supcon_protocol_target`

### Self-MIM
Only self-reconstruction remains:
- mask a subset of patches in each view
- reconstruct them from the same view's visible patch tokens

There is no register-token reconstruction path and no cross-series reconstruction path.

### Patch-size prediction
Each view predicts:

```python
log2(source_patch_mm)
```

using a small MLP on `view_cls`.

Loss:
- `SmoothL1` on `log2(mm)`
- averaged across views A and B
- weighted by `loss.w_patch_size`

## Config Surface

### Data
```python
DataConfig(
    catalog_path,
    n_scans,
    modality_filter,
    n_patches,
    source_patch_mm_min,
    source_patch_mm_max,
    source_patch_mm_distribution,
    workers,
    warm_pool_size,
    visits_per_scan,
    max_prefetch_replacements,
    use_totalseg_body_centers,
    pair_local_curriculum_steps,
    pair_local_final_prob,
    pair_local_start_radius_mm,
    pair_local_end_radius_mm,
    use_local_scratch,
)
```

### Loss
```python
LossConfig(
    w_distance,
    w_mim,
    w_supcon_instance_target,
    w_supcon_protocol_target,
    w_patch_size,
    supcon_temperature,
    supcon_warmup_steps,
    supcon_ramp_steps,
)
```

Removed from the active schema:
- `data.sample_unit`
- `data.patch_mm`
- cross-series auxiliary reconstruction weights
- `loss.mim_aux_warmup_steps`
- `loss.mim_aux_ramp_steps`

## Metrics
The maintained training loop logs:
- `train/loss`
- `train/loss_pair_relation`
- `train/loss_supcon_instance`
- `train/loss_supcon_protocol`
- `train/loss_patch_size`
- `train/loss_mim`
- `train/pair_relation_acc`
- `train/pair_relation_acc_shared`
- `train/w_supcon_instance`
- `train/w_supcon_protocol`
- `train/patch_size_mae_mm`
- `train/source_patch_mm_mean`
- `train/source_patch_mm_min`
- `train/source_patch_mm_max`
- `train/supcon_instance_positives_per_anchor_mean`
- `train/supcon_protocol_positives_per_anchor_mean`

Loader and health metrics remain part of the run summary:
- effective throughput
- replacement counts
- broken-scan ratio
- loaded-with-body ratio
- sampled-body-center ratio

## Runtime Constraints
- W&B online remains the default cluster mode.
- Heavy transient state stays under `/tmp/$USER`.
- Only one durable local checkpoint is kept by default.
- Broken scans are skipped, but the run aborts once the configured broken-ratio threshold is exceeded after the minimum-attempt gate.

## Ablations
The maintained ablation surface now focuses on:
- exact-series SupCon weight
- protocol-family SupCon weight
- patch-size loss weight
- maximum source patch scale

The current historical filename [`templates/configs/ablation_core24.yaml`](/Users/vineethgangaram/prism/templates/configs/ablation_core24.yaml) still says `core24`, but the active Cartesian product is 36 runs.

## Acceptance Criteria
The branch is considered internally consistent when:
1. no runtime path refers to any removed cross-series experiment code
2. no active runtime code expects rotation targets
3. model outputs expose the pair2-only head surface
4. baseline config launches the pair2-only training path
5. tests pass against the updated batch, model, and loss contracts
