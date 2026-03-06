import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


with app.setup:
    import os
    import sys
    import time
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from PIL import Image
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.config import load_run_config
    from prism_ssl.data import ShardedScanDataset, collate_prism_batch, load_catalog, sample_scan_candidates

    def patch_grid(patches: np.ndarray, max_patches: int, cols: int) -> np.ndarray:
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        n = min(int(max_patches), int(arr.shape[0]))
        if n <= 0:
            return np.zeros((16, 16), dtype=np.uint8)

        arr = arr[:n].copy()
        cols_eff = max(1, min(int(cols), n))
        rows = (n + cols_eff - 1) // cols_eff
        pad = rows * cols_eff - n
        if pad > 0:
            arr = np.concatenate([arr, np.zeros((pad, arr.shape[1], arr.shape[2]), dtype=arr.dtype)], axis=0)

        tiles = [np.concatenate(arr[r * cols_eff : (r + 1) * cols_eff], axis=1) for r in range(rows)]
        grid = np.concatenate(tiles, axis=0)

        # Force a fixed display range across samples for easier visual comparison.
        grid[0, 0] = -1.0
        grid[0, 1] = 1.0
        return np.clip((grid + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)


@app.cell
def _(mo):
    mo.md(
        """
# Training Sampler Sanity Notebook

This notebook builds the exact same `ShardedScanDataset` + `DataLoader` path used by training, then inspects:
- sampled A/B tensors and shapes,
- center-delta targets (R/A/S sign balance),
- window deltas and replacement counters,
- patch previews for selected samples.

Use this to debug whether the model should be able to learn from your current sampling settings.
"""
    )
    return


@app.cell
def _(Path, mo):
    default_cfg = str((Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"))
    config_path = mo.ui.text(label="Config path", value=default_cfg)
    catalog_path_override = mo.ui.text(label="Catalog path override (optional)", value="")
    seed = mo.ui.number(label="Seed", value=42, step=1)

    n_scans = mo.ui.number(label="n_scans", value=40, start=1, step=1)
    warm_pool_size = mo.ui.number(label="warm_pool_size", value=40, start=1, step=1)
    visits_per_scan = mo.ui.number(label="visits_per_scan", value=10_000_000, start=1, step=1000)

    batch_size = mo.ui.number(label="batch_size", value=8, start=1, step=1)
    n_patches = mo.ui.number(label="n_patches", value=256, start=1, step=1)
    workers = mo.ui.number(label="workers", value=0, start=0, step=1)

    mo.vstack(
        [
            mo.md("## Controls"),
            mo.hstack([config_path, catalog_path_override]),
            mo.hstack([seed, n_scans, warm_pool_size, visits_per_scan]),
            mo.hstack([batch_size, n_patches, workers]),
        ]
    )
    return (
        batch_size,
        catalog_path_override,
        config_path,
        n_patches,
        n_scans,
        seed,
        visits_per_scan,
        warm_pool_size,
        workers,
    )


@app.cell
def _(
    batch_size,
    catalog_path_override,
    config_path,
    load_run_config,
    mo,
    n_patches,
    n_scans,
    pl,
    seed,
    visits_per_scan,
    warm_pool_size,
    workers,
):
    cfg = load_run_config(str(config_path.value))
    catalog_path = str(catalog_path_override.value).strip() or str(cfg.data.catalog_path)

    resolved = {
        "catalog_path": catalog_path,
        "modality_filter": tuple(cfg.data.modality_filter),
        "patch_mm": float(cfg.data.patch_mm),
        "n_scans": int(n_scans.value),
        "n_patches": int(n_patches.value),
        "warm_pool_size": int(warm_pool_size.value),
        "visits_per_scan": int(visits_per_scan.value),
        "workers": int(workers.value),
        "batch_size": int(batch_size.value),
        "seed": int(seed.value),
        "max_prefetch_replacements": int(cfg.data.max_prefetch_replacements),
        "strict_background_errors": bool(cfg.data.strict_background_errors),
        "broken_abort_ratio": float(cfg.data.broken_abort_ratio),
        "broken_abort_min_attempts": int(cfg.data.broken_abort_min_attempts),
        "max_broken_series_log": int(cfg.data.max_broken_series_log),
        "use_local_scratch": bool(cfg.data.use_local_scratch),
    }

    cfg_tbl = pl.DataFrame(
        [{"field": k, "value": str(v)} for k, v in resolved.items()]
    )
    mo.vstack([mo.md("## Resolved Settings"), mo.ui.table(cfg_tbl, label="resolved")])
    return cfg, resolved


@app.cell
def _(Path, ShardedScanDataset, DataLoader, collate_prism_batch, load_catalog, mo, os, resolved, sample_scan_candidates, time):
    t0 = time.perf_counter()
    df = load_catalog(str(resolved["catalog_path"]))
    records = sample_scan_candidates(
        df,
        n_scans=int(resolved["n_scans"]),
        seed=int(resolved["seed"]),
        modality_filter=tuple(resolved["modality_filter"]),
    )
    mo.stop(len(records) == 0, mo.callout("No scan records found with current settings.", kind="danger"))

    scratch_dir = (
        f"/tmp/{os.environ.get('USER', 'user')}/prism_ssl_notebook_scratch"
        if bool(resolved["use_local_scratch"])
        else None
    )
    broken_log_path = str(Path("/tmp") / f"prism_ssl_notebook_broken_{os.getpid()}.jsonl")

    ds = ShardedScanDataset(
        scan_records=records,
        n_patches=int(resolved["n_patches"]),
        base_patch_mm=float(resolved["patch_mm"]),
        warm_pool_size=int(resolved["warm_pool_size"]),
        visits_per_scan=int(resolved["visits_per_scan"]),
        seed=int(resolved["seed"]),
        max_prefetch_replacements=int(resolved["max_prefetch_replacements"]),
        strict_background_errors=bool(resolved["strict_background_errors"]),
        broken_abort_ratio=float(resolved["broken_abort_ratio"]),
        broken_abort_min_attempts=int(resolved["broken_abort_min_attempts"]),
        max_broken_series_log=int(resolved["max_broken_series_log"]),
        broken_series_log_path=broken_log_path,
        scratch_dir=scratch_dir,
        pair_views=True,
    )

    loader_kwargs = {
        "batch_size": int(resolved["batch_size"]),
        "num_workers": int(resolved["workers"]),
        "pin_memory": False,
        "collate_fn": collate_prism_batch,
    }
    if int(resolved["workers"]) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(ds, **loader_kwargs)
    batch = next(iter(loader))
    fetch_ms = (time.perf_counter() - t0) * 1000.0
    return batch, fetch_ms, records


@app.cell
def _(batch, fetch_ms, mo, np, pl, records):
    deltas = batch["center_delta_mm"].numpy()
    targets = (deltas > 0).astype(np.int64)
    _center_dist = batch["center_distance_mm"].numpy()
    window_delta = batch["window_delta"].numpy()
    positives = targets.mean(axis=0)

    summary = pl.DataFrame(
        [
            {
                "records_sampled": int(len(records)),
                "batch_size": int(batch["patches_a"].shape[0]),
                "n_patches": int(batch["patches_a"].shape[1]),
                "patch_shape": f"{int(batch['patches_a'].shape[2])}x{int(batch['patches_a'].shape[3])}x{int(batch['patches_a'].shape[4])}",
                "fetch_ms": float(fetch_ms),
                "center_distance_mean_mm": float(_center_dist.mean()),
                "center_distance_p95_mm": float(np.percentile(_center_dist, 95.0)),
                "target_pos_frac_R": float(positives[0]),
                "target_pos_frac_A": float(positives[1]),
                "target_pos_frac_S": float(positives[2]),
                "window_delta_abs_max": float(np.abs(window_delta).max()),
                "replacement_completed_delta": int(batch["replacement_completed_count_delta"]),
                "replacement_failed_delta": int(batch["replacement_failed_count_delta"]),
                "replacement_wait_ms_delta": float(batch["replacement_wait_time_ms_delta"]),
                "attempted_series_delta": int(batch["attempted_series_delta"]),
                "broken_series_delta": int(batch["broken_series_delta"]),
            }
        ]
    )

    callout = (
        mo.callout("Window deltas are zero/nearly-zero between A and B (expected with shared wc/ww).", kind="success")
        if float(np.abs(window_delta).max()) < 1e-5
        else mo.callout("Window deltas are non-trivial between A and B.", kind="warn")
    )
    mo.vstack([mo.md("## Batch Summary"), callout, mo.ui.table(summary, label="batch")])
    return deltas, targets


@app.cell
def _(batch, deltas, mo, np, pl, targets):
    rows = pl.DataFrame(
        {
            "idx": np.arange(batch["center_delta_mm"].shape[0], dtype=np.int64),
            "scan_id": batch["scan_id"],
            "series_id": batch["series_id"],
            "delta_R_mm": deltas[:, 0],
            "delta_A_mm": deltas[:, 1],
            "delta_S_mm": deltas[:, 2],
            "target_R": targets[:, 0],
            "target_A": targets[:, 1],
            "target_S": targets[:, 2],
            "center_distance_mm": batch["center_distance_mm"].numpy(),
        }
    )
    mo.vstack([mo.md("## Per-Sample Target Table"), mo.ui.table(rows, label="targets")])
    return rows


@app.cell
def _(alt, mo, rows):
    chart = (
        alt.Chart(rows.to_dicts())
        .mark_circle(size=80)
        .encode(
            x=alt.X("delta_R_mm:Q", title="Delta R (mm)"),
            y=alt.Y("delta_A_mm:Q", title="Delta A (mm)"),
            color=alt.Color("target_S:N", title="Target S"),
            tooltip=[
                alt.Tooltip("idx:Q", title="idx"),
                alt.Tooltip("delta_R_mm:Q", title="delta_R_mm"),
                alt.Tooltip("delta_A_mm:Q", title="delta_A_mm"),
                alt.Tooltip("delta_S_mm:Q", title="delta_S_mm"),
                alt.Tooltip("center_distance_mm:Q", title="center_distance_mm"),
            ],
        )
        .properties(title="Center Delta Scatter", width=420, height=360)
    )
    mo.vstack([mo.md("## Delta Scatter"), chart])
    return


@app.cell
def _(batch, mo):
    max_idx = max(int(batch["patches_a"].shape[0]) - 1, 0)
    sample_idx = mo.ui.slider(0, max_idx, value=0, label="sample_idx")
    preview_patches = mo.ui.slider(4, min(128, int(batch["patches_a"].shape[1])), value=32, step=4, label="preview_patches")
    preview_cols = mo.ui.slider(2, 16, value=8, step=1, label="preview_cols")
    mo.hstack([sample_idx, preview_patches, preview_cols])
    return preview_cols, preview_patches, sample_idx


@app.cell
def _(Image, batch, mo, np, patch_grid, preview_cols, preview_patches, sample_idx):
    i = int(sample_idx.value)
    pa = batch["patches_a"][i].numpy()
    pb = batch["patches_b"][i].numpy()
    grid_a = patch_grid(pa, max_patches=int(preview_patches.value), cols=int(preview_cols.value))
    grid_b = patch_grid(pb, max_patches=int(preview_patches.value), cols=int(preview_cols.value))

    center_delta = batch["center_delta_mm"][i].numpy()
    _center_dist = float(batch["center_distance_mm"][i].item())
    signs = (center_delta > 0).astype(np.int64)
    header = (
        f"sample={i} | delta_mm=({center_delta[0]:.2f}, {center_delta[1]:.2f}, {center_delta[2]:.2f}) "
        f"| target=({int(signs[0])}, {int(signs[1])}, {int(signs[2])}) | dist={_center_dist:.2f}mm"
    )

    mo.vstack(
        [
            mo.md("## Patch Preview"),
            mo.md(f"`{header}`"),
            mo.hstack(
                [
                    mo.vstack([mo.md("View A"), mo.image(Image.fromarray(grid_a), width=520)]),
                    mo.vstack([mo.md("View B"), mo.image(Image.fromarray(grid_b), width=520)]),
                ]
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Cluster Overfit Run (40 fixed scans)

Single-line launch example:

```bash
cd ~/prism-ssl/templates && export PATH="$HOME/.local/bin:$PATH" && jid=$(N_SCANS=40 WARM_POOL_SIZE=40 VISITS_PER_SCAN=10000000 BATCH_SIZE=64 N_PATCHES=256 WORKERS=16 sbatch scripts/job_train_prism_ssl.sh | awk '{print $4}') && echo "JOB=$jid"
```
"""
    )
    return


if __name__ == "__main__":
    app.run()
