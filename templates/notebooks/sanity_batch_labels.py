import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.data import ShardedScanDataset, collate_prism_batch, load_catalog, sample_scan_candidates

    def _patch_grid(patches: np.ndarray, max_patches: int = 24, cols: int = 6) -> np.ndarray:
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        n = min(max_patches, int(arr.shape[0]))
        if n == 0:
            return np.zeros((16, 16), dtype=np.uint8)
        arr = arr[:n]
        rows = (n + cols - 1) // cols
        pad = rows * cols - n
        if pad > 0:
            arr = np.concatenate([arr, np.zeros((pad, arr.shape[1], arr.shape[2]), dtype=arr.dtype)], axis=0)
        row_imgs = [np.concatenate(arr[r * cols : (r + 1) * cols], axis=1) for r in range(rows)]
        grid = np.concatenate(row_imgs, axis=0)
        return ((grid + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

    return (
        DataLoader,
        Path,
        ShardedScanDataset,
        alt,
        collate_prism_batch,
        load_catalog,
        mo,
        np,
        os,
        pl,
        sample_scan_candidates,
        torch,
        _patch_grid,
    )


@app.cell
def _(mo):
    from textwrap import dedent

    mo.md(
        dedent(
            """
# Prism SSL Batch/Label Sanity Notebook

Creates a real `ShardedScanDataset` + `DataLoader` batch and verifies:
- pair-view target correctness (`center_distance_mm`, `rotation_delta_deg`, `window_delta`)
- tensor shapes used by training
- same-series labels (`series_label`) used for SupCon positives
- replacement/broken-series counters emitted by the data pipeline
"""
        )
    )
    return


@app.cell
def _(Path, mo, os):
    default_catalog = os.environ.get(
        "CATALOG_PATH",
        str((Path(__file__).resolve().parents[1] / "data" / "pmbb_catalog.csv.gz")),
    )
    catalog_path = mo.ui.text(label="Catalog path", value=default_catalog)
    modality_csv = mo.ui.text(label="Modalities (comma-separated)", value="CT,MR")
    n_scans = mo.ui.number(label="n_scans", value=512, start=1, step=64)
    seed = mo.ui.number(label="seed", value=42, step=1)

    n_patches = mo.ui.number(label="n_patches", value=256, start=4, step=4)
    batch_size = mo.ui.number(label="batch_size", value=8, start=1, step=1)
    workers = mo.ui.number(label="workers", value=0, start=0, step=1)
    method = mo.ui.dropdown(options=["optimized_fused", "legacy_loop"], value="optimized_fused", label="method")
    warm_pool_size = mo.ui.number(label="warm_pool_size", value=16, start=1, step=1)
    visits_per_scan = mo.ui.number(label="visits_per_scan", value=100, start=1, step=1)
    max_prefetch = mo.ui.number(label="max_prefetch_replacements", value=2, start=1, step=1)

    use_scratch = mo.ui.checkbox(label="use local scratch dir", value=True)
    scratch_dir = mo.ui.text(
        label="scratch_dir",
        value=f"/tmp/{os.environ.get('USER', 'user')}/prism_ssl_notebook_scratch",
    )

    mo.vstack(
        [
            mo.md("## Dataset + Batch Controls"),
            mo.hstack([catalog_path]),
            mo.hstack([modality_csv, n_scans, seed]),
            mo.hstack([n_patches, batch_size, workers, method]),
            mo.hstack([warm_pool_size, visits_per_scan, max_prefetch]),
            mo.hstack([use_scratch, scratch_dir]),
        ]
    )
    return (
        batch_size,
        catalog_path,
        max_prefetch,
        method,
        modality_csv,
        n_patches,
        n_scans,
        scratch_dir,
        seed,
        use_scratch,
        visits_per_scan,
        warm_pool_size,
        workers,
    )


@app.cell
def _(
    DataLoader,
    ShardedScanDataset,
    batch_size,
    catalog_path,
    collate_prism_batch,
    load_catalog,
    max_prefetch,
    modality_csv,
    mo,
    n_patches,
    n_scans,
    sample_scan_candidates,
    seed,
    scratch_dir,
    use_scratch,
    visits_per_scan,
    warm_pool_size,
    workers,
    method,
):
    modalities = tuple(m.strip().upper() for m in str(modality_csv.value).split(",") if m.strip())
    mo.stop(len(modalities) == 0, mo.callout("Specify at least one modality", kind="warn"))

    df = load_catalog(str(catalog_path.value))
    records = sample_scan_candidates(
        df,
        n_scans=int(n_scans.value),
        seed=int(seed.value),
        modality_filter=modalities,
    )
    mo.stop(len(records) == 0, mo.callout("No records available after filtering", kind="danger"))

    ds = ShardedScanDataset(
        scan_records=records,
        n_patches=int(n_patches.value),
        base_patch_mm=16.0,
        method=str(method.value),
        warm_pool_size=int(warm_pool_size.value),
        visits_per_scan=int(visits_per_scan.value),
        seed=int(seed.value),
        max_prefetch_replacements=int(max_prefetch.value),
        strict_background_errors=False,
        broken_abort_ratio=0.99,
        broken_abort_min_attempts=100_000,
        max_broken_series_log=2000,
        broken_series_log_path="/tmp/prism_ssl_notebook_broken_series.jsonl",
        scratch_dir=str(scratch_dir.value) if bool(use_scratch.value) else None,
        pair_views=True,
    )

    loader_kwargs = {
        "batch_size": int(batch_size.value),
        "num_workers": int(workers.value),
        "collate_fn": collate_prism_batch,
        "pin_memory": False,
    }
    if int(workers.value) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(ds, **loader_kwargs)
    batch = next(iter(loader))
    return batch, records


@app.cell
def _(batch, mo, np, pl, torch):
    center_expected = torch.linalg.norm(batch["prism_center_pt_b"] - batch["prism_center_pt_a"], dim=1)
    rotation_expected = batch["rotation_degrees_b"] - batch["rotation_degrees_a"]
    window_expected = batch["window_params_b"] - batch["window_params_a"]

    center_err = float(torch.max(torch.abs(center_expected - batch["center_distance_mm"])).item())
    rot_err = float(torch.max(torch.abs(rotation_expected - batch["rotation_delta_deg"])).item())
    win_err = float(torch.max(torch.abs(window_expected - batch["window_delta"])).item())

    labels = batch["series_label"]
    same = labels[:, None] == labels[None, :]
    positives = torch.clamp(same.sum(dim=1) - 1, min=0)

    status = pl.DataFrame(
        [
            {
                "batch_size": int(batch["patches_a"].shape[0]),
                "n_patches": int(batch["patches_a"].shape[1]),
                "patch_hw": f"{int(batch['patches_a'].shape[2])}x{int(batch['patches_a'].shape[3])}",
                "center_distance_max_abs_err": center_err,
                "rotation_delta_max_abs_err": rot_err,
                "window_delta_max_abs_err": win_err,
                "supcon_positives_mean": float(positives.float().mean().item()),
                "supcon_positives_min": int(positives.min().item()),
                "supcon_positives_max": int(positives.max().item()),
                "replacement_completed_delta": int(batch["replacement_completed_count_delta"]),
                "replacement_failed_delta": int(batch["replacement_failed_count_delta"]),
                "replacement_wait_ms_delta": float(batch["replacement_wait_time_ms_delta"]),
                "attempted_series_delta": int(batch["attempted_series_delta"]),
                "broken_series_delta": int(batch["broken_series_delta"]),
            }
        ]
    )

    checks_ok = center_err < 1e-5 and rot_err < 1e-5 and win_err < 1e-5
    msg = (
        mo.callout("All pair-label consistency checks passed.", kind="success")
        if checks_ok
        else mo.callout(
            f"Label mismatch detected: center_err={center_err:.3e}, rot_err={rot_err:.3e}, win_err={win_err:.3e}",
            kind="danger",
        )
    )

    mo.vstack([mo.md("## Batch Consistency Checks"), msg, status])
    return center_expected, positives


@app.cell
def _(alt, batch, mo, np, pl):
    rot_norm = np.linalg.norm(batch["rotation_delta_deg"].numpy(), axis=1)
    tbl = pl.DataFrame(
        {
            "sample_idx": np.arange(batch["center_distance_mm"].shape[0], dtype=np.int64),
            "center_distance_mm": batch["center_distance_mm"].numpy(),
            "rotation_delta_norm_deg": rot_norm,
            "window_delta_wc": batch["window_delta"].numpy()[:, 0],
            "window_delta_ww": batch["window_delta"].numpy()[:, 1],
            "series_label": batch["series_label"].numpy(),
        }
    )

    chart = (
        alt.Chart(tbl.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("center_distance_mm:Q", bin=alt.Bin(maxbins=20), title="center_distance_mm"),
            y=alt.Y("count()", title="count"),
        )
        .properties(title="Distribution: center_distance_mm")
    )
    mo.vstack([mo.md("## Label Distributions"), chart, tbl.head(20)])
    return


@app.cell
def _(_patch_grid, batch, mo):
    patches_a = batch["patches_a"][0].numpy()
    patches_b = batch["patches_b"][0].numpy()
    grid_a = _patch_grid(patches_a, max_patches=24, cols=6)
    grid_b = _patch_grid(patches_b, max_patches=24, cols=6)

    mo.vstack(
        [
            mo.md("## First Sample Patch Preview"),
            mo.hstack(
                [
                    mo.vstack([mo.md("patches_a (first sample, first 24 patches)"), mo.image(src=grid_a, width=520)]),
                    mo.vstack([mo.md("patches_b (first sample, first 24 patches)"), mo.image(src=grid_b, width=520)]),
                ]
            ),
        ]
    )
    return


@app.cell
def _(batch, mo, pl):
    n = len(batch["scan_id"])
    preview = pl.DataFrame(
        {
            "scan_id": [str(x) for x in batch["scan_id"][: min(20, n)]],
            "series_id": [str(x) for x in batch["series_id"][: min(20, n)]],
            "series_label": batch["series_label"][: min(20, n)].numpy(),
            "center_distance_mm": batch["center_distance_mm"][: min(20, n)].numpy(),
        }
    )
    mo.vstack([mo.md("## Batch Identity Preview"), preview])
    return


if __name__ == "__main__":
    app.run()
