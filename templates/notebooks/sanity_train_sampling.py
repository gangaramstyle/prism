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
    from prism_ssl.data import (
        ShardedScanDataset,
        collate_prism_batch,
        load_catalog,
        load_nifti_scan,
        sample_scan_candidates,
        world_points_to_voxel,
    )

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

    def window_to_rgb(slice_2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
        _ww = max(float(ww), 1e-6)
        _wmin = float(wc) - 0.5 * _ww
        _wmax = float(wc) + 0.5 * _ww
        _clipped = np.clip(slice_2d, _wmin, _wmax)
        _gray = np.clip((_clipped - _wmin) / max(_wmax - _wmin, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        return np.stack([_gray, _gray, _gray], axis=-1)

    def draw_square(img: np.ndarray, row: int, col: int, color: tuple[int, int, int], radius: int) -> None:
        _h, _w = img.shape[:2]
        _r0 = max(0, int(row) - int(radius))
        _r1 = min(_h, int(row) + int(radius) + 1)
        _c0 = max(0, int(col) - int(radius))
        _c1 = min(_w, int(col) + int(radius) + 1)
        img[_r0:_r1, _c0:_c1] = np.array(color, dtype=np.uint8)

    def vox_points_to_rc(points_vox: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        _pts = np.asarray(points_vox, dtype=np.int64)
        if axis == 2:
            return _pts[:, 0], _pts[:, 1]
        if axis == 1:
            return _pts[:, 0], _pts[:, 2]
        return _pts[:, 1], _pts[:, 2]


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
def _(batch, mo, records, sample_idx):
    sample_batch_index = int(sample_idx.value)
    sample_scan_id = str(batch["scan_id"][sample_batch_index])
    sample_series_id = str(batch["series_id"][sample_batch_index])

    sample_record = None
    for _rec in records:
        if str(_rec.scan_id) == sample_scan_id and str(_rec.series_id) == sample_series_id:
            sample_record = _rec
            break
    if sample_record is None:
        for _rec in records:
            if str(_rec.scan_id) == sample_scan_id:
                sample_record = _rec
                break

    mo.stop(
        sample_record is None,
        mo.callout(
            f"Could not map sampled pair back to records for scan_id={sample_scan_id}, series_id={sample_series_id}.",
            kind="danger",
        ),
    )
    return sample_batch_index, sample_record, sample_scan_id, sample_series_id


@app.cell
def _(batch, load_nifti_scan, np, resolved, sample_batch_index, sample_record, world_points_to_voxel):
    sample_scan, sample_nifti_path = load_nifti_scan(sample_record, base_patch_mm=float(resolved["patch_mm"]))
    scan_volume = np.asarray(sample_scan.data, dtype=np.float32)
    scan_shape_xyz = np.asarray(scan_volume.shape, dtype=np.int64)
    _clip_hi = np.maximum(scan_shape_xyz - 1, 0)

    _wc_a = float(batch["window_params_a"][sample_batch_index][0].item())
    _ww_a = float(batch["window_params_a"][sample_batch_index][1].item())
    _wc_b = float(batch["window_params_b"][sample_batch_index][0].item())
    _ww_b = float(batch["window_params_b"][sample_batch_index][1].item())
    scan_overlay_wc = 0.5 * (_wc_a + _wc_b)
    scan_overlay_ww = max(1e-3, 0.5 * (_ww_a + _ww_b))

    _center_a_world = batch["prism_center_pt_a"][sample_batch_index].numpy().astype(np.float32, copy=False)
    _center_b_world = batch["prism_center_pt_b"][sample_batch_index].numpy().astype(np.float32, copy=False)
    center_a_vox = np.rint(world_points_to_voxel(_center_a_world, sample_scan.affine)[0]).astype(np.int64)
    center_b_vox = np.rint(world_points_to_voxel(_center_b_world, sample_scan.affine)[0]).astype(np.int64)
    center_a_vox = np.clip(center_a_vox, 0, _clip_hi).astype(np.int64, copy=False)
    center_b_vox = np.clip(center_b_vox, 0, _clip_hi).astype(np.int64, copy=False)

    _rel_a = batch["positions_a"][sample_batch_index].numpy().astype(np.float32, copy=False)
    _rel_b = batch["positions_b"][sample_batch_index].numpy().astype(np.float32, copy=False)
    _patch_world_a = _center_a_world[np.newaxis, :] + _rel_a
    _patch_world_b = _center_b_world[np.newaxis, :] + _rel_b
    patch_centers_a_vox = np.rint(world_points_to_voxel(_patch_world_a, sample_scan.affine)).astype(np.int64)
    patch_centers_b_vox = np.rint(world_points_to_voxel(_patch_world_b, sample_scan.affine)).astype(np.int64)
    patch_centers_a_vox = np.clip(patch_centers_a_vox, 0, _clip_hi[np.newaxis, :]).astype(np.int64, copy=False)
    patch_centers_b_vox = np.clip(patch_centers_b_vox, 0, _clip_hi[np.newaxis, :]).astype(np.int64, copy=False)

    return (
        center_a_vox,
        center_b_vox,
        patch_centers_a_vox,
        patch_centers_b_vox,
        sample_nifti_path,
        scan_overlay_wc,
        scan_overlay_ww,
        scan_shape_xyz,
        scan_volume,
    )


@app.cell
def _(center_a_vox, center_b_vox, mo, patch_centers_a_vox, scan_shape_xyz):
    viewer_plane = mo.ui.dropdown(
        options=["axial (z)", "coronal (y)", "sagittal (x)"],
        value="axial (z)",
        label="slice plane",
    )
    if viewer_plane.value == "axial (z)":
        viewer_axis = 2
    elif viewer_plane.value == "coronal (y)":
        viewer_axis = 1
    else:
        viewer_axis = 0

    _axis_size = int(scan_shape_xyz[viewer_axis])
    _default_slice = int(round(0.5 * (int(center_a_vox[viewer_axis]) + int(center_b_vox[viewer_axis]))))
    _default_slice = min(max(_default_slice, 0), max(_axis_size - 1, 0))
    viewer_slice_idx = mo.ui.slider(0, max(_axis_size - 1, 0), value=_default_slice, step=1, label="slice index")
    viewer_slice_tol = mo.ui.slider(0, 8, value=1, step=1, label="slice tolerance (vox)")

    _n_patch = int(patch_centers_a_vox.shape[0])
    viewer_max_points = mo.ui.slider(
        0,
        max(_n_patch, 0),
        value=min(_n_patch, 128),
        step=1,
        label="max patch points per view",
    )
    viewer_center_radius = mo.ui.slider(1, 8, value=4, step=1, label="center marker radius")
    viewer_patch_radius = mo.ui.slider(1, 4, value=1, step=1, label="patch marker radius")

    mo.vstack(
        [
            mo.md("## Original Scan + A/B Centers"),
            mo.hstack([viewer_plane, viewer_slice_idx, viewer_slice_tol]),
            mo.hstack([viewer_max_points, viewer_center_radius, viewer_patch_radius]),
        ]
    )
    return (
        viewer_axis,
        viewer_center_radius,
        viewer_max_points,
        viewer_patch_radius,
        viewer_plane,
        viewer_slice_idx,
        viewer_slice_tol,
    )


@app.cell
def _(
    Image,
    alt,
    center_a_vox,
    center_b_vox,
    draw_square,
    mo,
    np,
    patch_centers_a_vox,
    patch_centers_b_vox,
    pl,
    sample_nifti_path,
    sample_scan_id,
    sample_series_id,
    scan_overlay_wc,
    scan_overlay_ww,
    scan_volume,
    viewer_axis,
    viewer_center_radius,
    viewer_max_points,
    viewer_patch_radius,
    viewer_plane,
    viewer_slice_idx,
    viewer_slice_tol,
    vox_points_to_rc,
    window_to_rgb,
):
    _axis = int(viewer_axis)
    _slice_idx = int(viewer_slice_idx.value)
    _slice_tol = int(viewer_slice_tol.value)
    if _axis == 2:
        _slice_2d = scan_volume[:, :, _slice_idx]
    elif _axis == 1:
        _slice_2d = scan_volume[:, _slice_idx, :]
    else:
        _slice_2d = scan_volume[_slice_idx, :, :]

    _overlay_rgb = window_to_rgb(_slice_2d, wc=float(scan_overlay_wc), ww=float(scan_overlay_ww))

    def _slice_subset(points_vox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _pts = np.asarray(points_vox, dtype=np.int64)
        if _pts.ndim != 2 or _pts.shape[1] != 3 or _pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        _dist = np.abs(_pts[:, _axis] - _slice_idx).astype(np.int64, copy=False)
        _keep = np.where(_dist <= _slice_tol)[0]
        if _keep.size == 0:
            return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        _order = np.argsort(_dist[_keep], kind="stable")
        _keep = _keep[_order]
        _max_points = int(viewer_max_points.value)
        if _max_points >= 0:
            _keep = _keep[:_max_points]
        return _pts[_keep], _dist[_keep]

    _patch_a, _dist_a = _slice_subset(patch_centers_a_vox)
    _patch_b, _dist_b = _slice_subset(patch_centers_b_vox)

    _rows_a, _cols_a = vox_points_to_rc(_patch_a, _axis)
    _rows_b, _cols_b = vox_points_to_rc(_patch_b, _axis)

    _patch_radius = int(viewer_patch_radius.value)
    for _r, _c in zip(_rows_a.tolist(), _cols_a.tolist()):
        draw_square(_overlay_rgb, int(_r), int(_c), (0, 255, 255), radius=_patch_radius)
    for _r, _c in zip(_rows_b.tolist(), _cols_b.tolist()):
        draw_square(_overlay_rgb, int(_r), int(_c), (255, 0, 255), radius=_patch_radius)

    _a_center_delta = abs(int(center_a_vox[_axis]) - _slice_idx)
    _b_center_delta = abs(int(center_b_vox[_axis]) - _slice_idx)
    _center_radius = int(viewer_center_radius.value)
    if _a_center_delta <= _slice_tol:
        _r_a, _c_a = vox_points_to_rc(np.asarray([center_a_vox], dtype=np.int64), _axis)
        draw_square(_overlay_rgb, int(_r_a[0]), int(_c_a[0]), (255, 80, 80), radius=_center_radius)
    if _b_center_delta <= _slice_tol:
        _r_b, _c_b = vox_points_to_rc(np.asarray([center_b_vox], dtype=np.int64), _axis)
        draw_square(_overlay_rgb, int(_r_b[0]), int(_c_b[0]), (80, 255, 80), radius=_center_radius)

    _points_rows = []
    for _view, _rows, _cols, _dist in (("A", _rows_a, _cols_a, _dist_a), ("B", _rows_b, _cols_b, _dist_b)):
        for _row, _col, _delta in zip(_rows.tolist(), _cols.tolist(), _dist.tolist()):
            _points_rows.append(
                {
                    "view": _view,
                    "row": int(_row),
                    "col": int(_col),
                    "slice_delta_vox": int(_delta),
                }
            )

    if len(_points_rows) > 0:
        _points_chart = (
            alt.Chart(_points_rows)
            .mark_circle(size=55, opacity=0.65)
            .encode(
                x=alt.X("col:Q", title="col"),
                y=alt.Y("row:Q", title="row"),
                color=alt.Color("view:N", title="view"),
                tooltip=[
                    alt.Tooltip("view:N", title="view"),
                    alt.Tooltip("row:Q", title="row"),
                    alt.Tooltip("col:Q", title="col"),
                    alt.Tooltip("slice_delta_vox:Q", title="slice_delta_vox"),
                ],
            )
            .properties(title="Patch centers near current slice", width=360, height=300)
        )
    else:
        _points_chart = mo.callout("No A/B patch centers fall inside current slice tolerance.", kind="warn")

    _summary = pl.DataFrame(
        [
            {
                "plane": str(viewer_plane.value),
                "slice_idx": int(_slice_idx),
                "slice_tol_vox": int(_slice_tol),
                "a_center_coord_axis": int(center_a_vox[_axis]),
                "b_center_coord_axis": int(center_b_vox[_axis]),
                "a_center_slice_delta": int(_a_center_delta),
                "b_center_slice_delta": int(_b_center_delta),
                "a_patch_points_shown": int(_patch_a.shape[0]),
                "b_patch_points_shown": int(_patch_b.shape[0]),
            }
        ]
    )
    mo.vstack(
        [
            mo.md(
                f"`scan_id={sample_scan_id}`  `series_id={sample_series_id}`  `nifti={sample_nifti_path}`"
            ),
            mo.md(
                "Legend: A center=red, B center=green, A patch centers=cyan, B patch centers=magenta."
            ),
            mo.hstack(
                [
                    mo.image(Image.fromarray(_overlay_rgb), width=760),
                    mo.vstack([mo.ui.table(_summary, label="overlay"), _points_chart]),
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
