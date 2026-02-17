import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.data import load_catalog, load_nifti_scan, sample_scan_candidates

    def window_to_rgb(slice_2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
        ww_safe = max(float(ww), 1e-6)
        w_min = float(wc) - 0.5 * ww_safe
        w_max = float(wc) + 0.5 * ww_safe
        clipped = np.clip(slice_2d, w_min, w_max)
        gray = ((clipped - w_min) / max(w_max - w_min, 1e-6) * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def draw_cross(img: np.ndarray, row: int, col: int, color: tuple[int, int, int], radius: int = 3) -> None:
        h, w = img.shape[:2]
        if row < 0 or row >= h or col < 0 or col >= w:
            return
        r0 = max(0, row - radius)
        r1 = min(h, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(w, col + radius + 1)
        img[r0:r1, col] = np.array(color, dtype=np.uint8)
        img[row, c0:c1] = np.array(color, dtype=np.uint8)

    def overlay_slice(
        volume: np.ndarray,
        axis: int,
        slice_idx: int,
        center_vox: np.ndarray,
        patch_centers_vox: np.ndarray,
        wc: float,
        ww: float,
        max_points: int = 300,
    ) -> tuple[np.ndarray, int]:
        center = np.asarray(center_vox, dtype=np.int64)
        patches = np.asarray(patch_centers_vox, dtype=np.int64)

        if axis == 2:
            base = window_to_rgb(volume[:, :, slice_idx], wc, ww)
            center_rc = (int(center[0]), int(center[1]))
            mask = patches[:, 2] == int(slice_idx)
            patch_rc = [(int(p[0]), int(p[1])) for p in patches[mask][:max_points]]
        elif axis == 1:
            base = window_to_rgb(volume[:, slice_idx, :], wc, ww)
            center_rc = (int(center[0]), int(center[2]))
            mask = patches[:, 1] == int(slice_idx)
            patch_rc = [(int(p[0]), int(p[2])) for p in patches[mask][:max_points]]
        else:
            base = window_to_rgb(volume[slice_idx, :, :], wc, ww)
            center_rc = (int(center[1]), int(center[2]))
            mask = patches[:, 0] == int(slice_idx)
            patch_rc = [(int(p[1]), int(p[2])) for p in patches[mask][:max_points]]

        out = np.asarray(base, dtype=np.uint8).copy()
        for row, col in patch_rc:
            if 0 <= row < out.shape[0] and 0 <= col < out.shape[1]:
                out[row, col] = np.array([255, 220, 0], dtype=np.uint8)
        draw_cross(out, center_rc[0], center_rc[1], color=(255, 64, 64), radius=4)
        return out, int(np.count_nonzero(mask))

    def patch_grid(patches: np.ndarray, max_patches: int = 36, cols: int = 6) -> np.ndarray:
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        n = min(int(max_patches), arr.shape[0])
        if n == 0:
            return np.zeros((16, 16), dtype=np.uint8)
        arr = arr[:n]
        cols_eff = max(1, min(int(cols), n))
        rows = (n + cols_eff - 1) // cols_eff
        pad = rows * cols_eff - n
        if pad > 0:
            arr = np.concatenate([arr, np.zeros((pad, arr.shape[1], arr.shape[2]), dtype=arr.dtype)], axis=0)
        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.concatenate(arr[r * cols_eff : (r + 1) * cols_eff], axis=1))
        grid = np.concatenate(row_imgs, axis=0)
        grid = ((grid + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        return grid

    return Path, draw_cross, load_catalog, load_nifti_scan, mo, np, os, overlay_slice, patch_grid, pl, sample_scan_candidates, time, window_to_rgb


@app.cell
def _(mo):
    from textwrap import dedent

    mo.md(
        dedent(
            """
# Prism SSL Data Pipeline Sanity Notebook

Interactive debugging notebook for the exact `prism_ssl` data path:
- catalog loading + deterministic candidate sampling
- lazy NIfTI resolution/loading
- prism/patch sampling (random pipeline mode or manual overrides)
- augmentation/label outputs used by training
"""
        )
    )
    return


@app.cell
def _(Path, mo, os, time):
    default_catalog = os.environ.get(
        "CATALOG_PATH",
        str((Path(__file__).resolve().parents[1] / "data" / "pmbb_catalog.csv.gz")),
    )
    catalog_path = mo.ui.text(label="Catalog path", value=default_catalog)
    modality_csv = mo.ui.text(label="Modalities (comma-separated)", value="CT,MR")
    n_scans = mo.ui.number(label="Candidate scans to sample", value=5000, start=1, step=100)
    catalog_seed = mo.ui.number(label="Catalog sample seed", value=42, start=0, step=1)
    patch_mm = mo.ui.number(label="Base patch size (mm)", value=16.0, start=1.0, step=1.0)

    selection_mode = mo.ui.dropdown(options=["random", "index"], value="random", label="Scan selection mode")
    random_key = mo.ui.number(label="Random key (change for new random scan)", value=int(time.time()), step=1)
    scan_index = mo.ui.number(label="Scan index (used in index mode)", value=0, start=0, step=1)

    mo.vstack(
        [
            mo.md("## Catalog and Scan Selection"),
            mo.hstack([catalog_path]),
            mo.hstack([modality_csv, n_scans, catalog_seed, patch_mm]),
            mo.hstack([selection_mode, random_key, scan_index]),
        ]
    )
    return catalog_path, catalog_seed, modality_csv, n_scans, patch_mm, random_key, scan_index, selection_mode


@app.cell
def _(
    catalog_path,
    load_catalog,
    modality_csv,
    mo,
    n_scans,
    np,
    pl,
    random_key,
    sample_scan_candidates,
    catalog_seed,
    scan_index,
    selection_mode,
):
    modalities = tuple(m.strip().upper() for m in str(modality_csv.value).split(",") if m.strip())
    mo.stop(len(modalities) == 0, mo.callout("Specify at least one modality", kind="warn"))

    df = load_catalog(str(catalog_path.value))
    records = sample_scan_candidates(
        df,
        n_scans=int(n_scans.value),
        seed=int(catalog_seed.value),
        modality_filter=modalities,
    )
    mo.stop(len(records) == 0, mo.callout("No records available after filtering", kind="danger"))

    if str(selection_mode.value) == "random":
        pick_hash = int(np.uint64(abs(hash((int(catalog_seed.value), int(random_key.value), len(records))))))
        selected_index = int(pick_hash % len(records))
    else:
        selected_index = int(np.clip(int(scan_index.value), 0, len(records) - 1))
    selected_record = records[selected_index]

    record_info = pl.DataFrame(
        [
            {
                "catalog_rows": int(len(df)),
                "candidate_records": int(len(records)),
                "selected_index": int(selected_index),
                "scan_id": selected_record.scan_id,
                "series_id": selected_record.series_id,
                "modality": selected_record.modality,
                "series_path": selected_record.series_path,
            }
        ]
    )
    mo.vstack([mo.md("### Selected record"), record_info])
    return record_info, records, selected_record


@app.cell
def _(load_nifti_scan, mo, patch_mm, selected_record):
    try:
        scan, resolved_nifti_path = load_nifti_scan(selected_record, base_patch_mm=float(patch_mm.value))
    except Exception as exc:
        mo.stop(True, mo.callout(f"Failed to load selected scan: {exc}", kind="danger"))

    scan_shape = tuple(int(x) for x in scan.data.shape)
    spacing = tuple(float(x) for x in scan.spacing.tolist())
    patch_shape_vox = tuple(int(x) for x in scan.patch_shape_vox.tolist())
    mo.md(
        f"""
### Loaded Scan
- `resolved_nifti_path`: `{resolved_nifti_path}`
- `shape`: `{scan_shape}`
- `spacing_mm`: `{spacing}`
- `base_patch_mm`: `{float(patch_mm.value):.1f}`
- `patch_shape_vox` (before resize to 16x16): `{patch_shape_vox}`
- `robust_median/std`: `{scan.robust_median:.3f} / {scan.robust_std:.3f}`
"""
    )
    return resolved_nifti_path, scan


@app.cell
def _(mo, scan):
    n_patches = mo.ui.slider(label="n_patches", start=1, stop=2048, step=1, value=256)
    method = mo.ui.dropdown(options=["optimized_fused", "legacy_loop"], value="optimized_fused", label="Sampling method")
    sample_mode = mo.ui.dropdown(options=["pipeline-random", "manual-debug"], value="pipeline-random", label="Sampling mode")
    view_b_mode = mo.ui.dropdown(
        options=["independent", "same-as-view-a"],
        value="independent",
        label="View B mode (for pair labels)",
    )
    sample_seed = mo.ui.number(label="Sample seed", value=1234, step=1)

    manual_radius = mo.ui.slider(label="Manual sampling_radius_mm", start=1.0, stop=80.0, step=1.0, value=25.0)
    manual_rot_x = mo.ui.slider(label="Manual rotation X (deg)", start=-30, stop=30, step=1, value=0)
    manual_rot_y = mo.ui.slider(label="Manual rotation Y (deg)", start=-30, stop=30, step=1, value=0)
    manual_rot_z = mo.ui.slider(label="Manual rotation Z (deg)", start=-30, stop=30, step=1, value=0)
    manual_use_window = mo.ui.checkbox(label="Manual window (WC/WW)", value=False)
    manual_wc = mo.ui.number(label="Manual WC", value=40.0, step=1.0)
    manual_ww = mo.ui.number(label="Manual WW", value=400.0, start=1.0, step=1.0)

    sx, sy, sz = [int(x) for x in scan.data.shape]
    manual_center_x = mo.ui.number(label="Manual center X", value=sx // 2, start=0, stop=sx - 1, step=1)
    manual_center_y = mo.ui.number(label="Manual center Y", value=sy // 2, start=0, stop=sy - 1, step=1)
    manual_center_z = mo.ui.number(label="Manual center Z", value=sz // 2, start=0, stop=sz - 1, step=1)

    mo.vstack(
        [
            mo.md("## Prism Sampling Controls"),
            mo.hstack([n_patches, method, sample_mode, view_b_mode, sample_seed]),
            mo.hstack([manual_radius, manual_rot_x, manual_rot_y, manual_rot_z]),
            mo.hstack([manual_use_window, manual_wc, manual_ww]),
            mo.hstack([manual_center_x, manual_center_y, manual_center_z]),
        ]
    )
    return (
        manual_center_x,
        manual_center_y,
        manual_center_z,
        manual_radius,
        manual_rot_x,
        manual_rot_y,
        manual_rot_z,
        manual_use_window,
        manual_wc,
        manual_ww,
        method,
        n_patches,
        sample_mode,
        sample_seed,
        view_b_mode,
    )


@app.cell
def _(
    manual_center_x,
    manual_center_y,
    manual_center_z,
    manual_radius,
    manual_rot_x,
    manual_rot_y,
    manual_rot_z,
    manual_use_window,
    manual_wc,
    manual_ww,
    method,
    n_patches,
    np,
    sample_mode,
    sample_seed,
    scan,
    view_b_mode,
):
    common = {
        "n_patches": int(n_patches.value),
        "method": str(method.value),
    }

    if str(sample_mode.value) == "pipeline-random":
        view_a = scan.train_sample(seed=int(sample_seed.value) * 2, **common)
        view_b = scan.train_sample(seed=int(sample_seed.value) * 2 + 1, **common)
    else:
        manual_kwargs = {
            "sampling_radius_mm": float(manual_radius.value),
            "rotation_degrees": (
                float(manual_rot_x.value),
                float(manual_rot_y.value),
                float(manual_rot_z.value),
            ),
            "subset_center_vox": np.asarray(
                [int(manual_center_x.value), int(manual_center_y.value), int(manual_center_z.value)],
                dtype=np.int64,
            ),
        }
        if bool(manual_use_window.value):
            manual_kwargs["wc"] = float(manual_wc.value)
            manual_kwargs["ww"] = float(manual_ww.value)
        view_a = scan.train_sample(seed=int(sample_seed.value), **common, **manual_kwargs)
        if str(view_b_mode.value) == "same-as-view-a":
            view_b = scan.train_sample(seed=int(sample_seed.value), **common, **manual_kwargs)
        else:
            view_b = scan.train_sample(seed=int(sample_seed.value) + 1, **common)

    center_delta = np.asarray(view_b["prism_center_pt"] - view_a["prism_center_pt"], dtype=np.float32)
    center_distance = float(np.linalg.norm(center_delta))
    rotation_delta_deg = np.asarray(view_b["rotation_degrees"] - view_a["rotation_degrees"], dtype=np.float32)
    window_delta = np.asarray(
        [float(view_b["wc"] - view_a["wc"]), float(view_b["ww"] - view_a["ww"])],
        dtype=np.float32,
    )

    pair_labels = {
        "center_delta_mm": center_delta,
        "center_distance_mm": center_distance,
        "rotation_delta_deg": rotation_delta_deg,
        "window_delta": window_delta,
    }
    return pair_labels, view_a, view_b


@app.cell
def _(mo, pair_labels, pl, view_a, view_b):
    summary = pl.DataFrame(
        [
            {
                "view_a_method": str(view_a["method"]),
                "view_b_method": str(view_b["method"]),
                "view_a_wc": float(view_a["wc"]),
                "view_a_ww": float(view_a["ww"]),
                "view_b_wc": float(view_b["wc"]),
                "view_b_ww": float(view_b["ww"]),
                "view_a_sampling_radius_mm": float(view_a["sampling_radius_mm"]),
                "view_b_sampling_radius_mm": float(view_b["sampling_radius_mm"]),
                "center_distance_mm": float(pair_labels["center_distance_mm"]),
                "rotation_delta_x": float(pair_labels["rotation_delta_deg"][0]),
                "rotation_delta_y": float(pair_labels["rotation_delta_deg"][1]),
                "rotation_delta_z": float(pair_labels["rotation_delta_deg"][2]),
                "window_delta_wc": float(pair_labels["window_delta"][0]),
                "window_delta_ww": float(pair_labels["window_delta"][1]),
            }
        ]
    )
    mo.vstack([mo.md("## Final Labels / Targets (same format as training)"), summary])
    return summary


@app.cell
def _(
    mo,
    scan,
    view_a,
    view_b,
):
    _sx, _sy, _sz = [int(x) for x in scan.data.shape]
    a_center = [int(x) for x in view_a["prism_center_vox"]]
    b_center = [int(x) for x in view_b["prism_center_vox"]]

    a_axial_idx = mo.ui.slider(label="A axial slice (z)", start=0, stop=max(_sz - 1, 0), step=1, value=a_center[2])
    a_coronal_idx = mo.ui.slider(label="A coronal slice (y)", start=0, stop=max(_sy - 1, 0), step=1, value=a_center[1])
    a_sagittal_idx = mo.ui.slider(label="A sagittal slice (x)", start=0, stop=max(_sx - 1, 0), step=1, value=a_center[0])

    b_axial_idx = mo.ui.slider(label="B axial slice (z)", start=0, stop=max(_sz - 1, 0), step=1, value=b_center[2])
    b_coronal_idx = mo.ui.slider(label="B coronal slice (y)", start=0, stop=max(_sy - 1, 0), step=1, value=b_center[1])
    b_sagittal_idx = mo.ui.slider(label="B sagittal slice (x)", start=0, stop=max(_sx - 1, 0), step=1, value=b_center[0])
    max_points_overlay = mo.ui.slider(label="Max overlay points per slice", start=20, stop=800, step=20, value=200)

    mo.vstack(
        [
            mo.md("## Overlay Slice Controls"),
            mo.md(
                "Orientation grounding: "
                "Axial slider `z` moves Inferior -> Superior; "
                "Coronal slider `y` moves Posterior -> Anterior; "
                "Sagittal slider `x` moves Left -> Right."
            ),
            mo.hstack([a_axial_idx, a_coronal_idx, a_sagittal_idx]),
            mo.hstack([b_axial_idx, b_coronal_idx, b_sagittal_idx]),
            mo.hstack([max_points_overlay]),
        ]
    )
    return a_axial_idx, a_coronal_idx, a_sagittal_idx, b_axial_idx, b_coronal_idx, b_sagittal_idx, max_points_overlay


@app.cell
def _(
    overlay_slice,
    a_axial_idx,
    a_coronal_idx,
    a_sagittal_idx,
    b_axial_idx,
    b_coronal_idx,
    b_sagittal_idx,
    max_points_overlay,
    mo,
    scan,
    view_a,
    view_b,
):
    a_axial, a_axial_n = overlay_slice(
        scan.data,
        axis=2,
        slice_idx=int(a_axial_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=view_a["patch_centers_vox"],
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )
    a_coronal, a_coronal_n = overlay_slice(
        scan.data,
        axis=1,
        slice_idx=int(a_coronal_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=view_a["patch_centers_vox"],
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )
    a_sagittal, a_sagittal_n = overlay_slice(
        scan.data,
        axis=0,
        slice_idx=int(a_sagittal_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=view_a["patch_centers_vox"],
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )

    b_axial, b_axial_n = overlay_slice(
        scan.data,
        axis=2,
        slice_idx=int(b_axial_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=view_b["patch_centers_vox"],
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )
    b_coronal, b_coronal_n = overlay_slice(
        scan.data,
        axis=1,
        slice_idx=int(b_coronal_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=view_b["patch_centers_vox"],
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )
    b_sagittal, b_sagittal_n = overlay_slice(
        scan.data,
        axis=0,
        slice_idx=int(b_sagittal_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=view_b["patch_centers_vox"],
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )

    mo.vstack(
        [
            mo.md("## Scan Overlays (yellow=patch centers in slice, red=prism center)"),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"View A axial z={a_axial_idx.value} ({a_axial_n} pts)"), mo.image(src=a_axial, width=320)]),
                    mo.vstack([mo.md(f"View A coronal y={a_coronal_idx.value} ({a_coronal_n} pts)"), mo.image(src=a_coronal, width=320)]),
                    mo.vstack([mo.md(f"View A sagittal x={a_sagittal_idx.value} ({a_sagittal_n} pts)"), mo.image(src=a_sagittal, width=320)]),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"View B axial z={b_axial_idx.value} ({b_axial_n} pts)"), mo.image(src=b_axial, width=320)]),
                    mo.vstack([mo.md(f"View B coronal y={b_coronal_idx.value} ({b_coronal_n} pts)"), mo.image(src=b_coronal, width=320)]),
                    mo.vstack([mo.md(f"View B sagittal x={b_sagittal_idx.value} ({b_sagittal_n} pts)"), mo.image(src=b_sagittal, width=320)]),
                ]
            ),
        ]
    )
    return


@app.cell
def _(mo, patch_grid, view_a, view_b):
    n_a = int(view_a["normalized_patches"].shape[0])
    n_b = int(view_b["normalized_patches"].shape[0])
    show_a = min(36, max(1, n_a))
    show_b = min(36, max(1, n_b))
    grid_a = patch_grid(view_a["normalized_patches"], max_patches=show_a, cols=min(6, show_a))
    grid_b = patch_grid(view_b["normalized_patches"], max_patches=show_b, cols=min(6, show_b))
    mo.vstack(
        [
            mo.md("## Normalized Patch Grids (preview, each patch resized to 16x16)"),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"View A (showing {show_a}/{n_a} patches)"), mo.image(src=grid_a, width=520)]),
                    mo.vstack([mo.md(f"View B (showing {show_b}/{n_b} patches)"), mo.image(src=grid_b, width=520)]),
                ]
            ),
        ]
    )
    return


@app.cell
def _(mo, np, pl, view_a):
    centers = np.asarray(view_a["patch_centers_vox"], dtype=np.int64)
    rel = np.asarray(view_a["relative_patch_centers_pt"], dtype=np.float32)
    n_show = min(25, centers.shape[0])
    preview = pl.DataFrame(
        {
            "x_vox": centers[:n_show, 0],
            "y_vox": centers[:n_show, 1],
            "z_vox": centers[:n_show, 2],
            "rel_x_mm": rel[:n_show, 0],
            "rel_y_mm": rel[:n_show, 1],
            "rel_z_mm": rel[:n_show, 2],
        }
    )
    mo.vstack([mo.md("## Patch-center Preview (View A, first 25)"), preview])
    return


if __name__ == "__main__":
    app.run()
