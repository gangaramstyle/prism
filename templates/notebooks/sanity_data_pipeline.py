import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


with app.setup:
    import os
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy import ndimage

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.data import infer_scan_geometry, load_catalog, load_nifti_scan, sample_scan_candidates

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
        max_points: int = 400,
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

    def patch_grid(patches: np.ndarray, max_patches: int, cols: int) -> np.ndarray:
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        n = min(int(max_patches), int(arr.shape[0]))
        if n <= 0:
            return np.zeros((16, 16), dtype=np.uint8)

        arr = arr[:n]
        cols_eff = max(1, min(int(cols), n))
        rows = (n + cols_eff - 1) // cols_eff
        pad = rows * cols_eff - n
        if pad > 0:
            arr = np.concatenate([arr, np.zeros((pad, arr.shape[1], arr.shape[2]), dtype=arr.dtype)], axis=0)

        row_tiles = []
        for row in range(rows):
            row_tiles.append(np.concatenate(arr[row * cols_eff : (row + 1) * cols_eff], axis=1))
        grid = np.concatenate(row_tiles, axis=0)
        return ((grid + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

    def rotate_volume_about_center(volume: np.ndarray, center_vox: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        center = np.asarray(center_vox, dtype=np.float64)
        r = np.asarray(rotation_matrix, dtype=np.float64)
        r_inv = np.linalg.inv(r)
        offset = center - r_inv @ center
        return ndimage.affine_transform(
            volume,
            r_inv,
            offset=offset,
            order=1,
            mode="nearest",
        ).astype(np.float32, copy=False)

    def rotated_patch_centers_vox(
        relative_patch_centers_pt_rotated: np.ndarray,
        prism_center_vox: np.ndarray,
        spacing_mm: np.ndarray,
        shape: tuple[int, int, int],
    ) -> np.ndarray:
        spacing = np.asarray(spacing_mm, dtype=np.float32)
        center_vox = np.asarray(prism_center_vox, dtype=np.float32)
        center_pt = center_vox * spacing
        centers_pt = center_pt[np.newaxis, :] + np.asarray(relative_patch_centers_pt_rotated, dtype=np.float32)
        centers_vox = np.rint(centers_pt / spacing).astype(np.int64)
        upper = np.asarray(shape, dtype=np.int64) - 1
        return np.clip(centers_vox, 0, upper)

    def tensorize_like_training(view: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        patches = np.asarray(view["normalized_patches"], dtype=np.float32)
        if patches.ndim == 2:
            patches = patches[np.newaxis, ...]
        patches = patches[..., np.newaxis]
        positions = np.atleast_2d(np.asarray(view["relative_patch_centers_pt"], dtype=np.float32))
        return patches, positions


@app.cell
def _():
    from textwrap import dedent

    mo.md(
        dedent(
            """
# Prism SSL Data Pipeline Sanity Notebook

Goal: verify data correctness with the **same sampling path used in training**, but with manual controls for debugging.

- Step 1. Load scan + catalog metadata + inferred native orientation
- Step 2. Set prism center/rotation/window for views A/B and inspect native + rotated overlays
- Step 3. Inspect sampled patches and coordinate labels
- Step 4. Inspect exact model input tensors (patches + relative positions)
"""
        )
    )
    return


@app.cell
def _():
    default_catalog = os.environ.get(
        "CATALOG_PATH",
        str((Path(__file__).resolve().parents[1] / "data" / "pmbb_catalog.csv.gz")),
    )
    catalog_path = mo.ui.text(label="Catalog path", value=default_catalog)
    modality_csv = mo.ui.text(label="Modalities (comma-separated)", value="CT,MR")
    n_scans = mo.ui.number(label="Candidate scans to sample", value=5000, start=1, step=100)
    catalog_seed = mo.ui.number(label="Catalog sample seed", value=42, start=0, step=1)
    patch_mm = mo.ui.number(label="Base patch size (mm)", value=64.0, start=4.0, step=4.0)

    selection_mode = mo.ui.dropdown(options=["random", "index"], value="random", label="Series selection mode")
    random_key = mo.ui.number(label="Random key (change to pick a new random series)", value=int(time.time()), step=1)
    scan_index = mo.ui.number(label="Series index (used in index mode)", value=0, start=0, step=1)

    mo.vstack(
        [
            mo.md("## Step 1: Load Scan"),
            mo.hstack([catalog_path]),
            mo.hstack([modality_csv, n_scans, catalog_seed, patch_mm]),
            mo.hstack([selection_mode, random_key, scan_index]),
        ]
    )
    return catalog_path, modality_csv, n_scans, catalog_seed, patch_mm, selection_mode, random_key, scan_index


@app.cell
def _(catalog_path, catalog_seed, modality_csv, n_scans, random_key, scan_index, selection_mode):
    modalities = tuple(m.strip().upper() for m in str(modality_csv.value).split(",") if m.strip())
    mo.stop(len(modalities) == 0, mo.callout("Specify at least one modality", kind="warn"))

    df = load_catalog(str(catalog_path.value))
    records = sample_scan_candidates(
        df,
        n_scans=int(n_scans.value),
        seed=int(catalog_seed.value),
        modality_filter=modalities,
    )
    mo.stop(len(records) == 0, mo.callout("No candidate records available after filtering", kind="danger"))

    if str(selection_mode.value) == "random":
        pick_hash = int(np.uint64(abs(hash((int(catalog_seed.value), int(random_key.value), len(records))))))
        selected_index = int(pick_hash % len(records))
    else:
        selected_index = int(np.clip(int(scan_index.value), 0, len(records) - 1))

    selected_record = records[selected_index]
    selected_row_df = df.filter(pl.col("series_path") == selected_record.series_path).head(1)
    selected_row = selected_row_df.to_dicts()[0] if len(selected_row_df) > 0 else {}

    overview = pl.DataFrame(
        [
            {
                "catalog_rows": int(len(df)),
                "candidate_records": int(len(records)),
                "selected_index": int(selected_index),
                "scan_id": selected_record.scan_id,
                "series_id": selected_record.series_id,
                "modality": selected_record.modality,
            }
        ]
    )
    mo.vstack([mo.md("### Candidate Selection"), overview])
    return selected_record, selected_row


@app.cell
def _(patch_mm, selected_record):
    try:
        scan, resolved_nifti_path = load_nifti_scan(selected_record, base_patch_mm=float(patch_mm.value))
    except Exception as exc:
        mo.stop(True, mo.callout(f"Failed to load selected scan: {exc}", kind="danger"))

    geometry = infer_scan_geometry(scan.data.shape, scan.spacing)
    return geometry, resolved_nifti_path, scan


@app.cell
def _(geometry, patch_mm, resolved_nifti_path, scan, selected_row):
    scan_shape = tuple(int(x) for x in scan.data.shape)
    spacing = tuple(float(x) for x in scan.spacing.tolist())
    patch_shape_vox = tuple(int(x) for x in scan.patch_shape_vox.tolist())

    preferred_cols = [
        "pmbb_id",
        "modality",
        "series_description",
        "body_part",
        "manufacturer",
        "series_size_mb",
        "series_path",
    ]
    meta_payload = {k: selected_row.get(k, "") for k in preferred_cols if k in selected_row}
    metadata_table = pl.DataFrame([meta_payload]) if meta_payload else pl.DataFrame([{"series_path": "metadata not found"}])

    mo.vstack(
        [
            mo.md("### Step 1 Output"),
            mo.md(
                f"""
- `resolved_nifti_path`: `{resolved_nifti_path}`
- `shape`: `{scan_shape}`
- `spacing_mm`: `{spacing}`
- `base_patch_mm`: `{float(patch_mm.value):.1f}`
- `patch_shape_vox` (before resize to 16x16): `{patch_shape_vox}`
- `robust_median/std`: `{scan.robust_median:.3f} / {scan.robust_std:.3f}`
- `native_plane`: `{geometry.acquisition_plane}`
- `thin_axis`: `{geometry.thin_axis_name}` (index `{geometry.thin_axis}`)
- `baseline_rotation_hint_deg`: `{geometry.baseline_rotation_degrees}`
- `orientation_inference`: `{geometry.inference_reason}`
"""
            ),
            mo.md("#### Selected Series Metadata"),
            metadata_table,
        ]
    )
    return


@app.cell
def _(geometry, scan):
    n_patches = mo.ui.slider(label="n_patches", start=1, stop=2048, step=1, value=256)
    method = mo.ui.dropdown(options=["optimized_fused", "legacy_loop"], value="optimized_fused", label="Sampling method")
    sample_mode = mo.ui.dropdown(options=["pipeline-random", "manual-debug"], value="manual-debug", label="Sampling mode")
    sample_seed = mo.ui.number(label="Sample seed", value=1234, step=1)
    lock_b_to_a = mo.ui.checkbox(label="Manual mode: make View B identical to View A", value=False)

    default_center = [int(v) // 2 for v in scan.data.shape]
    default_wc = float(scan.robust_median)
    default_ww = float(max(2.0 * scan.robust_std, 1.0))
    rot_base = tuple(float(v) for v in geometry.baseline_rotation_degrees)

    sampling_radius_mm = mo.ui.slider(label="Manual sampling radius (mm)", start=1.0, stop=80.0, step=1.0, value=25.0)

    a_center_x = mo.ui.number(label="A center X", value=default_center[0], start=0, stop=int(scan.data.shape[0] - 1), step=1)
    a_center_y = mo.ui.number(label="A center Y", value=default_center[1], start=0, stop=int(scan.data.shape[1] - 1), step=1)
    a_center_z = mo.ui.number(label="A center Z", value=default_center[2], start=0, stop=int(scan.data.shape[2] - 1), step=1)
    a_rot_x = mo.ui.slider(label="A rot X (deg)", start=-45, stop=45, step=1, value=int(rot_base[0]))
    a_rot_y = mo.ui.slider(label="A rot Y (deg)", start=-45, stop=45, step=1, value=int(rot_base[1]))
    a_rot_z = mo.ui.slider(label="A rot Z (deg)", start=-45, stop=45, step=1, value=int(rot_base[2]))
    a_wc = mo.ui.number(label="A WC", value=default_wc, step=1.0)
    a_ww = mo.ui.number(label="A WW", value=default_ww, start=1.0, step=1.0)

    b_center_x = mo.ui.number(label="B center X", value=default_center[0], start=0, stop=int(scan.data.shape[0] - 1), step=1)
    b_center_y = mo.ui.number(label="B center Y", value=default_center[1], start=0, stop=int(scan.data.shape[1] - 1), step=1)
    b_center_z = mo.ui.number(label="B center Z", value=default_center[2], start=0, stop=int(scan.data.shape[2] - 1), step=1)
    b_rot_x = mo.ui.slider(label="B rot X (deg)", start=-45, stop=45, step=1, value=int(rot_base[0]))
    b_rot_y = mo.ui.slider(label="B rot Y (deg)", start=-45, stop=45, step=1, value=int(rot_base[1]))
    b_rot_z = mo.ui.slider(label="B rot Z (deg)", start=-45, stop=45, step=1, value=int(rot_base[2]))
    b_wc = mo.ui.number(label="B WC", value=default_wc, step=1.0)
    b_ww = mo.ui.number(label="B WW", value=default_ww, start=1.0, step=1.0)

    mo.vstack(
        [
            mo.md("## Step 2: Select Prism Center / Rotation / Window"),
            mo.hstack([n_patches, method, sample_mode, sample_seed, lock_b_to_a]),
            mo.hstack([sampling_radius_mm]),
            mo.md("### View A manual controls"),
            mo.hstack([a_center_x, a_center_y, a_center_z, a_rot_x, a_rot_y, a_rot_z, a_wc, a_ww]),
            mo.md("### View B manual controls"),
            mo.hstack([b_center_x, b_center_y, b_center_z, b_rot_x, b_rot_y, b_rot_z, b_wc, b_ww]),
        ]
    )

    return (
        n_patches,
        method,
        sample_mode,
        sample_seed,
        lock_b_to_a,
        sampling_radius_mm,
        a_center_x,
        a_center_y,
        a_center_z,
        a_rot_x,
        a_rot_y,
        a_rot_z,
        a_wc,
        a_ww,
        b_center_x,
        b_center_y,
        b_center_z,
        b_rot_x,
        b_rot_y,
        b_rot_z,
        b_wc,
        b_ww,
    )


@app.cell
def _(
    a_center_x,
    a_center_y,
    a_center_z,
    a_rot_x,
    a_rot_y,
    a_rot_z,
    a_wc,
    a_ww,
    b_center_x,
    b_center_y,
    b_center_z,
    b_rot_x,
    b_rot_y,
    b_rot_z,
    b_wc,
    b_ww,
    lock_b_to_a,
    method,
    n_patches,
    sample_mode,
    sample_seed,
    sampling_radius_mm,
    scan,
):
    common = {
        "n_patches": int(n_patches.value),
        "method": str(method.value),
    }

    if str(sample_mode.value) == "pipeline-random":
        view_a = scan.train_sample(seed=int(sample_seed.value) * 2, **common)
        view_b = scan.train_sample(seed=int(sample_seed.value) * 2 + 1, **common)
    else:
        a_kwargs = {
            "sampling_radius_mm": float(sampling_radius_mm.value),
            "rotation_degrees": (float(a_rot_x.value), float(a_rot_y.value), float(a_rot_z.value)),
            "subset_center_vox": np.asarray(
                [int(a_center_x.value), int(a_center_y.value), int(a_center_z.value)],
                dtype=np.int64,
            ),
            "wc": float(a_wc.value),
            "ww": float(a_ww.value),
        }
        view_a = scan.train_sample(seed=int(sample_seed.value), **common, **a_kwargs)

        if bool(lock_b_to_a.value):
            view_b = scan.train_sample(seed=int(sample_seed.value), **common, **a_kwargs)
        else:
            b_kwargs = {
                "sampling_radius_mm": float(sampling_radius_mm.value),
                "rotation_degrees": (float(b_rot_x.value), float(b_rot_y.value), float(b_rot_z.value)),
                "subset_center_vox": np.asarray(
                    [int(b_center_x.value), int(b_center_y.value), int(b_center_z.value)],
                    dtype=np.int64,
                ),
                "wc": float(b_wc.value),
                "ww": float(b_ww.value),
            }
            view_b = scan.train_sample(seed=int(sample_seed.value) + 1, **common, **b_kwargs)

    center_delta = np.asarray(view_b["prism_center_pt"] - view_a["prism_center_pt"], dtype=np.float32)
    pair_labels = {
        "center_delta_mm": center_delta,
        "center_distance_mm": float(np.linalg.norm(center_delta)),
        "rotation_delta_deg": np.asarray(view_b["rotation_degrees"] - view_a["rotation_degrees"], dtype=np.float32),
        "window_delta": np.asarray(
            [float(view_b["wc"] - view_a["wc"]), float(view_b["ww"] - view_a["ww"])],
            dtype=np.float32,
        ),
    }
    return pair_labels, view_a, view_b


@app.cell
def _(pair_labels, view_a, view_b):
    summary = pl.DataFrame(
        [
            {
                "a_center_vox": tuple(int(v) for v in np.asarray(view_a["prism_center_vox"]).tolist()),
                "a_center_mm": tuple(float(v) for v in np.asarray(view_a["prism_center_pt"]).tolist()),
                "a_rotation_deg": tuple(float(v) for v in np.asarray(view_a["rotation_degrees"]).tolist()),
                "a_window_wc_ww": (float(view_a["wc"]), float(view_a["ww"])),
                "b_center_vox": tuple(int(v) for v in np.asarray(view_b["prism_center_vox"]).tolist()),
                "b_center_mm": tuple(float(v) for v in np.asarray(view_b["prism_center_pt"]).tolist()),
                "b_rotation_deg": tuple(float(v) for v in np.asarray(view_b["rotation_degrees"]).tolist()),
                "b_window_wc_ww": (float(view_b["wc"]), float(view_b["ww"])),
                "label_center_distance_mm": float(pair_labels["center_distance_mm"]),
                "label_rotation_delta_deg": tuple(float(v) for v in pair_labels["rotation_delta_deg"].tolist()),
                "label_window_delta": tuple(float(v) for v in pair_labels["window_delta"].tolist()),
            }
        ]
    )
    mo.vstack([mo.md("### Step 2 Output (A/B prism settings and labels)"), summary])
    return


@app.cell
def _(scan, view_a, view_b):
    sx, sy, sz = [int(v) for v in scan.data.shape]
    a_center = [int(v) for v in np.asarray(view_a["prism_center_vox"]).tolist()]
    b_center = [int(v) for v in np.asarray(view_b["prism_center_vox"]).tolist()]

    a_axial_idx = mo.ui.slider(label="A axial slice (z)", start=0, stop=max(sz - 1, 0), step=1, value=a_center[2])
    a_coronal_idx = mo.ui.slider(label="A coronal slice (y)", start=0, stop=max(sy - 1, 0), step=1, value=a_center[1])
    a_sagittal_idx = mo.ui.slider(label="A sagittal slice (x)", start=0, stop=max(sx - 1, 0), step=1, value=a_center[0])

    b_axial_idx = mo.ui.slider(label="B axial slice (z)", start=0, stop=max(sz - 1, 0), step=1, value=b_center[2])
    b_coronal_idx = mo.ui.slider(label="B coronal slice (y)", start=0, stop=max(sy - 1, 0), step=1, value=b_center[1])
    b_sagittal_idx = mo.ui.slider(label="B sagittal slice (x)", start=0, stop=max(sx - 1, 0), step=1, value=b_center[0])

    max_points_overlay = mo.ui.slider(label="Max patch points shown per slice", start=20, stop=1000, step=20, value=250)

    mo.vstack(
        [
            mo.md("### Step 2 Visual Checks"),
            mo.md(
                "Slider grounding: `x` = Left→Right, `y` = Posterior→Anterior, `z` = Inferior→Superior."
            ),
            mo.hstack([a_axial_idx, a_coronal_idx, a_sagittal_idx]),
            mo.hstack([b_axial_idx, b_coronal_idx, b_sagittal_idx]),
            mo.hstack([max_points_overlay]),
        ]
    )
    return a_axial_idx, a_coronal_idx, a_sagittal_idx, b_axial_idx, b_coronal_idx, b_sagittal_idx, max_points_overlay


@app.cell
def _(scan, view_a, view_b):
    rot_a = rotate_volume_about_center(
        scan.data,
        center_vox=np.asarray(view_a["prism_center_vox"], dtype=np.float32),
        rotation_matrix=np.asarray(view_a["rotation_matrix_ras"], dtype=np.float32),
    )
    rot_b = rotate_volume_about_center(
        scan.data,
        center_vox=np.asarray(view_b["prism_center_vox"], dtype=np.float32),
        rotation_matrix=np.asarray(view_b["rotation_matrix_ras"], dtype=np.float32),
    )

    rot_centers_a = rotated_patch_centers_vox(
        np.asarray(view_a["relative_patch_centers_pt_rotated"], dtype=np.float32),
        np.asarray(view_a["prism_center_vox"], dtype=np.float32),
        scan.spacing,
        scan.data.shape,
    )
    rot_centers_b = rotated_patch_centers_vox(
        np.asarray(view_b["relative_patch_centers_pt_rotated"], dtype=np.float32),
        np.asarray(view_b["prism_center_vox"], dtype=np.float32),
        scan.spacing,
        scan.data.shape,
    )
    return rot_a, rot_b, rot_centers_a, rot_centers_b


@app.cell
def _(
    a_axial_idx,
    a_coronal_idx,
    a_sagittal_idx,
    b_axial_idx,
    b_coronal_idx,
    b_sagittal_idx,
    max_points_overlay,
    rot_a,
    rot_b,
    rot_centers_a,
    rot_centers_b,
    scan,
    view_a,
    view_b,
):
    a_native_axial, _ = overlay_slice(
        scan.data,
        axis=2,
        slice_idx=int(a_axial_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=view_a["patch_centers_vox"],
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )
    a_native_coronal, _ = overlay_slice(
        scan.data,
        axis=1,
        slice_idx=int(a_coronal_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=view_a["patch_centers_vox"],
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )
    a_native_sagittal, _ = overlay_slice(
        scan.data,
        axis=0,
        slice_idx=int(a_sagittal_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=view_a["patch_centers_vox"],
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )

    b_native_axial, _ = overlay_slice(
        scan.data,
        axis=2,
        slice_idx=int(b_axial_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=view_b["patch_centers_vox"],
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )
    b_native_coronal, _ = overlay_slice(
        scan.data,
        axis=1,
        slice_idx=int(b_coronal_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=view_b["patch_centers_vox"],
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )
    b_native_sagittal, _ = overlay_slice(
        scan.data,
        axis=0,
        slice_idx=int(b_sagittal_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=view_b["patch_centers_vox"],
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )

    a_rot_axial, _ = overlay_slice(
        rot_a,
        axis=2,
        slice_idx=int(a_axial_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=rot_centers_a,
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )
    a_rot_coronal, _ = overlay_slice(
        rot_a,
        axis=1,
        slice_idx=int(a_coronal_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=rot_centers_a,
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )
    a_rot_sagittal, _ = overlay_slice(
        rot_a,
        axis=0,
        slice_idx=int(a_sagittal_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=rot_centers_a,
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
    )

    b_rot_axial, _ = overlay_slice(
        rot_b,
        axis=2,
        slice_idx=int(b_axial_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=rot_centers_b,
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )
    b_rot_coronal, _ = overlay_slice(
        rot_b,
        axis=1,
        slice_idx=int(b_coronal_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=rot_centers_b,
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )
    b_rot_sagittal, _ = overlay_slice(
        rot_b,
        axis=0,
        slice_idx=int(b_sagittal_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=rot_centers_b,
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
    )

    mo.vstack(
        [
            mo.md("#### Native-space overlays (yellow=patch centers, red=prism center)"),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"A axial z={a_axial_idx.value}"), mo.image(src=a_native_axial, width=300)]),
                    mo.vstack([mo.md(f"A coronal y={a_coronal_idx.value}"), mo.image(src=a_native_coronal, width=300)]),
                    mo.vstack([mo.md(f"A sagittal x={a_sagittal_idx.value}"), mo.image(src=a_native_sagittal, width=300)]),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"B axial z={b_axial_idx.value}"), mo.image(src=b_native_axial, width=300)]),
                    mo.vstack([mo.md(f"B coronal y={b_coronal_idx.value}"), mo.image(src=b_native_coronal, width=300)]),
                    mo.vstack([mo.md(f"B sagittal x={b_sagittal_idx.value}"), mo.image(src=b_native_sagittal, width=300)]),
                ]
            ),
            mo.md("#### Rotated-space overlays (same centers/slices, rotated around prism center)"),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"A axial z={a_axial_idx.value}"), mo.image(src=a_rot_axial, width=300)]),
                    mo.vstack([mo.md(f"A coronal y={a_coronal_idx.value}"), mo.image(src=a_rot_coronal, width=300)]),
                    mo.vstack([mo.md(f"A sagittal x={a_sagittal_idx.value}"), mo.image(src=a_rot_sagittal, width=300)]),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"B axial z={b_axial_idx.value}"), mo.image(src=b_rot_axial, width=300)]),
                    mo.vstack([mo.md(f"B coronal y={b_coronal_idx.value}"), mo.image(src=b_rot_coronal, width=300)]),
                    mo.vstack([mo.md(f"B sagittal x={b_sagittal_idx.value}"), mo.image(src=b_rot_sagittal, width=300)]),
                ]
            ),
        ]
    )
    return


@app.cell
def _(n_patches):
    preview_patches = mo.ui.slider(label="Patch previews", start=1, stop=max(int(n_patches.value), 1), step=1, value=min(36, int(n_patches.value)))
    preview_cols = mo.ui.slider(label="Grid columns", start=1, stop=12, step=1, value=6)
    coord_rows = mo.ui.slider(label="Patch rows in coordinate table", start=1, stop=200, step=1, value=30)
    coord_view = mo.ui.dropdown(options=["A", "B"], value="A", label="Coordinate table view")

    mo.vstack(
        [
            mo.md("## Step 3: Select/Inspect Patches"),
            mo.hstack([preview_patches, preview_cols, coord_rows, coord_view]),
        ]
    )
    return coord_rows, coord_view, preview_cols, preview_patches


@app.cell
def _(coord_rows, coord_view, patch_mm, preview_cols, preview_patches, scan, view_a, view_b):
    grid_a = patch_grid(view_a["normalized_patches"], max_patches=int(preview_patches.value), cols=int(preview_cols.value))
    grid_b = patch_grid(view_b["normalized_patches"], max_patches=int(preview_patches.value), cols=int(preview_cols.value))

    selected_view = view_a if str(coord_view.value) == "A" else view_b
    centers_vox = np.asarray(selected_view["patch_centers_vox"], dtype=np.int64)
    centers_pt = np.asarray(selected_view["patch_centers_pt"], dtype=np.float32)
    rel = np.asarray(selected_view["relative_patch_centers_pt"], dtype=np.float32)
    rel_rot = np.asarray(selected_view["relative_patch_centers_pt_rotated"], dtype=np.float32)
    rows = min(int(coord_rows.value), int(centers_vox.shape[0]))

    coords_df = pl.DataFrame(
        {
            "patch_idx": np.arange(rows, dtype=np.int64),
            "x_vox": centers_vox[:rows, 0],
            "y_vox": centers_vox[:rows, 1],
            "z_vox": centers_vox[:rows, 2],
            "x_mm": centers_pt[:rows, 0],
            "y_mm": centers_pt[:rows, 1],
            "z_mm": centers_pt[:rows, 2],
            "rel_x_mm": rel[:rows, 0],
            "rel_y_mm": rel[:rows, 1],
            "rel_z_mm": rel[:rows, 2],
            "rel_rot_x_mm": rel_rot[:rows, 0],
            "rel_rot_y_mm": rel_rot[:rows, 1],
            "rel_rot_z_mm": rel_rot[:rows, 2],
        }
    )

    mo.vstack(
        [
            mo.md(
                f"""
- `base_patch_mm`: `{float(patch_mm.value):.1f}`
- `patch_shape_vox` before resize: `{tuple(int(v) for v in scan.patch_shape_vox.tolist())}`
- final per-patch tensor shape: `(16, 16)`
"""
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md("View A normalized patches"), mo.image(src=grid_a, width=520)]),
                    mo.vstack([mo.md("View B normalized patches"), mo.image(src=grid_b, width=520)]),
                ]
            ),
            mo.md(f"Patch coordinates preview (View {coord_view.value})"),
            coords_df,
        ]
    )
    return


@app.cell
def _(view_a, view_b):
    patches_a, positions_a = tensorize_like_training(view_a)
    patches_b, positions_b = tensorize_like_training(view_b)

    model_summary = pl.DataFrame(
        [
            {
                "view": "A",
                "patches_shape": str(tuple(int(v) for v in patches_a.shape)),
                "patches_dtype": str(patches_a.dtype),
                "positions_shape": str(tuple(int(v) for v in positions_a.shape)),
                "positions_dtype": str(positions_a.dtype),
                "positions_abs_max_mm": float(np.max(np.abs(positions_a))),
            },
            {
                "view": "B",
                "patches_shape": str(tuple(int(v) for v in patches_b.shape)),
                "patches_dtype": str(patches_b.dtype),
                "positions_shape": str(tuple(int(v) for v in positions_b.shape)),
                "positions_dtype": str(positions_b.dtype),
                "positions_abs_max_mm": float(np.max(np.abs(positions_b))),
            },
        ]
    )

    pos_preview = pl.DataFrame(
        {
            "idx": np.arange(min(20, int(positions_a.shape[0])), dtype=np.int64),
            "A_x_mm": positions_a[:20, 0],
            "A_y_mm": positions_a[:20, 1],
            "A_z_mm": positions_a[:20, 2],
            "B_x_mm": positions_b[:20, 0],
            "B_y_mm": positions_b[:20, 1],
            "B_z_mm": positions_b[:20, 2],
        }
    )

    mo.vstack(
        [
            mo.md("## Step 4: Model Inputs (exact training tensor contract)"),
            mo.callout(
                "Current model path uses linear position projection (`pos_proj`) in the encoder; RoPE is not active in this codebase.",
                kind="info",
            ),
            model_summary,
            mo.md("First 20 relative position vectors (mm)"),
            pos_preview,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
