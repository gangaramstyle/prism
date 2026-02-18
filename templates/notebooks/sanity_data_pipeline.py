import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


with app.setup:
    import os
    import base64
    import io
    import sys
    import time
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from PIL import Image, ImageDraw

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.data import (
        build_dataset_item,
        collate_prism_batch,
        compute_pair_targets,
        infer_scan_geometry,
        load_catalog,
        load_nifti_scan,
        rotate_volume_about_center,
        rotated_relative_points_to_voxel,
        sample_scan_candidates,
        tensorize_sample_view,
    )

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
        patch_colors_rgb: np.ndarray | None = None,
        reference_patch_centers_vox: np.ndarray | None = None,
        reference_color_rgb: tuple[int, int, int] = (64, 224, 255),
    ) -> tuple[np.ndarray, int]:
        center = np.asarray(center_vox, dtype=np.int64)
        patches = np.asarray(patch_centers_vox, dtype=np.int64)
        patch_colors = None if patch_colors_rgb is None else np.asarray(patch_colors_rgb, dtype=np.uint8)
        ref_patches = None if reference_patch_centers_vox is None else np.asarray(reference_patch_centers_vox, dtype=np.int64)

        if axis == 2:
            base = window_to_rgb(volume[:, :, slice_idx], wc, ww)
            center_rc = (int(center[0]), int(center[1]))
            mask = patches[:, 2] == int(slice_idx)
            patch_indices = np.flatnonzero(mask)[:max_points]
            patch_rc = [(int(patches[i, 0]), int(patches[i, 1])) for i in patch_indices]
            if ref_patches is not None:
                ref_mask = ref_patches[:, 2] == int(slice_idx)
                ref_indices = np.flatnonzero(ref_mask)[:max_points]
                ref_rc = [(int(ref_patches[i, 0]), int(ref_patches[i, 1])) for i in ref_indices]
            else:
                ref_rc = []
        elif axis == 1:
            base = window_to_rgb(volume[:, slice_idx, :], wc, ww)
            center_rc = (int(center[0]), int(center[2]))
            mask = patches[:, 1] == int(slice_idx)
            patch_indices = np.flatnonzero(mask)[:max_points]
            patch_rc = [(int(patches[i, 0]), int(patches[i, 2])) for i in patch_indices]
            if ref_patches is not None:
                ref_mask = ref_patches[:, 1] == int(slice_idx)
                ref_indices = np.flatnonzero(ref_mask)[:max_points]
                ref_rc = [(int(ref_patches[i, 0]), int(ref_patches[i, 2])) for i in ref_indices]
            else:
                ref_rc = []
        else:
            base = window_to_rgb(volume[slice_idx, :, :], wc, ww)
            center_rc = (int(center[1]), int(center[2]))
            mask = patches[:, 0] == int(slice_idx)
            patch_indices = np.flatnonzero(mask)[:max_points]
            patch_rc = [(int(patches[i, 1]), int(patches[i, 2])) for i in patch_indices]
            if ref_patches is not None:
                ref_mask = ref_patches[:, 0] == int(slice_idx)
                ref_indices = np.flatnonzero(ref_mask)[:max_points]
                ref_rc = [(int(ref_patches[i, 1]), int(ref_patches[i, 2])) for i in ref_indices]
            else:
                ref_rc = []

        out = np.asarray(base, dtype=np.uint8).copy()
        for local_i, (row, col) in enumerate(patch_rc):
            if 0 <= row < out.shape[0] and 0 <= col < out.shape[1]:
                if patch_colors is not None and local_i < patch_indices.shape[0] and patch_indices[local_i] < patch_colors.shape[0]:
                    out[row, col] = patch_colors[patch_indices[local_i]]
                else:
                    out[row, col] = np.array([255, 220, 0], dtype=np.uint8)
        for row, col in ref_rc:
            if 0 <= row < out.shape[0] and 0 <= col < out.shape[1]:
                draw_cross(out, row, col, color=reference_color_rgb, radius=2)
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

    def euler_xyz_to_matrix(degrees_xyz: tuple[float, float, float]) -> np.ndarray:
        x, y, z = [np.deg2rad(float(v)) for v in degrees_xyz]
        cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
        sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return rz @ ry @ rx

    def _hex_to_rgb(value: str) -> tuple[int, int, int]:
        text = value.strip().lstrip("#")
        if len(text) != 6:
            return (255, 255, 0)
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))

    def render_position_parallax_gif_base64(
        rel_orig: np.ndarray,
        rel_selected: np.ndarray,
        color_hex: list[str],
        *,
        width: int = 520,
        height: int = 360,
        frames: int = 28,
    ) -> str:
        orig = np.asarray(rel_orig, dtype=np.float32)
        sel = np.asarray(rel_selected, dtype=np.float32)
        if orig.shape != sel.shape or orig.ndim != 2 or orig.shape[1] != 3 or orig.shape[0] == 0:
            return ""

        pts = np.concatenate([orig, sel], axis=0)
        max_abs = float(np.max(np.abs(pts)))
        scale = max(max_abs, 1e-6)
        orig_n = orig / scale
        sel_n = sel / scale

        center_x = width // 2
        center_y = height // 2
        proj_scale = int(min(width, height) * 0.34)
        focal = 2.5
        pad = 20

        palette = [_hex_to_rgb(c) for c in color_hex]
        if len(palette) < orig.shape[0]:
            palette = palette + [(255, 255, 0)] * (orig.shape[0] - len(palette))

        pil_frames: list[Image.Image] = []
        for i in range(int(max(frames, 2))):
            phase = 2.0 * np.pi * float(i) / float(max(frames, 2))
            yaw = 18.0 * np.sin(phase)
            pitch = 8.0 * np.sin(phase + 0.9)
            rot = euler_xyz_to_matrix((pitch, 0.0, yaw)).astype(np.float32)

            o = (rot @ orig_n.T).T
            s = (rot @ sel_n.T).T

            def _project(arr: np.ndarray) -> np.ndarray:
                z = arr[:, 2]
                depth = focal / np.clip(focal - z, 0.6, None)
                x = arr[:, 0] * depth
                y = arr[:, 1] * depth
                px = np.clip(np.rint(center_x + x * proj_scale), pad, width - pad).astype(np.int32)
                py = np.clip(np.rint(center_y - y * proj_scale), pad, height - pad).astype(np.int32)
                return np.stack([px, py], axis=1)

            o2 = _project(o)
            s2 = _project(s)

            img = Image.new("RGB", (width, height), color=(15, 18, 26))
            draw = ImageDraw.Draw(img)

            for j in range(orig.shape[0]):
                color = tuple(int(v) for v in palette[j])
                ox, oy = int(o2[j, 0]), int(o2[j, 1])
                sx, sy = int(s2[j, 0]), int(s2[j, 1])
                draw.line([(ox - 4, oy), (ox + 4, oy)], fill=color, width=2)
                draw.line([(ox, oy - 4), (ox, oy + 4)], fill=color, width=2)
                draw.ellipse([(sx - 3, sy - 3), (sx + 3, sy + 3)], outline=color, fill=color, width=1)

            draw.text((10, 10), "3D parallax: circle=selected, cross=original (same color)", fill=(220, 220, 220))
            pil_frames.append(img)

        buff = io.BytesIO()
        pil_frames[0].save(
            buff,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=90,
            disposal=2,
        )
        return base64.b64encode(buff.getvalue()).decode("ascii")

    def _plane_axes_for_slice(axis: int) -> tuple[int, int]:
        if axis == 2:  # axial -> rows=x, cols=y
            return 0, 1
        if axis == 1:  # coronal -> rows=x, cols=z
            return 0, 2
        if axis == 0:  # sagittal -> rows=y, cols=z
            return 1, 2
        raise ValueError(f"Invalid axis={axis}")

    def _aspect_ratio_for_display(
        image: np.ndarray,
        axis: int,
        spacing_mm: np.ndarray,
        *,
        mode: str,
    ) -> float:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return 1.0
        if mode == "spacing-aware":
            row_axis, col_axis = _plane_axes_for_slice(axis)
            spacing = np.asarray(spacing_mm, dtype=np.float32)
            row_mm = float(spacing[row_axis])
            col_mm = float(spacing[col_axis])
            return (float(h) * row_mm) / max(float(w) * col_mm, 1e-6)
        return float(h) / max(float(w), 1e-6)

    def fit_image_for_display(
        image: np.ndarray,
        axis: int,
        spacing_mm: np.ndarray,
        *,
        target_width_px: int,
        max_height_px: int,
        aspect_mode: str,
    ) -> tuple[np.ndarray, int]:
        arr = np.asarray(image, dtype=np.uint8)
        aspect = _aspect_ratio_for_display(arr, axis, spacing_mm, mode=aspect_mode)
        preferred_width = max(int(target_width_px), 64)
        max_height = max(int(max_height_px), 64)

        predicted_height = float(preferred_width) * aspect
        if predicted_height <= max_height:
            width = preferred_width
        else:
            width = max(64, int(round(float(max_height) / max(aspect, 1e-6))))

        src_h, src_w = arr.shape[:2]
        dst_h = max(64, int(round(float(width) * aspect)))
        if dst_h > max_height:
            dst_h = max_height
            width = max(64, int(round(float(dst_h) / max(aspect, 1e-6))))
        if width == src_w and dst_h == src_h:
            return arr, width

        resized = np.asarray(Image.fromarray(arr).resize((int(width), int(dst_h)), resample=Image.BILINEAR))
        return resized, int(width)

    def slice_mask(mask_volume: np.ndarray, axis: int, slice_idx: int) -> np.ndarray:
        mask = np.asarray(mask_volume, dtype=bool)
        if axis == 2:
            return mask[:, :, slice_idx]
        if axis == 1:
            return mask[:, slice_idx, :]
        if axis == 0:
            return mask[slice_idx, :, :]
        raise ValueError(f"Invalid axis={axis}")

    def apply_valid_mask_policy(
        image: np.ndarray,
        valid_mask_2d: np.ndarray,
        *,
        policy: str,
        pad_px: int,
    ) -> np.ndarray:
        arr = np.asarray(image, dtype=np.uint8).copy()
        mask2d = np.asarray(valid_mask_2d, dtype=bool)
        if mask2d.shape[:2] != arr.shape[:2]:
            raise ValueError(f"Mask/image shape mismatch: mask={mask2d.shape} image={arr.shape}")

        if policy in {"mask-invalid", "crop-to-valid"}:
            arr[~mask2d] = np.array([0, 0, 0], dtype=np.uint8)
        if policy != "crop-to-valid":
            return arr

        ys, xs = np.where(mask2d)
        if ys.size == 0 or xs.size == 0:
            return arr
        h, w = arr.shape[:2]
        pad = max(int(pad_px), 0)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, h)
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, w)
        return arr[y0:y1, x0:x1]


@app.cell
def _():
    from textwrap import dedent

    mo.md(
        dedent(
            """
# Prism SSL Data Pipeline Sanity Notebook

This notebook now uses the **same core sampling->tensor->label path as training**.

- Step 1. Load scan + metadata + orientation inference
- Step 2. Generate A/B views with training sampler and inspect native/rotated overlays
- Step 3. Inspect sampled patch coordinates and previews
- Step 4. Inspect exact collated batch fields used by the train loop
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
    suggested_wc_default = 0.5 * (float(scan.robust_low) + float(scan.robust_high))
    suggested_ww_default = max(float(scan.robust_high) - float(scan.robust_low), 1.0)

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
- `suggested_default_wc_ww`: `({suggested_wc_default:.2f}, {suggested_ww_default:.2f})` from robust percentile window
"""
            ),
            mo.md("#### Selected Series Metadata"),
            metadata_table,
        ]
    )
    return suggested_wc_default, suggested_ww_default


@app.cell
def _(scan):
    hist_sample_points = mo.ui.slider(label="Histogram sample points", start=50000, stop=2000000, step=50000, value=500000)
    hist_bins = mo.ui.slider(label="Histogram bins", start=32, stop=512, step=16, value=192)
    clip_low_pct = mo.ui.slider(label="Clip low percentile", start=0.0, stop=10.0, step=0.5, value=0.5)
    clip_high_pct = mo.ui.slider(label="Clip high percentile", start=90.0, stop=100.0, step=0.5, value=99.5)

    mo.vstack(
        [
            mo.md("### Pixel Histogram (for WC/WW selection)"),
            mo.hstack([hist_sample_points, hist_bins, clip_low_pct, clip_high_pct]),
        ]
    )
    return clip_high_pct, clip_low_pct, hist_bins, hist_sample_points


@app.cell
def _(clip_high_pct, clip_low_pct, hist_bins, hist_sample_points, scan):
    flat = np.asarray(scan.data, dtype=np.float32).reshape(-1)
    n = int(hist_sample_points.value)
    stride = max(int(flat.size // max(n, 1)), 1)
    sampled = flat[::stride][:n]

    low = float(clip_low_pct.value)
    high = float(clip_high_pct.value)
    mo.stop(low >= high, mo.callout("Clip low percentile must be < clip high percentile", kind="warn"))

    p_low = float(np.percentile(sampled, low))
    p_high = float(np.percentile(sampled, high))
    clipped = np.clip(sampled, p_low, p_high)
    hist, edges = np.histogram(clipped, bins=int(hist_bins.value))

    bars = [
        {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(hist[i])}
        for i in range(len(hist))
    ]
    chart = (
        alt.Chart(bars)
        .mark_bar()
        .encode(
            x=alt.X("x0:Q", title="Intensity"),
            x2="x1:Q",
            y=alt.Y("count:Q", title="Voxel count"),
        )
        .properties(height=240)
    )

    suggested_wc = 0.5 * (p_low + p_high)
    suggested_ww = max(p_high - p_low, 1.0)

    mo.vstack(
        [
            mo.md(
                f"""
Sampled voxels: `{int(sampled.size):,}` of `{int(flat.size):,}`

Suggested from selected percentile window:
- `wc ~ {suggested_wc:.2f}`
- `ww ~ {suggested_ww:.2f}`
"""
            ),
            chart,
        ]
    )
    return


@app.cell
def _(geometry, scan, suggested_wc_default, suggested_ww_default):
    patch_square_options = [k * k for k in range(1, 33)]
    n_patches = mo.ui.dropdown(options=patch_square_options, value=16, label="n_patches (square)")
    patch_output_px = mo.ui.slider(label="Patch output size (px)", start=16, stop=192, step=8, value=64)
    method = mo.ui.dropdown(options=["optimized_fused", "legacy_loop"], value="optimized_fused", label="Sampling method")
    sample_mode = mo.ui.dropdown(options=["pipeline-random", "manual-debug"], value="manual-debug", label="Sampling mode")
    sample_seed = mo.ui.number(label="Sample seed", value=1234, step=1)
    lock_b_to_a = mo.ui.checkbox(label="Manual mode: make View B identical to View A", value=False)
    position_frame_for_model = mo.ui.dropdown(
        options=["ras", "aug", "final"],
        value="aug",
        label="Model position frame",
    )
    random_aug_max_deg = mo.ui.slider(
        label="Pipeline-random max |rotation augmentation| (deg)",
        start=0.0,
        stop=45.0,
        step=0.5,
        value=10.0,
    )

    default_center = [int(v) // 2 for v in scan.data.shape]
    default_wc = float(suggested_wc_default)
    default_ww = float(suggested_ww_default)
    rot_base = tuple(float(v) for v in geometry.baseline_rotation_degrees)

    sampling_radius_mm = mo.ui.slider(label="Sampling radius (mm)", start=0.0, stop=80.0, step=0.5, value=50.0)

    a_center_x = mo.ui.number(label="A center X", value=default_center[0], start=0, stop=int(scan.data.shape[0] - 1), step=1)
    a_center_y = mo.ui.number(label="A center Y", value=default_center[1], start=0, stop=int(scan.data.shape[1] - 1), step=1)
    a_center_z = mo.ui.number(label="A center Z", value=default_center[2], start=0, stop=int(scan.data.shape[2] - 1), step=1)
    a_aug_x = mo.ui.slider(label="A rot-aug X (deg)", start=-45, stop=45, step=1, value=0)
    a_aug_y = mo.ui.slider(label="A rot-aug Y (deg)", start=-45, stop=45, step=1, value=0)
    a_aug_z = mo.ui.slider(label="A rot-aug Z (deg)", start=-45, stop=45, step=1, value=0)
    a_wc = mo.ui.number(label="A WC", value=default_wc, step=1.0)
    a_ww = mo.ui.number(label="A WW", value=default_ww, start=1.0, step=1.0)

    b_center_x = mo.ui.number(label="B center X", value=default_center[0], start=0, stop=int(scan.data.shape[0] - 1), step=1)
    b_center_y = mo.ui.number(label="B center Y", value=default_center[1], start=0, stop=int(scan.data.shape[1] - 1), step=1)
    b_center_z = mo.ui.number(label="B center Z", value=default_center[2], start=0, stop=int(scan.data.shape[2] - 1), step=1)
    b_aug_x = mo.ui.slider(label="B rot-aug X (deg)", start=-45, stop=45, step=1, value=0)
    b_aug_y = mo.ui.slider(label="B rot-aug Y (deg)", start=-45, stop=45, step=1, value=0)
    b_aug_z = mo.ui.slider(label="B rot-aug Z (deg)", start=-45, stop=45, step=1, value=0)
    b_wc = mo.ui.number(label="B WC", value=default_wc, step=1.0)
    b_ww = mo.ui.number(label="B WW", value=default_ww, start=1.0, step=1.0)

    mo.vstack(
        [
            mo.md("## Step 2: Select Prism Center / Rotation / Window"),
            mo.md(
                f"Native orientation hint (deg): `{rot_base}`. Effective rotation uses global-RAS composition: `R_eff = R(hint) @ R(rot-aug)`."
            ),
            mo.hstack(
                [
                    n_patches,
                    patch_output_px,
                    method,
                    sample_mode,
                    sample_seed,
                    lock_b_to_a,
                    position_frame_for_model,
                    random_aug_max_deg,
                ]
            ),
            mo.hstack([sampling_radius_mm]),
            mo.md("### View A manual controls"),
            mo.hstack([a_center_x, a_center_y, a_center_z, a_aug_x, a_aug_y, a_aug_z, a_wc, a_ww]),
            mo.md("### View B manual controls"),
            mo.hstack([b_center_x, b_center_y, b_center_z, b_aug_x, b_aug_y, b_aug_z, b_wc, b_ww]),
        ]
    )

    return (
        n_patches,
        patch_output_px,
        method,
        sample_mode,
        sample_seed,
        lock_b_to_a,
        position_frame_for_model,
        random_aug_max_deg,
        sampling_radius_mm,
        a_center_x,
        a_center_y,
        a_center_z,
        a_aug_x,
        a_aug_y,
        a_aug_z,
        a_wc,
        a_ww,
        b_center_x,
        b_center_y,
        b_center_z,
        b_aug_x,
        b_aug_y,
        b_aug_z,
        b_wc,
        b_ww,
    )


@app.cell
def _(
    a_aug_x,
    a_aug_y,
    a_aug_z,
    a_center_x,
    a_center_y,
    a_center_z,
    a_wc,
    a_ww,
    b_aug_x,
    b_aug_y,
    b_aug_z,
    b_center_x,
    b_center_y,
    b_center_z,
    b_wc,
    b_ww,
    lock_b_to_a,
    method,
    n_patches,
    patch_output_px,
    position_frame_for_model,
    random_aug_max_deg,
    sample_mode,
    sample_seed,
    sampling_radius_mm,
    scan,
    selected_record,
):
    common = {
        "n_patches": int(n_patches.value),
        "target_patch_size": int(patch_output_px.value),
        "method": str(method.value),
    }
    a_rotation_aug = (
        float(a_aug_x.value),
        float(a_aug_y.value),
        float(a_aug_z.value),
    )
    b_rotation_aug = (
        float(b_aug_x.value),
        float(b_aug_y.value),
        float(b_aug_z.value),
    )
    shared_rotation_kwargs = {
        "apply_native_orientation_hint": True,
        "rotation_augmentation_max_degrees": float(random_aug_max_deg.value),
    }

    if str(sample_mode.value) == "pipeline-random":
        view_a = scan.train_sample(
            seed=int(sample_seed.value) * 2,
            sampling_radius_mm=float(sampling_radius_mm.value),
            **common,
            **shared_rotation_kwargs,
        )
        view_b = scan.train_sample(
            seed=int(sample_seed.value) * 2 + 1,
            sampling_radius_mm=float(sampling_radius_mm.value),
            **common,
            **shared_rotation_kwargs,
        )
    else:
        a_kwargs = {
            "sampling_radius_mm": float(sampling_radius_mm.value),
            "rotation_augmentation_degrees": a_rotation_aug,
            "apply_native_orientation_hint": True,
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
                "rotation_augmentation_degrees": b_rotation_aug,
                "apply_native_orientation_hint": True,
                "subset_center_vox": np.asarray(
                    [int(b_center_x.value), int(b_center_y.value), int(b_center_z.value)],
                    dtype=np.int64,
                ),
                "wc": float(b_wc.value),
                "ww": float(b_ww.value),
            }
            view_b = scan.train_sample(seed=int(sample_seed.value) + 1, **common, **b_kwargs)

    position_frame = str(position_frame_for_model.value)
    view_a_t = tensorize_sample_view(view_a, position_frame=position_frame)
    view_b_t = tensorize_sample_view(view_b, position_frame=position_frame)
    pair_targets = compute_pair_targets(view_a_t, view_b_t)

    dataset_item = build_dataset_item(
        result_a=view_a,
        result_b=view_b,
        scan_id=str(selected_record.scan_id),
        series_id=str(selected_record.series_id),
        position_frame=position_frame,
    )

    return dataset_item, pair_targets, position_frame, view_a, view_b, view_a_t, view_b_t


@app.cell
def _(pair_targets, position_frame, view_a, view_b):
    step2_summary = pl.DataFrame(
        [
            {
                "a_center_vox": tuple(int(v) for v in np.asarray(view_a["prism_center_vox"]).tolist()),
                "a_center_mm": tuple(float(v) for v in np.asarray(view_a["prism_center_pt"]).tolist()),
                "a_rotation_hint_deg": tuple(float(v) for v in np.asarray(view_a["rotation_hint_degrees"]).tolist()),
                "a_rotation_aug_deg": tuple(
                    float(v) for v in np.asarray(view_a["rotation_augmentation_degrees"]).tolist()
                ),
                "a_rotation_control_deg": tuple(float(v) for v in np.asarray(view_a["rotation_degrees"]).tolist()),
                "a_rotation_effective_deg": tuple(
                    float(v) for v in np.asarray(view_a["rotation_effective_degrees"]).tolist()
                ),
                "a_window_wc_ww": (float(view_a["wc"]), float(view_a["ww"])),
                "a_target_patch_size": int(view_a["target_patch_size"]),
                "b_center_vox": tuple(int(v) for v in np.asarray(view_b["prism_center_vox"]).tolist()),
                "b_center_mm": tuple(float(v) for v in np.asarray(view_b["prism_center_pt"]).tolist()),
                "b_rotation_hint_deg": tuple(float(v) for v in np.asarray(view_b["rotation_hint_degrees"]).tolist()),
                "b_rotation_aug_deg": tuple(
                    float(v) for v in np.asarray(view_b["rotation_augmentation_degrees"]).tolist()
                ),
                "b_rotation_control_deg": tuple(float(v) for v in np.asarray(view_b["rotation_degrees"]).tolist()),
                "b_rotation_effective_deg": tuple(
                    float(v) for v in np.asarray(view_b["rotation_effective_degrees"]).tolist()
                ),
                "b_window_wc_ww": (float(view_b["wc"]), float(view_b["ww"])),
                "b_target_patch_size": int(view_b["target_patch_size"]),
                "label_center_distance_mm": float(pair_targets["center_distance_mm"].item()),
                "label_rotation_delta_deg": tuple(float(v) for v in pair_targets["rotation_delta_deg"].tolist()),
                "label_window_delta": tuple(float(v) for v in pair_targets["window_delta"].tolist()),
                "position_frame_for_model": position_frame,
            }
        ]
    )
    mo.vstack([mo.md("### Step 2 Output (A/B prism settings and labels)"), step2_summary])
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
    display_aspect_mode = mo.ui.dropdown(
        options=["spacing-aware", "voxel-grid"],
        value="spacing-aware",
        label="Display aspect",
    )
    rotated_invalid_policy = mo.ui.dropdown(
        options=["crop-to-valid", "mask-invalid", "raw"],
        value="crop-to-valid",
        label="Rotated invalid voxels",
    )
    rotated_crop_pad_px = mo.ui.slider(label="Rotated crop padding (px)", start=0, stop=48, step=2, value=8)
    display_target_width = mo.ui.slider(label="Target display width (px)", start=180, stop=480, step=10, value=300)
    display_max_height = mo.ui.slider(label="Max display height (px)", start=140, stop=360, step=10, value=220)

    mo.vstack(
        [
            mo.md("### Step 2 Visual Checks"),
            mo.md("Slider grounding: `x` = Left→Right, `y` = Posterior→Anterior, `z` = Inferior→Superior."),
            mo.hstack([a_axial_idx, a_coronal_idx, a_sagittal_idx]),
            mo.hstack([b_axial_idx, b_coronal_idx, b_sagittal_idx]),
            mo.hstack(
                [
                    max_points_overlay,
                    display_aspect_mode,
                    rotated_invalid_policy,
                    rotated_crop_pad_px,
                    display_target_width,
                    display_max_height,
                ]
            ),
        ]
    )
    return (
        a_axial_idx,
        a_coronal_idx,
        a_sagittal_idx,
        b_axial_idx,
        b_coronal_idx,
        b_sagittal_idx,
        display_aspect_mode,
        display_max_height,
        display_target_width,
        max_points_overlay,
        rotated_crop_pad_px,
        rotated_invalid_policy,
    )


@app.cell
def _(euler_xyz_to_matrix, scan, view_a, view_b):
    aug_mat_a = euler_xyz_to_matrix(tuple(float(v) for v in np.asarray(view_a["rotation_augmentation_degrees"]).tolist()))
    aug_mat_b = euler_xyz_to_matrix(tuple(float(v) for v in np.asarray(view_b["rotation_augmentation_degrees"]).tolist()))

    rot_aug_only_a = rotate_volume_about_center(
        scan.data,
        center_vox=np.asarray(view_a["prism_center_vox"], dtype=np.float32),
        rotation_matrix=aug_mat_a,
        spacing_mm=scan.spacing,
        mode="constant",
    )
    rot_aug_only_b = rotate_volume_about_center(
        scan.data,
        center_vox=np.asarray(view_b["prism_center_vox"], dtype=np.float32),
        rotation_matrix=aug_mat_b,
        spacing_mm=scan.spacing,
        mode="constant",
    )
    ones = np.ones_like(scan.data, dtype=np.float32)
    rot_aug_only_valid_mask_a = (
        rotate_volume_about_center(
            ones,
            center_vox=np.asarray(view_a["prism_center_vox"], dtype=np.float32),
            rotation_matrix=aug_mat_a,
            spacing_mm=scan.spacing,
            interpolation_order=0,
            mode="constant",
        )
        > 0.5
    )
    rot_aug_only_valid_mask_b = (
        rotate_volume_about_center(
            ones,
            center_vox=np.asarray(view_b["prism_center_vox"], dtype=np.float32),
            rotation_matrix=aug_mat_b,
            spacing_mm=scan.spacing,
            interpolation_order=0,
            mode="constant",
        )
        > 0.5
    )
    rel_a = np.asarray(view_a["relative_patch_centers_pt_ras"], dtype=np.float32)
    rel_b = np.asarray(view_b["relative_patch_centers_pt_ras"], dtype=np.float32)
    rel_aug_a = (aug_mat_a @ rel_a.T).T.astype(np.float32, copy=False)
    rel_aug_b = (aug_mat_b @ rel_b.T).T.astype(np.float32, copy=False)
    rot_aug_only_centers_a = rotated_relative_points_to_voxel(
        rel_aug_a,
        np.asarray(view_a["prism_center_vox"], dtype=np.float32),
        scan.spacing,
        shape_vox=scan.data.shape,
    )
    rot_aug_only_centers_b = rotated_relative_points_to_voxel(
        rel_aug_b,
        np.asarray(view_b["prism_center_vox"], dtype=np.float32),
        scan.spacing,
        shape_vox=scan.data.shape,
    )

    rot_a = rotate_volume_about_center(
        scan.data,
        center_vox=np.asarray(view_a["prism_center_vox"], dtype=np.float32),
        rotation_matrix=np.asarray(view_a["rotation_matrix_ras"], dtype=np.float32),
        spacing_mm=scan.spacing,
        mode="constant",
    )
    rot_b = rotate_volume_about_center(
        scan.data,
        center_vox=np.asarray(view_b["prism_center_vox"], dtype=np.float32),
        rotation_matrix=np.asarray(view_b["rotation_matrix_ras"], dtype=np.float32),
        spacing_mm=scan.spacing,
        mode="constant",
    )
    rot_valid_mask_a = (
        rotate_volume_about_center(
            ones,
            center_vox=np.asarray(view_a["prism_center_vox"], dtype=np.float32),
            rotation_matrix=np.asarray(view_a["rotation_matrix_ras"], dtype=np.float32),
            spacing_mm=scan.spacing,
            interpolation_order=0,
            mode="constant",
        )
        > 0.5
    )
    rot_valid_mask_b = (
        rotate_volume_about_center(
            ones,
            center_vox=np.asarray(view_b["prism_center_vox"], dtype=np.float32),
            rotation_matrix=np.asarray(view_b["rotation_matrix_ras"], dtype=np.float32),
            spacing_mm=scan.spacing,
            interpolation_order=0,
            mode="constant",
        )
        > 0.5
    )

    rot_centers_a = rotated_relative_points_to_voxel(
        np.asarray(view_a["relative_patch_centers_pt_final"], dtype=np.float32),
        np.asarray(view_a["prism_center_vox"], dtype=np.float32),
        scan.spacing,
        shape_vox=scan.data.shape,
    )
    rot_centers_b = rotated_relative_points_to_voxel(
        np.asarray(view_b["relative_patch_centers_pt_final"], dtype=np.float32),
        np.asarray(view_b["prism_center_vox"], dtype=np.float32),
        scan.spacing,
        shape_vox=scan.data.shape,
    )
    return (
        rot_a,
        rot_aug_only_a,
        rot_aug_only_b,
        rot_b,
        rot_aug_only_centers_a,
        rot_aug_only_centers_b,
        rot_centers_a,
        rot_centers_b,
        rot_aug_only_valid_mask_a,
        rot_aug_only_valid_mask_b,
        rot_valid_mask_a,
        rot_valid_mask_b,
    )


@app.cell
def _(
    a_axial_idx,
    a_coronal_idx,
    a_sagittal_idx,
    apply_valid_mask_policy,
    b_axial_idx,
    b_coronal_idx,
    b_sagittal_idx,
    display_aspect_mode,
    display_max_height,
    display_target_width,
    fit_image_for_display,
    max_points_overlay,
    rot_a,
    rot_aug_only_a,
    rot_aug_only_b,
    rot_b,
    rot_aug_only_centers_a,
    rot_aug_only_centers_b,
    rot_centers_a,
    rot_centers_b,
    rot_aug_only_valid_mask_a,
    rot_aug_only_valid_mask_b,
    rot_valid_mask_a,
    rot_valid_mask_b,
    rotated_crop_pad_px,
    rotated_invalid_policy,
    scan,
    slice_mask,
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

    a_aug_axial, _ = overlay_slice(
        rot_aug_only_a,
        axis=2,
        slice_idx=int(a_axial_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=rot_aug_only_centers_a,
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
        reference_patch_centers_vox=np.asarray(view_a["patch_centers_vox"], dtype=np.int64),
    )
    a_aug_coronal, _ = overlay_slice(
        rot_aug_only_a,
        axis=1,
        slice_idx=int(a_coronal_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=rot_aug_only_centers_a,
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
        reference_patch_centers_vox=np.asarray(view_a["patch_centers_vox"], dtype=np.int64),
    )
    a_aug_sagittal, _ = overlay_slice(
        rot_aug_only_a,
        axis=0,
        slice_idx=int(a_sagittal_idx.value),
        center_vox=view_a["prism_center_vox"],
        patch_centers_vox=rot_aug_only_centers_a,
        wc=float(view_a["wc"]),
        ww=float(view_a["ww"]),
        max_points=int(max_points_overlay.value),
        reference_patch_centers_vox=np.asarray(view_a["patch_centers_vox"], dtype=np.int64),
    )

    b_aug_axial, _ = overlay_slice(
        rot_aug_only_b,
        axis=2,
        slice_idx=int(b_axial_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=rot_aug_only_centers_b,
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
        reference_patch_centers_vox=np.asarray(view_b["patch_centers_vox"], dtype=np.int64),
    )
    b_aug_coronal, _ = overlay_slice(
        rot_aug_only_b,
        axis=1,
        slice_idx=int(b_coronal_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=rot_aug_only_centers_b,
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
        reference_patch_centers_vox=np.asarray(view_b["patch_centers_vox"], dtype=np.int64),
    )
    b_aug_sagittal, _ = overlay_slice(
        rot_aug_only_b,
        axis=0,
        slice_idx=int(b_sagittal_idx.value),
        center_vox=view_b["prism_center_vox"],
        patch_centers_vox=rot_aug_only_centers_b,
        wc=float(view_b["wc"]),
        ww=float(view_b["ww"]),
        max_points=int(max_points_overlay.value),
        reference_patch_centers_vox=np.asarray(view_b["patch_centers_vox"], dtype=np.int64),
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
    aspect_mode = str(display_aspect_mode.value)
    invalid_policy = str(rotated_invalid_policy.value)
    crop_pad = int(rotated_crop_pad_px.value)

    a_aug_axial = apply_valid_mask_policy(
        a_aug_axial,
        slice_mask(rot_aug_only_valid_mask_a, axis=2, slice_idx=int(a_axial_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    a_aug_coronal = apply_valid_mask_policy(
        a_aug_coronal,
        slice_mask(rot_aug_only_valid_mask_a, axis=1, slice_idx=int(a_coronal_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    a_aug_sagittal = apply_valid_mask_policy(
        a_aug_sagittal,
        slice_mask(rot_aug_only_valid_mask_a, axis=0, slice_idx=int(a_sagittal_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    b_aug_axial = apply_valid_mask_policy(
        b_aug_axial,
        slice_mask(rot_aug_only_valid_mask_b, axis=2, slice_idx=int(b_axial_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    b_aug_coronal = apply_valid_mask_policy(
        b_aug_coronal,
        slice_mask(rot_aug_only_valid_mask_b, axis=1, slice_idx=int(b_coronal_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    b_aug_sagittal = apply_valid_mask_policy(
        b_aug_sagittal,
        slice_mask(rot_aug_only_valid_mask_b, axis=0, slice_idx=int(b_sagittal_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )

    a_rot_axial = apply_valid_mask_policy(
        a_rot_axial,
        slice_mask(rot_valid_mask_a, axis=2, slice_idx=int(a_axial_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    a_rot_coronal = apply_valid_mask_policy(
        a_rot_coronal,
        slice_mask(rot_valid_mask_a, axis=1, slice_idx=int(a_coronal_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    a_rot_sagittal = apply_valid_mask_policy(
        a_rot_sagittal,
        slice_mask(rot_valid_mask_a, axis=0, slice_idx=int(a_sagittal_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    b_rot_axial = apply_valid_mask_policy(
        b_rot_axial,
        slice_mask(rot_valid_mask_b, axis=2, slice_idx=int(b_axial_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    b_rot_coronal = apply_valid_mask_policy(
        b_rot_coronal,
        slice_mask(rot_valid_mask_b, axis=1, slice_idx=int(b_coronal_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )
    b_rot_sagittal = apply_valid_mask_policy(
        b_rot_sagittal,
        slice_mask(rot_valid_mask_b, axis=0, slice_idx=int(b_sagittal_idx.value)),
        policy=invalid_policy,
        pad_px=crop_pad,
    )

    def _display_ready(image: np.ndarray, axis: int) -> tuple[np.ndarray, int]:
        return fit_image_for_display(
            image,
            axis,
            scan.spacing,
            target_width_px=int(display_target_width.value),
            max_height_px=int(display_max_height.value),
            aspect_mode=aspect_mode,
        )

    a_native_axial_img, a_native_axial_w = _display_ready(a_native_axial, axis=2)
    a_native_coronal_img, a_native_coronal_w = _display_ready(a_native_coronal, axis=1)
    a_native_sagittal_img, a_native_sagittal_w = _display_ready(a_native_sagittal, axis=0)

    b_native_axial_img, b_native_axial_w = _display_ready(b_native_axial, axis=2)
    b_native_coronal_img, b_native_coronal_w = _display_ready(b_native_coronal, axis=1)
    b_native_sagittal_img, b_native_sagittal_w = _display_ready(b_native_sagittal, axis=0)

    a_aug_axial_img, a_aug_axial_w = _display_ready(a_aug_axial, axis=2)
    a_aug_coronal_img, a_aug_coronal_w = _display_ready(a_aug_coronal, axis=1)
    a_aug_sagittal_img, a_aug_sagittal_w = _display_ready(a_aug_sagittal, axis=0)

    b_aug_axial_img, b_aug_axial_w = _display_ready(b_aug_axial, axis=2)
    b_aug_coronal_img, b_aug_coronal_w = _display_ready(b_aug_coronal, axis=1)
    b_aug_sagittal_img, b_aug_sagittal_w = _display_ready(b_aug_sagittal, axis=0)

    a_rot_axial_img, a_rot_axial_w = _display_ready(a_rot_axial, axis=2)
    a_rot_coronal_img, a_rot_coronal_w = _display_ready(a_rot_coronal, axis=1)
    a_rot_sagittal_img, a_rot_sagittal_w = _display_ready(a_rot_sagittal, axis=0)

    b_rot_axial_img, b_rot_axial_w = _display_ready(b_rot_axial, axis=2)
    b_rot_coronal_img, b_rot_coronal_w = _display_ready(b_rot_coronal, axis=1)
    b_rot_sagittal_img, b_rot_sagittal_w = _display_ready(b_rot_sagittal, axis=0)

    mo.vstack(
        [
            mo.md("#### Native-space overlays (yellow=patch centers, red=prism center)"),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"A axial z={a_axial_idx.value}"), mo.image(src=a_native_axial_img, width=a_native_axial_w)]),
                    mo.vstack([mo.md(f"A coronal y={a_coronal_idx.value}"), mo.image(src=a_native_coronal_img, width=a_native_coronal_w)]),
                    mo.vstack([mo.md(f"A sagittal x={a_sagittal_idx.value}"), mo.image(src=a_native_sagittal_img, width=a_native_sagittal_w)]),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"B axial z={b_axial_idx.value}"), mo.image(src=b_native_axial_img, width=b_native_axial_w)]),
                    mo.vstack([mo.md(f"B coronal y={b_coronal_idx.value}"), mo.image(src=b_native_coronal_img, width=b_native_coronal_w)]),
                    mo.vstack([mo.md(f"B sagittal x={b_sagittal_idx.value}"), mo.image(src=b_native_sagittal_img, width=b_native_sagittal_w)]),
                ]
            ),
            mo.md("#### Aug-rotation overlays (RAS-only, before applying hint; red crosses = pre-rotation patch locations)"),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"A axial z={a_axial_idx.value}"), mo.image(src=a_aug_axial_img, width=a_aug_axial_w)]),
                    mo.vstack([mo.md(f"A coronal y={a_coronal_idx.value}"), mo.image(src=a_aug_coronal_img, width=a_aug_coronal_w)]),
                    mo.vstack([mo.md(f"A sagittal x={a_sagittal_idx.value}"), mo.image(src=a_aug_sagittal_img, width=a_aug_sagittal_w)]),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"B axial z={b_axial_idx.value}"), mo.image(src=b_aug_axial_img, width=b_aug_axial_w)]),
                    mo.vstack([mo.md(f"B coronal y={b_coronal_idx.value}"), mo.image(src=b_aug_coronal_img, width=b_aug_coronal_w)]),
                    mo.vstack([mo.md(f"B sagittal x={b_sagittal_idx.value}"), mo.image(src=b_aug_sagittal_img, width=b_aug_sagittal_w)]),
                ]
            ),
            mo.md("#### Rotated-space overlays (after hint + aug composition around prism center)"),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"A axial z={a_axial_idx.value}"), mo.image(src=a_rot_axial_img, width=a_rot_axial_w)]),
                    mo.vstack([mo.md(f"A coronal y={a_coronal_idx.value}"), mo.image(src=a_rot_coronal_img, width=a_rot_coronal_w)]),
                    mo.vstack([mo.md(f"A sagittal x={a_sagittal_idx.value}"), mo.image(src=a_rot_sagittal_img, width=a_rot_sagittal_w)]),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md(f"B axial z={b_axial_idx.value}"), mo.image(src=b_rot_axial_img, width=b_rot_axial_w)]),
                    mo.vstack([mo.md(f"B coronal y={b_coronal_idx.value}"), mo.image(src=b_rot_coronal_img, width=b_rot_coronal_w)]),
                    mo.vstack([mo.md(f"B sagittal x={b_sagittal_idx.value}"), mo.image(src=b_rot_sagittal_img, width=b_rot_sagittal_w)]),
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
    coord_frame = mo.ui.dropdown(options=["ras", "aug", "final"], value="aug", label="Coordinate frame")

    mo.vstack(
        [
            mo.md("## Step 3: Select/Inspect Patches"),
            mo.hstack([preview_patches, preview_cols, coord_rows, coord_view, coord_frame]),
        ]
    )
    return coord_frame, coord_rows, coord_view, preview_cols, preview_patches


@app.cell
def _(
    coord_frame,
    coord_rows,
    coord_view,
    fit_image_for_display,
    overlay_slice,
    patch_mm,
    preview_cols,
    preview_patches,
    render_position_parallax_gif_base64,
    sampling_radius_mm,
    scan,
    view_a,
    view_b,
):
    grid_a = patch_grid(view_a["normalized_patches"], max_patches=int(preview_patches.value), cols=int(preview_cols.value))
    grid_b = patch_grid(view_b["normalized_patches"], max_patches=int(preview_patches.value), cols=int(preview_cols.value))

    selected_view = view_a if str(coord_view.value) == "A" else view_b
    centers_vox = np.asarray(selected_view["patch_centers_vox"], dtype=np.int64)
    centers_pt = np.asarray(selected_view["patch_centers_pt"], dtype=np.float32)
    rel_ras = np.asarray(selected_view["relative_patch_centers_pt_ras"], dtype=np.float32)
    rel_aug = np.asarray(selected_view["relative_patch_centers_pt_aug"], dtype=np.float32)
    rel_final = np.asarray(selected_view["relative_patch_centers_pt_final"], dtype=np.float32)
    frame = str(coord_frame.value)
    frame_map = {
        "ras": rel_ras,
        "aug": rel_aug,
        "final": rel_final,
    }
    rel_selected = frame_map[frame]

    scale_mm = max(float(sampling_radius_mm.value), 1e-6)
    rel_scaled = np.clip(rel_selected / scale_mm, -1.0, 1.0)
    rgb_u8 = np.rint((rel_scaled + 1.0) * 127.5).astype(np.uint8)
    color_hex = [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in rgb_u8.tolist()]

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
            "rel_ras_x_mm": rel_ras[:rows, 0],
            "rel_ras_y_mm": rel_ras[:rows, 1],
            "rel_ras_z_mm": rel_ras[:rows, 2],
            "rel_aug_x_mm": rel_aug[:rows, 0],
            "rel_aug_y_mm": rel_aug[:rows, 1],
            "rel_aug_z_mm": rel_aug[:rows, 2],
            "rel_final_x_mm": rel_final[:rows, 0],
            "rel_final_y_mm": rel_final[:rows, 1],
            "rel_final_z_mm": rel_final[:rows, 2],
            "selected_x_mm": rel_selected[:rows, 0],
            "selected_y_mm": rel_selected[:rows, 1],
            "selected_z_mm": rel_selected[:rows, 2],
            "norm_ras_mm": np.linalg.norm(rel_ras[:rows], axis=1),
            "norm_aug_mm": np.linalg.norm(rel_aug[:rows], axis=1),
            "norm_final_mm": np.linalg.norm(rel_final[:rows], axis=1),
            "rgb_hex": color_hex[:rows],
            "rgb_r": rgb_u8[:rows, 0],
            "rgb_g": rgb_u8[:rows, 1],
            "rgb_b": rgb_u8[:rows, 2],
        }
    )

    debug_points = pl.DataFrame(
        {
            "patch_idx": np.arange(centers_vox.shape[0], dtype=np.int64),
            "x_sel_mm": rel_selected[:, 0],
            "y_sel_mm": rel_selected[:, 1],
            "z_sel_mm": rel_selected[:, 2],
            "x_orig_mm": rel_ras[:, 0],
            "y_orig_mm": rel_ras[:, 1],
            "z_orig_mm": rel_ras[:, 2],
            "color": color_hex,
        }
    )
    axial_sel = (
        alt.Chart(debug_points)
        .mark_circle(size=45)
        .encode(
            x=alt.X("x_sel_mm:Q", title="x (mm)"),
            y=alt.Y("y_sel_mm:Q", title="y (mm)"),
            color=alt.Color("color:N", scale=None),
            tooltip=["patch_idx:Q", "x_sel_mm:Q", "y_sel_mm:Q", "z_sel_mm:Q", "color:N"],
        )
    )
    axial_orig = (
        alt.Chart(debug_points)
        .mark_point(shape="cross", color="#ff3b30", size=140, strokeWidth=2.0)
        .encode(
            x="x_orig_mm:Q",
            y="y_orig_mm:Q",
            tooltip=["patch_idx:Q", "x_orig_mm:Q", "y_orig_mm:Q", "z_orig_mm:Q"],
        )
    )
    axial_chart = (axial_sel + axial_orig).properties(title=f"{frame.upper()} frame: axial XY", width=280, height=240)

    coronal_sel = (
        alt.Chart(debug_points)
        .mark_circle(size=45)
        .encode(
            x=alt.X("x_sel_mm:Q", title="x (mm)"),
            y=alt.Y("z_sel_mm:Q", title="z (mm)"),
            color=alt.Color("color:N", scale=None),
            tooltip=["patch_idx:Q", "x_sel_mm:Q", "y_sel_mm:Q", "z_sel_mm:Q", "color:N"],
        )
    )
    coronal_orig = (
        alt.Chart(debug_points)
        .mark_point(shape="cross", color="#ff3b30", size=140, strokeWidth=2.0)
        .encode(
            x="x_orig_mm:Q",
            y="z_orig_mm:Q",
            tooltip=["patch_idx:Q", "x_orig_mm:Q", "y_orig_mm:Q", "z_orig_mm:Q"],
        )
    )
    coronal_chart = (coronal_sel + coronal_orig).properties(title=f"{frame.upper()} frame: coronal XZ", width=280, height=240)

    sagittal_sel = (
        alt.Chart(debug_points)
        .mark_circle(size=45)
        .encode(
            x=alt.X("y_sel_mm:Q", title="y (mm)"),
            y=alt.Y("z_sel_mm:Q", title="z (mm)"),
            color=alt.Color("color:N", scale=None),
            tooltip=["patch_idx:Q", "x_sel_mm:Q", "y_sel_mm:Q", "z_sel_mm:Q", "color:N"],
        )
    )
    sagittal_orig = (
        alt.Chart(debug_points)
        .mark_point(shape="cross", color="#ff3b30", size=140, strokeWidth=2.0)
        .encode(
            x="y_orig_mm:Q",
            y="z_orig_mm:Q",
            tooltip=["patch_idx:Q", "x_orig_mm:Q", "y_orig_mm:Q", "z_orig_mm:Q"],
        )
    )
    sagittal_chart = (sagittal_sel + sagittal_orig).properties(title=f"{frame.upper()} frame: sagittal YZ", width=280, height=240)
    parallax_b64 = render_position_parallax_gif_base64(rel_ras, rel_selected, color_hex)
    if parallax_b64:
        parallax_view = mo.md(f"![3D position parallax](data:image/gif;base64,{parallax_b64})")
    else:
        parallax_view = mo.md("_3D parallax view unavailable (no points)._")

    center_vox = np.asarray(selected_view["prism_center_vox"], dtype=np.int64)
    center_color = rgb_u8
    overlay_axial, _ = overlay_slice(
        scan.data,
        axis=2,
        slice_idx=int(center_vox[2]),
        center_vox=center_vox,
        patch_centers_vox=centers_vox,
        wc=float(selected_view["wc"]),
        ww=float(selected_view["ww"]),
        max_points=1000,
        patch_colors_rgb=center_color,
    )
    overlay_coronal, _ = overlay_slice(
        scan.data,
        axis=1,
        slice_idx=int(center_vox[1]),
        center_vox=center_vox,
        patch_centers_vox=centers_vox,
        wc=float(selected_view["wc"]),
        ww=float(selected_view["ww"]),
        max_points=1000,
        patch_colors_rgb=center_color,
    )
    overlay_sagittal, _ = overlay_slice(
        scan.data,
        axis=0,
        slice_idx=int(center_vox[0]),
        center_vox=center_vox,
        patch_centers_vox=centers_vox,
        wc=float(selected_view["wc"]),
        ww=float(selected_view["ww"]),
        max_points=1000,
        patch_colors_rgb=center_color,
    )
    overlay_axial_img, overlay_axial_w = fit_image_for_display(
        overlay_axial,
        axis=2,
        spacing_mm=scan.spacing,
        target_width_px=280,
        max_height_px=220,
        aspect_mode="spacing-aware",
    )
    overlay_coronal_img, overlay_coronal_w = fit_image_for_display(
        overlay_coronal,
        axis=1,
        spacing_mm=scan.spacing,
        target_width_px=280,
        max_height_px=220,
        aspect_mode="spacing-aware",
    )
    overlay_sagittal_img, overlay_sagittal_w = fit_image_for_display(
        overlay_sagittal,
        axis=0,
        spacing_mm=scan.spacing,
        target_width_px=280,
        max_height_px=220,
        aspect_mode="spacing-aware",
    )

    mo.vstack(
        [
            mo.md(
                f"""
- `base_patch_mm`: `{float(patch_mm.value):.1f}`
- `patch_shape_vox` before resize: `{tuple(int(v) for v in scan.patch_shape_vox.tolist())}`
- final per-patch tensor shape: `{tuple(int(v) for v in np.asarray(view_a["normalized_patches"]).shape[1:3])}`
- Position debugger frame: `{frame}` (`rgb = normalized [x,y,z]` in selected frame; scale=`{scale_mm:.1f}mm`)
- Position scatter markers: `circles = selected frame`, `red crosses = pre-rotation RAS positions`
"""
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md("View A normalized patches"), mo.image(src=grid_a, width=520)]),
                    mo.vstack([mo.md("View B normalized patches"), mo.image(src=grid_b, width=520)]),
                ]
            ),
            mo.md("Color-coded relative-position debugger"),
            mo.hstack([axial_chart, coronal_chart, sagittal_chart]),
            parallax_view,
            mo.hstack(
                [
                    mo.vstack([mo.md(f"Native axial z={int(center_vox[2])}"), mo.image(src=overlay_axial_img, width=overlay_axial_w)]),
                    mo.vstack([mo.md(f"Native coronal y={int(center_vox[1])}"), mo.image(src=overlay_coronal_img, width=overlay_coronal_w)]),
                    mo.vstack([mo.md(f"Native sagittal x={int(center_vox[0])}"), mo.image(src=overlay_sagittal_img, width=overlay_sagittal_w)]),
                ]
            ),
            mo.md(f"Patch coordinates preview (View {coord_view.value}, frame={frame})"),
            coords_df,
        ]
    )
    return


@app.cell
def _(dataset_item):
    batch = collate_prism_batch([dataset_item])
    patch_hw = tuple(int(v) for v in batch["patches_a"].shape[2:4])
    patch_size_note = (
        "Using default training patch size (16x16)."
        if patch_hw == (16, 16)
        else f"Debug patch size override active: {patch_hw[0]}x{patch_hw[1]} (training default is 16x16)."
    )

    step4_summary = pl.DataFrame(
        [
            {
                "patches_a_shape": str(tuple(int(v) for v in batch["patches_a"].shape)),
                "positions_a_shape": str(tuple(int(v) for v in batch["positions_a"].shape)),
                "patches_b_shape": str(tuple(int(v) for v in batch["patches_b"].shape)),
                "positions_b_shape": str(tuple(int(v) for v in batch["positions_b"].shape)),
                "center_distance_mm_shape": str(tuple(int(v) for v in batch["center_distance_mm"].shape)),
                "rotation_delta_deg_shape": str(tuple(int(v) for v in batch["rotation_delta_deg"].shape)),
                "window_delta_shape": str(tuple(int(v) for v in batch["window_delta"].shape)),
                "series_label": int(batch["series_label"][0].item()),
            }
        ]
    )

    pos_a = np.asarray(batch["positions_a"][0].cpu().numpy(), dtype=np.float32)
    pos_b = np.asarray(batch["positions_b"][0].cpu().numpy(), dtype=np.float32)
    n_preview = min(20, int(pos_a.shape[0]))
    pos_preview = pl.DataFrame(
        {
            "idx": np.arange(n_preview, dtype=np.int64),
            "A_x_mm": pos_a[:n_preview, 0],
            "A_y_mm": pos_a[:n_preview, 1],
            "A_z_mm": pos_a[:n_preview, 2],
            "B_x_mm": pos_b[:n_preview, 0],
            "B_y_mm": pos_b[:n_preview, 1],
            "B_z_mm": pos_b[:n_preview, 2],
        }
    )

    mo.vstack(
        [
            mo.md("## Step 4: Model Inputs (exact training batch contract)"),
            mo.callout(
                f"These tensors are created via shared `build_dataset_item` + `collate_prism_batch`, same path as training. {patch_size_note}",
                kind="info",
            ),
            step4_summary,
            mo.md("First 20 relative position vectors from collated batch"),
            pos_preview,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
