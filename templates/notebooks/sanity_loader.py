import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


with app.setup:
    import sys
    import time
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.data import (
        load_catalog,
        load_nifti_scan,
        sample_scan_candidates,
    )
    from prism_ssl.config.schema import ScanRecord

    _AXIS_LABELS = {0: "x(R)", 1: "y(A)", 2: "z(S)"}

    def window_to_rgb(slice_2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
        ww_safe = max(float(ww), 1e-6)
        w_min = float(wc) - 0.5 * ww_safe
        w_max = float(wc) + 0.5 * ww_safe
        clipped = np.clip(slice_2d, w_min, w_max)
        gray = ((clipped - w_min) / max(w_max - w_min, 1e-6) * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def draw_cross(img: np.ndarray, row: int, col: int, color: tuple, radius: int = 3) -> None:
        h, w = img.shape[:2]
        if 0 <= row < h and 0 <= col < w:
            img[max(0, row - radius):min(h, row + radius + 1), col] = color
            img[row, max(0, col - radius):min(w, col + radius + 1)] = color

    def draw_dot(img: np.ndarray, row: int, col: int, color: tuple, radius: int = 2) -> None:
        h, w = img.shape[:2]
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    r, c = row + dr, col + dc
                    if 0 <= r < h and 0 <= c < w:
                        img[r, c] = color

    def draw_rect(img: np.ndarray, row: int, col: int, half_h: int, half_w: int, color: tuple) -> None:
        h, w = img.shape[:2]
        r0, r1 = max(0, row - half_h), min(h - 1, row + half_h)
        c0, c1 = max(0, col - half_w), min(w - 1, col + half_w)
        img[r0:r1 + 1, c0] = color
        img[r0:r1 + 1, c1] = color
        img[r0, c0:c1 + 1] = color
        img[r1, c0:c1 + 1] = color

    def patch_grid(patches: np.ndarray, max_patches: int = 64, cols: int = 8) -> np.ndarray:
        n = min(patches.shape[0], max_patches)
        rows = (n + cols - 1) // cols
        h, w = patches.shape[1], patches.shape[2]
        grid = np.zeros((rows * h, cols * w), dtype=np.float32)
        for i in range(n):
            r, c = divmod(i, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = patches[i]
        lo, hi = float(grid.min()), float(grid.max())
        if hi - lo < 1e-6:
            hi = lo + 1.0
        gray = ((grid - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return gray

    def resize_rgb(rgb: np.ndarray, max_dim: int = 600) -> np.ndarray:
        h, w = rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            return np.array(Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS))
        return rgb


@app.cell
def catalog_controls():
    catalog_path = mo.ui.text(
        value="data/pmbb_catalog.csv.gz",
        label="Catalog path",
    )
    modality = mo.ui.dropdown(
        options=["CT,MR", "CT", "MR"],
        value="CT,MR",
        label="Modality filter",
    )
    n_scans = mo.ui.slider(10, 5000, value=200, label="Max scans to sample")
    mo.vstack([
        mo.md("## 1. Catalog"),
        mo.hstack([catalog_path, modality, n_scans]),
    ])
    return catalog_path, modality, n_scans


@app.cell
def load_records(catalog_path, modality, n_scans):
    _t0 = time.perf_counter()
    catalog_df = load_catalog(catalog_path.value)
    _mods = tuple(m.strip() for m in modality.value.split(","))
    records = sample_scan_candidates(catalog_df, n_scans=n_scans.value, seed=42, modality_filter=_mods)
    _dt = time.perf_counter() - _t0
    mo.md(f"Loaded **{len(records)}** scan records from catalog ({len(catalog_df)} total rows) in {_dt:.2f}s")
    return records,


@app.cell
def scan_picker(records):
    scan_idx = mo.ui.slider(0, max(len(records) - 1, 0), value=0, label="Scan index")
    patch_mm = mo.ui.slider(8.0, 32.0, value=16.0, step=1.0, label="Patch size (mm)")
    mo.vstack([
        mo.md("## 2. Load Scan"),
        mo.hstack([scan_idx, patch_mm]),
    ])
    return scan_idx, patch_mm


@app.cell
def load_scan(records, scan_idx, patch_mm):
    _t0 = time.perf_counter()
    _rec = records[scan_idx.value]
    scan, nifti_path = load_nifti_scan(_rec, base_patch_mm=patch_mm.value)
    _load_ms = (time.perf_counter() - _t0) * 1000.0

    scan_geo = scan.geometry
    _scan_info = pl.DataFrame({
        "field": [
            "path", "modality", "shape_vox", "spacing_mm", "extent_mm",
            "acquisition_plane", "thin_axis", "patch_shape_vox", "load_time_ms",
        ],
        "value": [
            nifti_path, scan.modality,
            str(scan_geo.shape_vox), str(tuple(f"{v:.2f}" for v in scan_geo.spacing_mm)),
            str(tuple(f"{v:.1f}" for v in scan_geo.extent_mm)),
            scan_geo.acquisition_plane, f"{scan_geo.thin_axis} ({scan_geo.thin_axis_name})",
            str(tuple(scan.patch_shape_vox.tolist())),
            f"{_load_ms:.1f}",
        ],
    })
    mo.vstack([mo.ui.table(_scan_info, label="Scan geometry")])
    return scan, nifti_path, scan_geo


@app.cell
def slice_browser(scan, scan_geo):
    mo.md("## 3. Scroll Through Scan")

    _thin = scan_geo.thin_axis
    _in_plane = [i for i in range(3) if i != _thin]
    _max_slice = scan.data.shape[_thin] - 1

    slice_slider = mo.ui.slider(0, _max_slice, value=_max_slice // 2, label=f"Slice along {_AXIS_LABELS[_thin]}")
    browser_wc = mo.ui.slider(
        float(scan.robust_low), float(scan.robust_high),
        value=float(scan.robust_median), step=1.0, label="Window center",
    )
    browser_ww = mo.ui.slider(
        1.0, float(6.0 * scan.robust_std),
        value=float(4.0 * scan.robust_std), step=1.0, label="Window width",
    )

    _vol_slice = np.take(scan.data, indices=slice_slider.value, axis=_thin)
    _browse_rgb = window_to_rgb(_vol_slice, browser_wc.value, browser_ww.value)
    _browse_rgb = resize_rgb(_browse_rgb, max_dim=500)

    mo.vstack([
        mo.hstack([slice_slider, browser_wc, browser_ww]),
        mo.md(f"**{scan_geo.acquisition_plane}** plane | "
               f"slice {slice_slider.value}/{_max_slice} along {_AXIS_LABELS[_thin]} | "
               f"rows={_AXIS_LABELS[_in_plane[0]]}, cols={_AXIS_LABELS[_in_plane[1]]}"),
        mo.image(Image.fromarray(_browse_rgb)),
    ])
    return


@app.cell
def sample_controls(scan):
    n_patches = mo.ui.slider(1, 64, value=16, step=1, label="N patches")
    sample_seed = mo.ui.number(value=42, label="Seed")
    sample_wc = mo.ui.slider(
        float(scan.robust_low), float(scan.robust_high),
        value=float(scan.robust_median), step=1.0, label="Window center",
    )
    sample_ww = mo.ui.slider(
        1.0, float(6.0 * scan.robust_std),
        value=float(4.0 * scan.robust_std), step=1.0, label="Window width",
    )
    sample_radius = mo.ui.slider(5.0, 100.0, value=25.0, step=1.0, label="Sampling radius (mm)")
    mo.vstack([
        mo.md("## 4. Sample Patches"),
        mo.hstack([n_patches, sample_seed, sample_radius]),
        mo.hstack([sample_wc, sample_ww]),
    ])
    return n_patches, sample_seed, sample_wc, sample_ww, sample_radius


@app.cell
def do_sample(scan, n_patches, sample_seed, sample_wc, sample_ww, sample_radius):
    _t0 = time.perf_counter()
    result = scan.train_sample(
        n_patches.value,
        seed=int(sample_seed.value),
        wc=sample_wc.value,
        ww=sample_ww.value,
        sampling_radius_mm=sample_radius.value,
    )
    sample_time_ms = (time.perf_counter() - _t0) * 1000.0

    mo.md(
        f"Extracted **{n_patches.value}** patches in **{sample_time_ms:.2f} ms** "
        f"({n_patches.value / max(sample_time_ms, 0.001) * 1000:.0f} patches/sec)\n\n"
        f"Window: wc={result['wc']:.1f}, ww={result['ww']:.1f} | "
        f"Sampling radius: {result['sampling_radius_mm']:.1f} mm | "
        f"Plane: {result['native_acquisition_plane']}"
    )
    return result, sample_time_ms


@app.cell
def slice_overlay(scan, result, scan_geo):
    mo.md("## 5. Slice Overlay")

    _thin = scan_geo.thin_axis
    _in_plane = [i for i in range(3) if i != _thin]
    _ax_r, _ax_c = _in_plane[0], _in_plane[1]

    _prism_vox = result["prism_center_vox"]
    _overlay_idx = int(_prism_vox[_thin])
    _vol_slice = np.take(scan.data, indices=_overlay_idx, axis=_thin)

    _wc, _ww = result["wc"], result["ww"]
    _overlay_rgb = window_to_rgb(_vol_slice, _wc, _ww)

    draw_cross(_overlay_rgb, int(_prism_vox[_ax_r]), int(_prism_vox[_ax_c]), (255, 0, 0), radius=5)

    _centers_vox = result["patch_centers_vox"]
    for _cv in _centers_vox:
        draw_dot(_overlay_rgb, int(_cv[_ax_r]), int(_cv[_ax_c]), (255, 255, 0), radius=2)

    _overlay_rgb = resize_rgb(_overlay_rgb, max_dim=600)

    mo.vstack([
        mo.md(f"**{scan_geo.acquisition_plane}** slice at {_AXIS_LABELS[_thin]}={_overlay_idx} "
               f"| Red cross = prism center, Yellow dots = patch centers "
               f"| rows={_AXIS_LABELS[_ax_r]}, cols={_AXIS_LABELS[_ax_c]}"),
        mo.image(Image.fromarray(_overlay_rgb)),
    ])
    return


@app.cell
def patch_explorer(scan, result, scan_geo):
    mo.md("## 6. Patch Explorer")

    _patches = result["normalized_patches"]
    _centers_vox = result["patch_centers_vox"]
    _prism_vox = result["prism_center_vox"]
    _pos_rel = result["relative_patch_centers_pt"]
    _pos_world = result["patch_centers_pt"]
    _total = _patches.shape[0]

    patch_idx = mo.ui.slider(0, max(_total - 1, 0), value=0, step=1, label="Patch index")

    _thin = scan_geo.thin_axis
    _in_plane = [i for i in range(3) if i != _thin]
    _ax_r, _ax_c = _in_plane[0], _in_plane[1]

    # Render the selected patch enlarged
    _p = _patches[patch_idx.value]
    _lo, _hi = float(_p.min()), float(_p.max())
    if _hi - _lo < 1e-6:
        _hi = _lo + 1.0
    _patch_gray = ((_p - _lo) / (_hi - _lo) * 255.0).astype(np.uint8)
    _patch_big = np.array(Image.fromarray(_patch_gray).resize((128, 128), Image.NEAREST))

    # Render slice with this patch highlighted
    _cv = _centers_vox[patch_idx.value]
    _explore_slice_idx = int(_cv[_thin])
    _vol_slice = np.take(scan.data, indices=_explore_slice_idx, axis=_thin)
    _explore_rgb = window_to_rgb(_vol_slice, result["wc"], result["ww"])

    # Draw prism center
    draw_cross(_explore_rgb, int(_prism_vox[_ax_r]), int(_prism_vox[_ax_c]), (255, 0, 0), radius=5)

    # Draw all other patches as dim dots
    for _i, _ov in enumerate(_centers_vox):
        if _i != patch_idx.value:
            draw_dot(_explore_rgb, int(_ov[_ax_r]), int(_ov[_ax_c]), (100, 100, 0), radius=1)

    # Draw selected patch center and bounding box
    _half_h = _patches.shape[1] // 2
    _half_w = _patches.shape[2] // 2
    draw_dot(_explore_rgb, int(_cv[_ax_r]), int(_cv[_ax_c]), (0, 255, 0), radius=3)
    draw_rect(_explore_rgb, int(_cv[_ax_r]), int(_cv[_ax_c]), _half_h, _half_w, (0, 255, 0))

    _explore_rgb = resize_rgb(_explore_rgb, max_dim=500)

    _rel = _pos_rel[patch_idx.value]
    _world = _pos_world[patch_idx.value]
    _vox = _centers_vox[patch_idx.value]

    mo.vstack([
        patch_idx,
        mo.md(f"**Patch {patch_idx.value}/{_total - 1}** | "
               f"Vox: ({_vox[0]}, {_vox[1]}, {_vox[2]}) | "
               f"World: ({_world[0]:.1f}, {_world[1]:.1f}, {_world[2]:.1f}) mm | "
               f"Rel: ({_rel[0]:.1f}, {_rel[1]:.1f}, {_rel[2]:.1f}) mm | "
               f"Dist: {np.linalg.norm(_rel):.1f} mm"),
        mo.hstack([
            mo.vstack([mo.md("**Selected patch** (128x128 nearest)"), mo.image(Image.fromarray(_patch_big))]),
            mo.vstack([
                mo.md(f"**Slice {_explore_slice_idx}** — "
                       f"green box = selected patch, red cross = prism center, "
                       f"dim dots = other patches"),
                mo.image(Image.fromarray(_explore_rgb)),
            ]),
        ]),
    ])
    return


@app.cell
def show_patch_grid(result):
    mo.md("## 7. Patch Grid")

    _patches = result["normalized_patches"]
    _grid_img = patch_grid(_patches, max_patches=64, cols=8)

    _coord_range = np.ptp(result["relative_patch_centers_pt"], axis=0)
    mo.vstack([
        mo.md(f"Showing {min(64, _patches.shape[0])} of {_patches.shape[0]} patches "
               f"(shape per patch: {_patches.shape[1]}x{_patches.shape[2]})\n\n"
               f"Position range (mm): x={_coord_range[0]:.1f}, y={_coord_range[1]:.1f}, z={_coord_range[2]:.1f}"),
        mo.image(Image.fromarray(_grid_img)),
    ])
    return


@app.cell
def position_scatter(result):
    mo.md("## 8. Position Scatter (World mm, relative to prism center)")

    _pos = result["relative_patch_centers_pt"]
    _norm = np.linalg.norm(_pos, axis=1)

    _scatter_df = pl.DataFrame({
        "x_mm": _pos[:, 0].tolist(),
        "y_mm": _pos[:, 1].tolist(),
        "z_mm": _pos[:, 2].tolist(),
        "dist_mm": _norm.tolist(),
    })

    _base = alt.Chart(_scatter_df.to_pandas()).mark_circle(size=30, opacity=0.7)

    _xy = _base.encode(
        x=alt.X("x_mm:Q", title="X (R) mm"),
        y=alt.Y("y_mm:Q", title="Y (A) mm"),
        color=alt.Color("dist_mm:Q", scale=alt.Scale(scheme="viridis"), title="Distance (mm)"),
    ).properties(title="Axial (XY)", width=250, height=250)

    _xz = _base.encode(
        x=alt.X("x_mm:Q", title="X (R) mm"),
        y=alt.Y("z_mm:Q", title="Z (S) mm"),
        color=alt.Color("dist_mm:Q", scale=alt.Scale(scheme="viridis"), title="Distance (mm)"),
    ).properties(title="Coronal (XZ)", width=250, height=250)

    _yz = _base.encode(
        x=alt.X("y_mm:Q", title="Y (A) mm"),
        y=alt.Y("z_mm:Q", title="Z (S) mm"),
        color=alt.Color("dist_mm:Q", scale=alt.Scale(scheme="viridis"), title="Distance (mm)"),
    ).properties(title="Sagittal (YZ)", width=250, height=250)

    mo.ui.altair_chart(_xy | _xz | _yz)
    return


@app.cell
def pair_view(scan, n_patches, sample_seed, sample_wc, sample_ww, sample_radius):
    mo.md("## 9. Pair View Comparison")

    _seed_a = int(sample_seed.value) * 2
    _seed_b = int(sample_seed.value) * 2 + 1

    _t0 = time.perf_counter()
    _res_a = scan.train_sample(n_patches.value, seed=_seed_a, wc=sample_wc.value, ww=sample_ww.value, sampling_radius_mm=sample_radius.value)
    _res_b = scan.train_sample(n_patches.value, seed=_seed_b, wc=sample_wc.value, ww=sample_ww.value, sampling_radius_mm=sample_radius.value)
    _pair_dt_ms = (time.perf_counter() - _t0) * 1000.0

    _center_a = _res_a["prism_center_pt"]
    _center_b = _res_b["prism_center_pt"]
    _delta_mm = _center_b - _center_a
    _distance_mm = float(np.linalg.norm(_delta_mm))
    _window_delta = np.array([_res_b["wc"] - _res_a["wc"], _res_b["ww"] - _res_a["ww"]])

    _pair_info = pl.DataFrame({
        "metric": ["center_delta_mm (x,y,z)", "center_distance_mm", "window_delta (wc, ww)", "pair_sample_time_ms"],
        "value": [
            f"({_delta_mm[0]:.2f}, {_delta_mm[1]:.2f}, {_delta_mm[2]:.2f})",
            f"{_distance_mm:.2f}",
            f"({_window_delta[0]:.1f}, {_window_delta[1]:.1f})",
            f"{_pair_dt_ms:.2f}",
        ],
    })

    _grid_a = patch_grid(_res_a["normalized_patches"], max_patches=32, cols=8)
    _grid_b = patch_grid(_res_b["normalized_patches"], max_patches=32, cols=8)

    mo.vstack([
        mo.ui.table(_pair_info, label="Pair targets"),
        mo.hstack([
            mo.vstack([mo.md("**View A**"), mo.image(Image.fromarray(_grid_a))]),
            mo.vstack([mo.md("**View B**"), mo.image(Image.fromarray(_grid_b))]),
        ]),
    ])
    return


@app.cell
def bench_controls():
    run_bench = mo.ui.button(label="Run benchmark (100 samples from 1 scan)")
    mo.vstack([
        mo.md("## 10. Extraction Benchmark"),
        mo.hstack([run_bench]),
    ])
    return run_bench,


@app.cell
def run_benchmark(run_bench, records, scan_idx, patch_mm, n_patches):
    run_bench

    _rec = records[scan_idx.value]
    _scan, _ = load_nifti_scan(_rec, base_patch_mm=patch_mm.value)

    _n_iters = 100
    _times_list = []
    for _i in range(_n_iters):
        _t0 = time.perf_counter()
        _scan.train_sample(n_patches.value, seed=_i)
        _times_list.append((time.perf_counter() - _t0) * 1000.0)

    _times_arr = np.array(_times_list)
    _total_patches = _n_iters * n_patches.value
    _total_time_s = _times_arr.sum() / 1000.0

    _bench_stats = pl.DataFrame({
        "metric": [
            "samples", "patches_per_sample", "total_patches",
            "mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms",
            "throughput_patches_per_sec",
        ],
        "value": [
            str(_n_iters), str(n_patches.value), str(_total_patches),
            f"{_times_arr.mean():.2f}", f"{np.median(_times_arr):.2f}",
            f"{np.percentile(_times_arr, 95):.2f}", f"{np.percentile(_times_arr, 99):.2f}",
            f"{_times_arr.min():.2f}", f"{_times_arr.max():.2f}",
            f"{_total_patches / _total_time_s:.0f}",
        ],
    })

    _hist_df = pl.DataFrame({"time_ms": _times_list})
    _hist_chart = alt.Chart(_hist_df.to_pandas()).mark_bar().encode(
        x=alt.X("time_ms:Q", bin=alt.Bin(maxbins=30), title="Sample time (ms)"),
        y=alt.Y("count()", title="Count"),
    ).properties(width=400, height=200, title="Sample time distribution")

    mo.vstack([
        mo.ui.table(_bench_stats, label="Benchmark results"),
        mo.ui.altair_chart(_hist_chart),
    ])
    return


@app.cell
def coordinate_table(result):
    mo.md("## 11. Patch Coordinate Table")

    _pos_world = result["patch_centers_pt"]
    _pos_rel = result["relative_patch_centers_pt"]
    _pos_vox = result["patch_centers_vox"]
    _n = min(_pos_world.shape[0], 20)

    _coord_rows = {
        "idx": list(range(_n)),
        "vox_x": [int(_pos_vox[i, 0]) for i in range(_n)],
        "vox_y": [int(_pos_vox[i, 1]) for i in range(_n)],
        "vox_z": [int(_pos_vox[i, 2]) for i in range(_n)],
        "world_x_mm": [round(float(_pos_world[i, 0]), 2) for i in range(_n)],
        "world_y_mm": [round(float(_pos_world[i, 1]), 2) for i in range(_n)],
        "world_z_mm": [round(float(_pos_world[i, 2]), 2) for i in range(_n)],
        "rel_x_mm": [round(float(_pos_rel[i, 0]), 2) for i in range(_n)],
        "rel_y_mm": [round(float(_pos_rel[i, 1]), 2) for i in range(_n)],
        "rel_z_mm": [round(float(_pos_rel[i, 2]), 2) for i in range(_n)],
        "dist_mm": [round(float(np.linalg.norm(_pos_rel[i])), 2) for i in range(_n)],
    }
    _coord_df = pl.DataFrame(_coord_rows)
    mo.ui.table(_coord_df, label=f"First {_n} patches (of {_pos_world.shape[0]})")
    return


if __name__ == "__main__":
    app.run()
