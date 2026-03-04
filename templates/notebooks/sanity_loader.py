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

    def patch_grid(patches: np.ndarray, max_patches: int = 64, cols: int = 8) -> np.ndarray:
        n_show = min(patches.shape[0], max_patches)
        rows = (n_show + cols - 1) // cols
        h, w = patches.shape[1], patches.shape[2]
        grid = np.zeros((rows * h, cols * w), dtype=np.float32)
        for i in range(n_show):
            r, c = divmod(i, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = patches[i]
        lo, hi = float(grid.min()), float(grid.max())
        if hi - lo < 1e-6:
            hi = lo + 1.0
        gray = ((grid - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return gray


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
    mods = tuple(m.strip() for m in modality.value.split(","))
    records = sample_scan_candidates(catalog_df, n_scans=n_scans.value, seed=42, modality_filter=mods)
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
    scan_info = pl.DataFrame({
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
    mo.vstack([mo.ui.table(scan_info, label="Scan geometry")])
    return scan, nifti_path, scan_geo


@app.cell
def sample_controls():
    n_patches = mo.ui.slider(16, 1024, value=64, step=16, label="N patches")
    sample_seed = mo.ui.number(value=42, label="Seed")
    mo.vstack([
        mo.md("## 3. Sample Patches"),
        mo.hstack([n_patches, sample_seed]),
    ])
    return n_patches, sample_seed


@app.cell
def do_sample(scan, n_patches, sample_seed):
    _t0 = time.perf_counter()
    result = scan.train_sample(n_patches.value, seed=int(sample_seed.value))
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
    mo.md("## 4. Slice Overlay")

    thin = scan_geo.thin_axis
    in_plane = [i for i in range(3) if i != thin]
    ax_r, ax_c = in_plane[0], in_plane[1]

    prism_vox = result["prism_center_vox"]
    slice_idx = int(prism_vox[thin])
    vol_slice = np.take(scan.data, indices=slice_idx, axis=thin)

    wc, ww = result["wc"], result["ww"]
    rgb = window_to_rgb(vol_slice, wc, ww)

    draw_cross(rgb, int(prism_vox[ax_r]), int(prism_vox[ax_c]), (255, 0, 0), radius=5)

    centers_vox = result["patch_centers_vox"]
    for cv in centers_vox:
        draw_dot(rgb, int(cv[ax_r]), int(cv[ax_c]), (255, 255, 0), radius=2)

    h, w = rgb.shape[:2]
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        rgb = np.array(Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS))

    plane_name = scan_geo.acquisition_plane
    axis_labels = {0: "x(R)", 1: "y(A)", 2: "z(S)"}
    mo.vstack([
        mo.md(f"**{plane_name}** slice at {axis_labels[thin]}={slice_idx} "
               f"| rows={axis_labels[ax_r]}, cols={axis_labels[ax_c]}"),
        mo.image(Image.fromarray(rgb)),
    ])
    return


@app.cell
def show_patch_grid(result):
    mo.md("## 5. Patch Grid")

    patches = result["normalized_patches"]
    grid_img = patch_grid(patches, max_patches=64, cols=8)

    coords = result["relative_patch_centers_pt"]
    coord_range = np.ptp(coords, axis=0)
    mo.vstack([
        mo.md(f"Showing {min(64, patches.shape[0])} of {patches.shape[0]} patches "
               f"(shape per patch: {patches.shape[1]}x{patches.shape[2]})\n\n"
               f"Position range (mm): x={coord_range[0]:.1f}, y={coord_range[1]:.1f}, z={coord_range[2]:.1f}"),
        mo.image(Image.fromarray(grid_img)),
    ])
    return


@app.cell
def position_scatter(result):
    mo.md("## 6. Position Scatter (World mm, relative to prism center)")

    pos = result["relative_patch_centers_pt"]
    norm = np.linalg.norm(pos, axis=1)

    scatter_df = pl.DataFrame({
        "x_mm": pos[:, 0].tolist(),
        "y_mm": pos[:, 1].tolist(),
        "z_mm": pos[:, 2].tolist(),
        "dist_mm": norm.tolist(),
    })

    base = alt.Chart(scatter_df.to_pandas()).mark_circle(size=30, opacity=0.7)

    xy = base.encode(
        x=alt.X("x_mm:Q", title="X (R) mm"),
        y=alt.Y("y_mm:Q", title="Y (A) mm"),
        color=alt.Color("dist_mm:Q", scale=alt.Scale(scheme="viridis"), title="Distance (mm)"),
    ).properties(title="Axial (XY)", width=250, height=250)

    xz = base.encode(
        x=alt.X("x_mm:Q", title="X (R) mm"),
        y=alt.Y("z_mm:Q", title="Z (S) mm"),
        color=alt.Color("dist_mm:Q", scale=alt.Scale(scheme="viridis"), title="Distance (mm)"),
    ).properties(title="Coronal (XZ)", width=250, height=250)

    yz = base.encode(
        x=alt.X("y_mm:Q", title="Y (A) mm"),
        y=alt.Y("z_mm:Q", title="Z (S) mm"),
        color=alt.Color("dist_mm:Q", scale=alt.Scale(scheme="viridis"), title="Distance (mm)"),
    ).properties(title="Sagittal (YZ)", width=250, height=250)

    mo.ui.altair_chart(xy | xz | yz)
    return


@app.cell
def pair_view(scan, n_patches, sample_seed):
    mo.md("## 7. Pair View Comparison")

    seed_a = int(sample_seed.value) * 2
    seed_b = int(sample_seed.value) * 2 + 1

    _t0 = time.perf_counter()
    res_a = scan.train_sample(n_patches.value, seed=seed_a)
    res_b = scan.train_sample(n_patches.value, seed=seed_b)
    pair_dt_ms = (time.perf_counter() - _t0) * 1000.0

    center_a = res_a["prism_center_pt"]
    center_b = res_b["prism_center_pt"]
    delta_mm = center_b - center_a
    distance_mm = float(np.linalg.norm(delta_mm))
    window_delta = np.array([res_b["wc"] - res_a["wc"], res_b["ww"] - res_a["ww"]])

    pair_info = pl.DataFrame({
        "metric": ["center_delta_mm (x,y,z)", "center_distance_mm", "window_delta (wc, ww)", "pair_sample_time_ms"],
        "value": [
            f"({delta_mm[0]:.2f}, {delta_mm[1]:.2f}, {delta_mm[2]:.2f})",
            f"{distance_mm:.2f}",
            f"({window_delta[0]:.1f}, {window_delta[1]:.1f})",
            f"{pair_dt_ms:.2f}",
        ],
    })

    grid_a = patch_grid(res_a["normalized_patches"], max_patches=32, cols=8)
    grid_b = patch_grid(res_b["normalized_patches"], max_patches=32, cols=8)

    mo.vstack([
        mo.ui.table(pair_info, label="Pair targets"),
        mo.hstack([
            mo.vstack([mo.md("**View A**"), mo.image(Image.fromarray(grid_a))]),
            mo.vstack([mo.md("**View B**"), mo.image(Image.fromarray(grid_b))]),
        ]),
    ])
    return


@app.cell
def bench_controls():
    run_bench = mo.ui.button(label="Run benchmark (100 samples from 1 scan)")
    mo.vstack([
        mo.md("## 8. Extraction Benchmark"),
        mo.hstack([run_bench]),
    ])
    return run_bench,


@app.cell
def run_benchmark(run_bench, records, scan_idx, patch_mm, n_patches):
    run_bench

    _rec = records[scan_idx.value]
    _scan, _ = load_nifti_scan(_rec, base_patch_mm=patch_mm.value)

    n_iters = 100
    times_list = []
    for i in range(n_iters):
        _t0 = time.perf_counter()
        _scan.train_sample(n_patches.value, seed=i)
        times_list.append((time.perf_counter() - _t0) * 1000.0)

    times_arr = np.array(times_list)
    total_patches = n_iters * n_patches.value
    total_time_s = times_arr.sum() / 1000.0

    bench_stats = pl.DataFrame({
        "metric": [
            "samples", "patches_per_sample", "total_patches",
            "mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms",
            "throughput_patches_per_sec",
        ],
        "value": [
            str(n_iters), str(n_patches.value), str(total_patches),
            f"{times_arr.mean():.2f}", f"{np.median(times_arr):.2f}",
            f"{np.percentile(times_arr, 95):.2f}", f"{np.percentile(times_arr, 99):.2f}",
            f"{times_arr.min():.2f}", f"{times_arr.max():.2f}",
            f"{total_patches / total_time_s:.0f}",
        ],
    })

    hist_df = pl.DataFrame({"time_ms": times_list})
    hist_chart = alt.Chart(hist_df.to_pandas()).mark_bar().encode(
        x=alt.X("time_ms:Q", bin=alt.Bin(maxbins=30), title="Sample time (ms)"),
        y=alt.Y("count()", title="Count"),
    ).properties(width=400, height=200, title="Sample time distribution")

    mo.vstack([
        mo.ui.table(bench_stats, label="Benchmark results"),
        mo.ui.altair_chart(hist_chart),
    ])
    return


@app.cell
def coordinate_table(result):
    mo.md("## 9. Patch Coordinate Table")

    pos_world = result["patch_centers_pt"]
    pos_rel = result["relative_patch_centers_pt"]
    pos_vox = result["patch_centers_vox"]
    n_show = min(pos_world.shape[0], 20)

    coord_rows = {
        "idx": list(range(n_show)),
        "vox_x": [int(pos_vox[i, 0]) for i in range(n_show)],
        "vox_y": [int(pos_vox[i, 1]) for i in range(n_show)],
        "vox_z": [int(pos_vox[i, 2]) for i in range(n_show)],
        "world_x_mm": [round(float(pos_world[i, 0]), 2) for i in range(n_show)],
        "world_y_mm": [round(float(pos_world[i, 1]), 2) for i in range(n_show)],
        "world_z_mm": [round(float(pos_world[i, 2]), 2) for i in range(n_show)],
        "rel_x_mm": [round(float(pos_rel[i, 0]), 2) for i in range(n_show)],
        "rel_y_mm": [round(float(pos_rel[i, 1]), 2) for i in range(n_show)],
        "rel_z_mm": [round(float(pos_rel[i, 2]), 2) for i in range(n_show)],
        "dist_mm": [round(float(np.linalg.norm(pos_rel[i])), 2) for i in range(n_show)],
    }
    coord_df = pl.DataFrame(coord_rows)
    mo.ui.table(coord_df, label=f"First {n_show} patches (of {pos_world.shape[0]})")
    return


if __name__ == "__main__":
    app.run()
