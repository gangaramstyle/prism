"""Profile data loader bottlenecks: NIfTI I/O, patch extraction, resize, collation."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism_ssl.data import load_catalog, load_nifti_scan, sample_scan_candidates


def profile_single_scan(record, base_patch_mm: float = 20.0, n_patches: int = 256, n_iters: int = 20):
    # Time NIfTI load
    t0 = time.perf_counter()
    scan, path = load_nifti_scan(record, base_patch_mm=base_patch_mm)
    load_ms = (time.perf_counter() - t0) * 1000

    geo = scan.geometry
    patch_vox = scan._mm_patch_vox_shape(base_patch_mm)
    in_plane = [i for i in range(3) if i != geo.thin_axis]
    native_h = int(patch_vox[in_plane[0]])
    native_w = int(patch_vox[in_plane[1]])

    print(f"\n{'='*60}")
    print(f"Scan: {path}")
    print(f"Shape: {scan.data.shape}, Spacing: {tuple(f'{v:.2f}' for v in scan.voxel_axis_mm)}")
    print(f"Plane: {geo.acquisition_plane}, Thin: {geo.thin_axis_name}")
    print(f"Native patch vox: {tuple(patch_vox.tolist())} -> in-plane {native_h}x{native_w}")
    print(f"Output: 16x16 (resize {'needed' if native_h != 16 or native_w != 16 else 'skipped'})")
    print(f"NIfTI load: {load_ms:.1f} ms")

    # Time train_sample breakdown
    extract_times = []
    total_times = []
    for i in range(n_iters):
        t0 = time.perf_counter()
        result = scan.train_sample(n_patches, seed=i)
        total_ms = (time.perf_counter() - t0) * 1000
        total_times.append(total_ms)

    total_arr = np.array(total_times)
    print(f"\ntrain_sample({n_patches} patches) x {n_iters} iters:")
    print(f"  mean: {total_arr.mean():.2f} ms")
    print(f"  median: {np.median(total_arr):.2f} ms")
    print(f"  p95: {np.percentile(total_arr, 95):.2f} ms")
    print(f"  patches/sec: {n_patches / (total_arr.mean() / 1000):.0f}")

    # Break down components: center sampling vs extraction vs windowing
    # Time just center sampling
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        centers = []
        prism = scan._sample_center(rng, patch_vox)
        radius = 25.0
        max_attempts = n_patches * 4
        attempts = 0
        while len(centers) < n_patches and attempts < max_attempts:
            delta_mm = rng.uniform(-radius, radius, size=3)
            if float(np.linalg.norm(delta_mm)) > radius:
                delta_mm = delta_mm * (radius / max(np.linalg.norm(delta_mm), 1e-6))
            delta_vox = (scan.affine_linear_inv @ np.asarray(delta_mm, dtype=np.float32)).astype(np.float32)
            center = prism + np.rint(delta_vox).astype(np.int64)
            attempts += 1
            if scan._patch_has_overlap(center, patch_vox):
                centers.append(center)
        while len(centers) < n_patches:
            centers.append(centers[-1] if centers else prism.copy())
    center_ms = (time.perf_counter() - t0) * 1000 / n_iters
    print(f"\n  Center sampling: {center_ms:.2f} ms ({center_ms/total_arr.mean()*100:.0f}%)")

    # Time just extraction + resize
    centers_arr = np.array(centers, dtype=np.int64)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        scan._extract_patches(centers_arr, patch_vox, 16)
    extract_ms = (time.perf_counter() - t0) * 1000 / n_iters
    print(f"  Extract+resize: {extract_ms:.2f} ms ({extract_ms/total_arr.mean()*100:.0f}%)")

    # Time just windowing/normalization
    raw = scan._extract_patches(centers_arr, patch_vox, 16)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        wc = float(scan.robust_median)
        ww = float(4.0 * scan.robust_std)
        w_min = wc - 0.5 * ww
        w_max = wc + 0.5 * ww
        clipped = np.clip(raw, w_min, w_max)
        normalized = ((clipped - w_min) / max(w_max - w_min, 1e-6)) * 2.0 - 1.0
    window_ms = (time.perf_counter() - t0) * 1000 / n_iters
    print(f"  Windowing: {window_ms:.2f} ms ({window_ms/total_arr.mean()*100:.0f}%)")

    # Time just the PIL resize portion
    from PIL import Image
    if native_h != 16 or native_w != 16:
        native = np.zeros((n_patches, native_h, native_w), dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = np.empty((n_patches, 16, 16), dtype=np.float32)
            for j in range(n_patches):
                out[j] = np.array(
                    Image.fromarray(native[j], mode="F").resize((16, 16), Image.BILINEAR),
                    dtype=np.float32,
                )
        resize_ms = (time.perf_counter() - t0) * 1000 / n_iters
        print(f"  PIL resize alone ({native_h}x{native_w}->16x16, {n_patches} patches): {resize_ms:.2f} ms ({resize_ms/total_arr.mean()*100:.0f}%)")

    # Time pair (2x train_sample, like actual training)
    t0 = time.perf_counter()
    for i in range(n_iters):
        scan.train_sample(n_patches, seed=i * 2)
        scan.train_sample(n_patches, seed=i * 2 + 1)
    pair_ms = (time.perf_counter() - t0) * 1000 / n_iters
    print(f"\n  Pair sample (2x): {pair_ms:.2f} ms")
    print(f"  Required rate for batch=8, step=50ms: {8 / 0.050:.0f} pairs/sec")
    print(f"  Actual per-worker: {1000 / pair_ms:.0f} pairs/sec")
    print(f"  With 16 workers: {16 * 1000 / pair_ms:.0f} pairs/sec")


if __name__ == "__main__":
    catalog_path = sys.argv[1] if len(sys.argv) > 1 else "data/pmbb_catalog.csv.gz"
    df = load_catalog(catalog_path)
    records = sample_scan_candidates(df, n_scans=10, seed=42, modality_filter=("CT", "MR"))

    for rec in records[:3]:
        try:
            profile_single_scan(rec)
        except Exception as e:
            print(f"Skipped: {e}")
