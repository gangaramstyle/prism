# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair",
#     "marimo",
#     "nibabel",
#     "numpy",
#     "pillow",
#     "polars",
#     "pyyaml",
#     "timm",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


with app.setup:
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.config import load_run_config_from_flat
    from prism_ssl.model import PrismSSLModel
    from prism_ssl.utils.hashing import stable_int_hash
    from prism_ssl.validation import build_ordered_within_scan_view_pairs, load_ct_view_validation_cache

    alt.data_transformers.disable_max_rows()


@app.cell
def _(Image, Path, PrismSSLModel, build_ordered_within_scan_view_pairs, load_ct_view_validation_cache, load_run_config_from_flat, np, os, pl, stable_int_hash, torch):
    axis_order = ("x", "y", "z", "wc", "ww")
    axis_titles = {
        "x": "Delta X Sign",
        "y": "Delta Y Sign",
        "z": "Delta Z Sign",
        "wc": "Window Center Sign",
        "ww": "Window Width Sign",
    }
    axis_value_columns = {
        "x": "delta_x_mm",
        "y": "delta_y_mm",
        "z": "delta_z_mm",
        "wc": "delta_wc",
        "ww": "delta_ww",
    }

    def default_checkpoint_path() -> str:
        env = str(os.environ.get("PRISM_NOTEBOOK_CHECKPOINT", "")).strip()
        if env:
            return env
        ckpt_root = Path.home() / "prism_ssl" / "checkpoints"
        if not ckpt_root.exists():
            return ""
        candidates = sorted(ckpt_root.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
        return str(candidates[0]) if candidates else ""

    def default_cache_dir() -> str:
        env = str(os.environ.get("PRISM_NOTEBOOK_CACHE_DIR", "")).strip()
        if env:
            return env
        return str((Path.home() / "prism-ssl-validation" / "ct_view_phase1").expanduser())

    def resolve_device(device_key: str) -> torch.device:
        key = str(device_key).strip().lower()
        if key in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(key)

    def build_model_from_checkpoint(checkpoint_path: str | Path, device_key: str) -> tuple[PrismSSLModel, object, dict, torch.device]:
        device = resolve_device(device_key)
        payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=device)
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError("Checkpoint payload is missing a flat config dictionary")
        config = load_run_config_from_flat(config_payload)
        model = PrismSSLModel(
            patch_dim=16 * 16 * 1,
            n_patches=config.data.n_patches,
            model_name=config.model.name,
            d_model=config.model.d_model,
            proj_dim=config.model.proj_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            pos_min_wavelength_mm=config.model.pos_min_wavelength_mm,
            pos_max_wavelength_mm=config.model.pos_max_wavelength_mm,
            mim_mask_ratio=config.model.mim_mask_ratio,
            mim_decoder_layers=config.model.mim_decoder_layers,
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()
        return model, config, payload, device

    def pair_grid_image(patches: np.ndarray, *, max_patches: int = 32, cols: int = 8) -> Image.Image:
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        n = min(int(max_patches), int(arr.shape[0]))
        if n <= 0:
            return Image.fromarray(np.zeros((16, 16), dtype=np.uint8))
        arr = arr[:n]
        rows = (n + cols - 1) // cols
        pad = rows * cols - n
        if pad > 0:
            arr = np.concatenate([arr, np.zeros((pad, arr.shape[1], arr.shape[2]), dtype=arr.dtype)], axis=0)
        row_images = [np.concatenate(arr[row * cols : (row + 1) * cols], axis=1) for row in range(rows)]
        grid = np.concatenate(row_images, axis=0)
        grid[0, 0] = -1.0
        if grid.shape[1] > 1:
            grid[0, 1] = 1.0
        gray = np.clip((grid + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(gray).resize((gray.shape[1] * 4, gray.shape[0] * 4), Image.NEAREST)

    def enrich_pair_metadata(cache: dict[str, object], *, include_self: bool, max_scans: int, max_pairs: int, seed: int) -> pl.DataFrame:
        views_df = cache["views_df"].sort("view_index")
        scan_df = cache["scans_df"].sort("scan_index").head(int(max_scans))
        pair_df = build_ordered_within_scan_view_pairs(cache, include_self=bool(include_self))
        pair_df = pair_df.filter(pl.col("scan_index").is_in(scan_df["scan_index"].to_list()))

        meta_cols = [
            "view_index",
            "scan_id",
            "semantic_target_key",
            "normalized_patch_std",
            "source_patch_mm",
            "series_description",
        ]
        meta_a = views_df.select(meta_cols).rename({col: f"{col}_a" for col in meta_cols if col != "view_index"}).rename({"view_index": "view_index_a"})
        meta_b = views_df.select(meta_cols).rename({col: f"{col}_b" for col in meta_cols if col != "view_index"}).rename({"view_index": "view_index_b"})
        pair_df = (
            pair_df.join(meta_a, on="view_index_a", how="left")
            .join(meta_b, on="view_index_b", how="left")
            .with_columns(
                [
                    pl.col("center_delta_mm").list.get(0).cast(pl.Float64).alias("delta_x_mm"),
                    pl.col("center_delta_mm").list.get(1).cast(pl.Float64).alias("delta_y_mm"),
                    pl.col("center_delta_mm").list.get(2).cast(pl.Float64).alias("delta_z_mm"),
                    pl.col("window_delta").list.get(0).cast(pl.Float64).alias("delta_wc"),
                    pl.col("window_delta").list.get(1).cast(pl.Float64).alias("delta_ww"),
                    pl.max_horizontal(pl.col("source_patch_mm_a"), pl.col("source_patch_mm_b")).alias("max_source_patch_mm"),
                ]
            )
            .with_columns(
                [
                    pl.concat_str(
                        [
                            pl.min_horizontal(pl.col("semantic_target_key_a"), pl.col("semantic_target_key_b")),
                            pl.lit(" vs "),
                            pl.max_horizontal(pl.col("semantic_target_key_a"), pl.col("semantic_target_key_b")),
                        ]
                    ).alias("semantic_pair"),
                ]
            )
            .sort(["scan_index", "view_index_a", "view_index_b"])
        )

        if pair_df.height > int(max_pairs):
            rows = pair_df.to_dicts()
            rows.sort(key=lambda row: stable_int_hash(f"{int(seed)}|{row['scan_index']}|{row['view_index_a']}|{row['view_index_b']}"))
            pair_df = pl.DataFrame(rows[: int(max_pairs)]).sort(["scan_index", "view_index_a", "view_index_b"])
        return pair_df

    def evaluate_pair_relation(*, model: PrismSSLModel, device: torch.device, cache: dict[str, object], pair_df: pl.DataFrame, batch_size: int) -> pl.DataFrame:
        if pair_df.height == 0:
            return pl.DataFrame([])

        view_lookup = {int(view_idx): int(row_idx) for row_idx, view_idx in enumerate(cache["view_index"].cpu().tolist())}
        row_index_a = np.asarray([view_lookup[int(v)] for v in pair_df["view_index_a"].to_list()], dtype=np.int64)
        row_index_b = np.asarray([view_lookup[int(v)] for v in pair_df["view_index_b"].to_list()], dtype=np.int64)

        logits_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, pair_df.height, int(batch_size)):
                stop = min(start + int(batch_size), pair_df.height)
                idx_a = torch.tensor(row_index_a[start:stop], dtype=torch.long)
                idx_b = torch.tensor(row_index_b[start:stop], dtype=torch.long)
                patches_a = cache["normalized_patches"][idx_a].to(device=device, dtype=torch.float32)
                positions_a = cache["relative_patch_centers_pt"][idx_a].to(device=device, dtype=torch.float32)
                patches_b = cache["normalized_patches"][idx_b].to(device=device, dtype=torch.float32)
                positions_b = cache["relative_patch_centers_pt"][idx_b].to(device=device, dtype=torch.float32)
                outputs = model(patches_a, positions_a, patches_b, positions_b)
                logits_chunks.append(outputs.pair_relation_logits.detach().cpu().numpy().astype(np.float32, copy=False))

        logits = np.concatenate(logits_chunks, axis=0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = probs > 0.5
        center_delta = np.asarray(pair_df["center_delta_mm"].to_list(), dtype=np.float32)
        window_delta = np.asarray(pair_df["window_delta"].to_list(), dtype=np.float32)
        targets = np.concatenate([(center_delta > 0.0), (window_delta > 0.0)], axis=1)
        valid = np.concatenate([(np.abs(center_delta) >= 1.0), np.ones_like(window_delta, dtype=bool)], axis=1)

        eval_df = pair_df
        for axis_idx, axis_key in enumerate(axis_order):
            eval_df = eval_df.with_columns(
                [
                    pl.Series(f"prob_{axis_key}", probs[:, axis_idx].tolist(), dtype=pl.Float32),
                    pl.Series(f"pred_{axis_key}", preds[:, axis_idx].tolist(), dtype=pl.Boolean),
                    pl.Series(f"target_{axis_key}", targets[:, axis_idx].tolist(), dtype=pl.Boolean),
                    pl.Series(f"valid_{axis_key}", valid[:, axis_idx].tolist(), dtype=pl.Boolean),
                    pl.Series(f"correct_{axis_key}", ((preds[:, axis_idx] == targets[:, axis_idx]) & valid[:, axis_idx]).tolist(), dtype=pl.Boolean),
                ]
            )
        return eval_df

    def pca_project(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return np.zeros((0, int(n_components)), dtype=np.float32)
        centered = arr - arr.mean(axis=0, keepdims=True)
        u, s, _vh = np.linalg.svd(centered, full_matrices=False)
        rank = min(int(n_components), u.shape[1], s.shape[0])
        proj = u[:, :rank] * s[:rank]
        if rank < int(n_components):
            pad = np.zeros((arr.shape[0], int(n_components) - rank), dtype=np.float32)
            proj = np.concatenate([proj, pad], axis=1)
        return np.asarray(proj, dtype=np.float32)

    def compute_view_embeddings(*, model: PrismSSLModel, device: torch.device, cache: dict[str, object], batch_size: int) -> pl.DataFrame:
        views_df = cache["views_df"].sort("view_index")
        n_views = int(views_df.height)
        if n_views == 0:
            return pl.DataFrame([])

        view_cls_chunks: list[np.ndarray] = []
        series_cls_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, n_views, int(batch_size)):
                stop = min(start + int(batch_size), n_views)
                idx = torch.arange(start, stop, dtype=torch.long)
                patches = cache["normalized_patches"][idx].to(device=device, dtype=torch.float32)
                positions = cache["relative_patch_centers_pt"][idx].to(device=device, dtype=torch.float32)
                outputs = model(patches, positions, patches, positions)
                view_cls_chunks.append(outputs.view_cls_a.detach().cpu().numpy().astype(np.float32, copy=False))
                series_cls_chunks.append(outputs.series_cls_a.detach().cpu().numpy().astype(np.float32, copy=False))

        view_cls = np.concatenate(view_cls_chunks, axis=0)
        series_cls = np.concatenate(series_cls_chunks, axis=0)
        view_proj = pca_project(view_cls, n_components=2)
        series_proj = pca_project(series_cls, n_components=2)

        return views_df.with_columns(
            [
                pl.Series("view_cls_pc1", view_proj[:, 0].tolist(), dtype=pl.Float32),
                pl.Series("view_cls_pc2", view_proj[:, 1].tolist(), dtype=pl.Float32),
                pl.Series("series_cls_pc1", series_proj[:, 0].tolist(), dtype=pl.Float32),
                pl.Series("series_cls_pc2", series_proj[:, 1].tolist(), dtype=pl.Float32),
                pl.Series("view_cls_l2", np.linalg.norm(view_cls, axis=1).tolist(), dtype=pl.Float32),
                pl.Series("series_cls_l2", np.linalg.norm(series_cls, axis=1).tolist(), dtype=pl.Float32),
            ]
        )

    def filtered_eval_df(eval_df: pl.DataFrame, threshold: float, filter_mode: str) -> pl.DataFrame:
        if eval_df.height == 0 or str(filter_mode) == "all":
            return eval_df
        return eval_df.filter((pl.col("normalized_patch_std_a") >= float(threshold)) & (pl.col("normalized_patch_std_b") >= float(threshold)))

    def axis_summary_table(eval_df: pl.DataFrame, threshold: float) -> pl.DataFrame:
        if eval_df.height == 0 or "valid_x" not in eval_df.columns:
            return pl.DataFrame([])
        rows: list[dict[str, object]] = []
        for subset_name in ("all", "filtered"):
            subset = filtered_eval_df(eval_df, threshold, subset_name)
            for axis_key in axis_order:
                valid = subset.filter(pl.col(f"valid_{axis_key}"))
                accuracy = float(valid.select(pl.col(f"correct_{axis_key}").cast(pl.Float64).mean()).item()) if valid.height > 0 else None
                rows.append(
                    {
                        "subset": subset_name,
                        "axis": axis_key,
                        "title": axis_titles[axis_key],
                        "n": int(valid.height),
                        "accuracy": accuracy,
                        "error_rate": (1.0 - accuracy) if accuracy is not None else None,
                    }
                )
        return pl.DataFrame(rows)

    def confusion_table(eval_df: pl.DataFrame, axis_key: str, threshold: float, filter_mode: str) -> pl.DataFrame:
        if eval_df.height == 0 or f"valid_{axis_key}" not in eval_df.columns:
            return pl.DataFrame([])
        subset = filtered_eval_df(eval_df, threshold, filter_mode).filter(pl.col(f"valid_{axis_key}"))
        rows: list[dict[str, object]] = []
        total = max(int(subset.height), 1)
        for target in (False, True):
            target_df = subset.filter(pl.col(f"target_{axis_key}") == target)
            row_total = max(int(target_df.height), 1)
            for pred in (False, True):
                count = int(target_df.filter(pl.col(f"pred_{axis_key}") == pred).height)
                rows.append(
                    {
                        "target": "pos" if target else "neg",
                        "pred": "pos" if pred else "neg",
                        "count": count,
                        "row_rate": float(count / row_total),
                        "overall_rate": float(count / total),
                    }
                )
        return pl.DataFrame(rows)

    def axis_abs_bin_expr(axis_key: str) -> pl.Expr:
        col = axis_value_columns[axis_key]
        if axis_key in {"x", "y", "z"}:
            return (
                pl.when(pl.col(col).abs() < 4.0).then(pl.lit("<4"))
                .when(pl.col(col).abs() < 8.0).then(pl.lit("4-8"))
                .when(pl.col(col).abs() < 16.0).then(pl.lit("8-16"))
                .when(pl.col(col).abs() < 32.0).then(pl.lit("16-32"))
                .otherwise(pl.lit("32+"))
            )
        return (
            pl.when(pl.col(col).abs() < 50.0).then(pl.lit("<50"))
            .when(pl.col(col).abs() < 100.0).then(pl.lit("50-100"))
            .when(pl.col(col).abs() < 200.0).then(pl.lit("100-200"))
            .otherwise(pl.lit("200+"))
        )

    def distance_bin_expr() -> pl.Expr:
        return (
            pl.when(pl.col("center_distance_mm") < 16.0).then(pl.lit("<16"))
            .when(pl.col("center_distance_mm") < 32.0).then(pl.lit("16-32"))
            .when(pl.col("center_distance_mm") < 64.0).then(pl.lit("32-64"))
            .when(pl.col("center_distance_mm") < 128.0).then(pl.lit("64-128"))
            .otherwise(pl.lit("128+"))
        )

    def source_patch_bin_expr() -> pl.Expr:
        return (
            pl.when(pl.col("max_source_patch_mm") < 24.0).then(pl.lit("16-24"))
            .when(pl.col("max_source_patch_mm") < 32.0).then(pl.lit("24-32"))
            .when(pl.col("max_source_patch_mm") < 48.0).then(pl.lit("32-48"))
            .otherwise(pl.lit("48-64"))
        )

    def slice_summary_table(eval_df: pl.DataFrame, *, axis_key: str, threshold: float, filter_mode: str, group_key: str, min_count: int) -> pl.DataFrame:
        if eval_df.height == 0 or f"valid_{axis_key}" not in eval_df.columns:
            return pl.DataFrame([])
        subset = filtered_eval_df(eval_df, threshold, filter_mode).filter(pl.col(f"valid_{axis_key}"))
        if subset.height == 0:
            return pl.DataFrame([])
        subset = subset.with_columns(
            [
                distance_bin_expr().alias("distance_bin"),
                axis_abs_bin_expr(axis_key).alias("axis_abs_bin"),
                source_patch_bin_expr().alias("source_patch_bin"),
                ((pl.col("normalized_patch_std_a") < float(threshold)) | (pl.col("normalized_patch_std_b") < float(threshold))).alias("low_variation_pair"),
            ]
        )
        return (
            subset.group_by(group_key)
            .agg(
                [
                    pl.len().alias("n"),
                    pl.col(f"correct_{axis_key}").cast(pl.Float64).mean().alias("accuracy"),
                    pl.col("center_distance_mm").mean().alias("center_distance_mm_mean"),
                ]
            )
            .filter(pl.col("n") >= int(min_count))
            .with_columns((1.0 - pl.col("accuracy")).alias("error_rate"))
            .sort(["error_rate", "n"], descending=[True, True])
        )

    def error_examples(eval_df: pl.DataFrame, *, axis_key: str, threshold: float, filter_mode: str, mode: str) -> pl.DataFrame:
        if eval_df.height == 0 or f"valid_{axis_key}" not in eval_df.columns:
            return pl.DataFrame([])
        subset = filtered_eval_df(eval_df, threshold, filter_mode).filter(pl.col(f"valid_{axis_key}"))
        if str(mode) == "false_positive":
            subset = subset.filter((pl.col(f"target_{axis_key}") == False) & (pl.col(f"pred_{axis_key}") == True))
            return subset.sort(f"prob_{axis_key}", descending=True)
        subset = subset.filter((pl.col(f"target_{axis_key}") == True) & (pl.col(f"pred_{axis_key}") == False))
        return subset.sort(f"prob_{axis_key}", descending=False)

    def example_pair_images(cache: dict[str, object], row: dict[str, object]) -> tuple[Image.Image, Image.Image]:
        lookup = {int(view_idx): int(row_idx) for row_idx, view_idx in enumerate(cache["view_index"].cpu().tolist())}
        patches_a = cache["normalized_patches"][lookup[int(row["view_index_a"])]] .cpu().numpy()
        patches_b = cache["normalized_patches"][lookup[int(row["view_index_b"])]] .cpu().numpy()
        return pair_grid_image(patches_a), pair_grid_image(patches_b)

    return (
        axis_order,
        axis_titles,
        axis_summary_table,
        build_model_from_checkpoint,
        compute_view_embeddings,
        confusion_table,
        default_cache_dir,
        default_checkpoint_path,
        enrich_pair_metadata,
        error_examples,
        evaluate_pair_relation,
        example_pair_images,
        resolve_device,
        slice_summary_table,
    )


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    intro = mo.md(
        """
# Pair-Relation Checkpoint Probe

This notebook evaluates the current pair-relation head on the offline CT semantic view cache.

Use it to inspect:
- confusion matrices for `(x, y, z, wc, ww)` sign prediction
- before/after low-variation filtering
- failure slices by distance, target magnitude, semantic pair, and source patch size
- concrete false positives and false negatives with patch grids
"""
    )
    intro
    return


@app.cell
def _(axis_order, default_cache_dir, default_checkpoint_path):
    checkpoint_path = mo.ui.text(label="Checkpoint path", value=default_checkpoint_path())
    cache_dir = mo.ui.text(label="Validation cache dir", value=default_cache_dir())
    device_key = mo.ui.dropdown(options=["auto", "cpu", "cuda"], value="auto", label="Device")
    batch_size = mo.ui.slider(16, 512, value=128, step=16, label="Eval batch size")
    max_scans = mo.ui.slider(8, 128, value=128, step=8, label="Max scans")
    max_pairs = mo.ui.slider(1024, 32768, value=32768, step=1024, label="Max ordered pairs")
    include_self = mo.ui.checkbox(label="Include self-pairs", value=False)
    low_variation_threshold = mo.ui.slider(0.0, 0.25, value=0.05, step=0.01, label="Low-variation std threshold")
    axis_key = mo.ui.dropdown(options=list(axis_order), value="x", label="Axis")
    filter_mode = mo.ui.dropdown(options=["all", "filtered"], value="all", label="Subset")
    embedding_stream = mo.ui.dropdown(options=["view_cls", "series_cls"], value="view_cls", label="Embedding stream")
    embedding_color = mo.ui.dropdown(
        options=["semantic_target_key", "series_description", "scan_id", "source_patch_bin", "low_variation"],
        value="semantic_target_key",
        label="Embedding color",
    )
    slice_group = mo.ui.dropdown(
        options=["distance_bin", "axis_abs_bin", "semantic_pair", "source_patch_bin", "semantic_target_key_a", "semantic_target_key_b", "low_variation_pair"],
        value="semantic_pair",
        label="Failure slice group",
    )
    min_group_count = mo.ui.slider(1, 128, value=8, step=1, label="Min slice count")
    example_mode = mo.ui.dropdown(options=["false_positive", "false_negative"], value="false_positive", label="Example mode")
    run_eval = mo.ui.button(label="Load checkpoint and run evaluation")
    mo.vstack(
        [
            mo.hstack([checkpoint_path, cache_dir]),
            mo.hstack([device_key, batch_size, max_scans, max_pairs]),
            mo.hstack([include_self, low_variation_threshold, axis_key, filter_mode]),
            mo.hstack([embedding_stream, embedding_color]),
            mo.hstack([slice_group, min_group_count, example_mode, run_eval]),
        ]
    )
    return axis_key, batch_size, cache_dir, checkpoint_path, device_key, embedding_color, embedding_stream, example_mode, filter_mode, include_self, low_variation_threshold, max_pairs, max_scans, min_group_count, run_eval, slice_group


@app.cell
def _(cache_dir, checkpoint_path, is_script_mode, run_eval):
    should_run = bool(is_script_mode or run_eval.value)
    checkpoint_exists = Path(checkpoint_path.value).expanduser().exists() if checkpoint_path.value.strip() else False
    cache_exists = Path(cache_dir.value).expanduser().exists() if cache_dir.value.strip() else False
    mo.ui.table(
        pl.DataFrame(
            {
                "item": ["should_run", "checkpoint_exists", "cache_exists"],
                "value": [str(should_run), str(checkpoint_exists), str(cache_exists)],
            }
        ),
        label="Input status",
    )
    return cache_exists, checkpoint_exists, should_run


@app.cell
def _(build_model_from_checkpoint, checkpoint_exists, checkpoint_path, device_key, resolve_device, should_run):
    if should_run and checkpoint_exists:
        model, config, payload, device = build_model_from_checkpoint(checkpoint_path.value, device_key.value)
        model_status = {"status": "loaded", "device": str(device), "step": int(payload.get("step", 0)), "n_patches": int(config.data.n_patches)}
    else:
        model = None
        device = resolve_device(device_key.value)
        model_status = {"status": "idle" if not should_run else "missing_checkpoint", "device": str(device), "step": None, "n_patches": None}
    mo.ui.table(pl.DataFrame([model_status]), label="Model status")
    return device, model


@app.cell
def _(cache_dir, cache_exists, should_run):
    if should_run and cache_exists:
        cache = load_ct_view_validation_cache(cache_dir.value)
        cache_status = {"status": "loaded", "n_scans": int(cache["scans_df"].height), "n_views": int(cache["views_df"].height)}
    else:
        cache = None
        cache_status = {"status": "idle" if not should_run else "missing_cache", "n_scans": 0, "n_views": 0}
    mo.ui.table(pl.DataFrame([cache_status]), label="Cache status")
    return (cache,)


@app.cell
def _(batch_size, cache, compute_view_embeddings, device, model):
    if cache is None or model is None:
        embedding_df = pl.DataFrame([])
    else:
        embedding_df = compute_view_embeddings(model=model, device=device, cache=cache, batch_size=int(batch_size.value))
    return (embedding_df,)


@app.cell
def _(alt, embedding_color, embedding_df, embedding_stream, low_variation_threshold, mo, pl):
    if embedding_df.height == 0:
        embedding_panel = mo.md("No embedding projections available yet.")
    else:
        plot_df = embedding_df.with_columns(
            [
                (
                    pl.when(pl.col("source_patch_mm") < 24.0).then(pl.lit("16-24"))
                    .when(pl.col("source_patch_mm") < 32.0).then(pl.lit("24-32"))
                    .when(pl.col("source_patch_mm") < 48.0).then(pl.lit("32-48"))
                    .otherwise(pl.lit("48-64"))
                ).alias("source_patch_bin"),
                (
                    pl.when(pl.col("normalized_patch_std") < float(low_variation_threshold.value))
                    .then(pl.lit("low_variation"))
                    .otherwise(pl.lit("informative"))
                ).alias("low_variation"),
            ]
        )
        x_col = f"{embedding_stream.value}_pc1"
        y_col = f"{embedding_stream.value}_pc2"
        color_col = str(embedding_color.value)
        embedding_chart = mo.ui.altair_chart(
            alt.Chart(alt.Data(values=plot_df.to_dicts()))
            .mark_circle(size=48, opacity=0.7)
            .encode(
                x=alt.X(f"{x_col}:Q", title=f"{embedding_stream.value} PC1"),
                y=alt.Y(f"{y_col}:Q", title=f"{embedding_stream.value} PC2"),
                color=alt.Color(f"{color_col}:N", title=color_col),
                tooltip=["view_index:Q", "scan_id:N", "semantic_target_key:N", "series_description:N", "normalized_patch_std:Q", "source_patch_mm:Q"],
            )
            .properties(width=720, height=420, title=f"{embedding_stream.value} PCA")
        )
        summary = (
            plot_df.group_by(color_col)
            .agg([pl.len().alias("n"), pl.col("normalized_patch_std").mean().alias("mean_patch_std")])
            .sort("n", descending=True)
        )
        embedding_panel = mo.vstack([embedding_chart, mo.ui.table(summary.head(30), label="Embedding color summary")])
    embedding_panel
    return


@app.cell
def _(cache, enrich_pair_metadata, include_self, max_pairs, max_scans):
    if cache is None:
        pair_df = pl.DataFrame([])
    else:
        pair_df = enrich_pair_metadata(cache, include_self=include_self.value, max_scans=int(max_scans.value), max_pairs=int(max_pairs.value), seed=17)
    mo.ui.table(pl.DataFrame([{"n_pairs": int(pair_df.height)}]), label="Pair selection")
    return (pair_df,)


@app.cell
def _(batch_size, cache, device, evaluate_pair_relation, model, pair_df):
    if cache is None or model is None or pair_df.height == 0:
        eval_df = pl.DataFrame([])
    else:
        eval_df = evaluate_pair_relation(model=model, device=device, cache=cache, pair_df=pair_df, batch_size=int(batch_size.value))
    return (eval_df,)


@app.cell
def _(axis_summary_table, eval_df, low_variation_threshold):
    summary_df = axis_summary_table(eval_df, float(low_variation_threshold.value))
    mo.ui.table(summary_df, label="Axis summary")
    return (summary_df,)


@app.cell
def _(axis_key, axis_titles, confusion_table, eval_df, filter_mode, low_variation_threshold):
    conf_df = confusion_table(eval_df, axis_key=axis_key.value, threshold=float(low_variation_threshold.value), filter_mode=filter_mode.value)
    if conf_df.height == 0:
        confusion_panel = mo.md("No valid pairs for this axis.")
    else:
        base = alt.Chart(alt.Data(values=conf_df.to_dicts())).properties(width=220, height=220, title=f"{axis_titles[axis_key.value]} confusion")
        heatmap = base.mark_rect().encode(
            x=alt.X("pred:N", title="Predicted sign"),
            y=alt.Y("target:N", title="Target sign"),
            color=alt.Color("row_rate:Q", title="Row rate"),
        )
        text = base.mark_text(color="white").encode(x="pred:N", y="target:N", text="count:Q")
        confusion_panel = mo.ui.altair_chart(heatmap + text)
    confusion_panel
    return (conf_df,)


@app.cell
def _(axis_key, axis_titles, eval_df, filter_mode, low_variation_threshold, min_group_count, slice_group, slice_summary_table):
    slice_df = slice_summary_table(
        eval_df,
        axis_key=axis_key.value,
        threshold=float(low_variation_threshold.value),
        filter_mode=filter_mode.value,
        group_key=str(slice_group.value),
        min_count=int(min_group_count.value),
    )
    if slice_df.height == 0:
        slice_panel = mo.md("No slice groups met the threshold.")
    else:
        slice_chart = mo.ui.altair_chart(
            alt.Chart(alt.Data(values=slice_df.head(24).to_dicts()))
            .mark_bar()
            .encode(
                x=alt.X(f"{slice_group.value}:N", sort="-y", title=slice_group.value),
                y=alt.Y("error_rate:Q", title="Error rate"),
                tooltip=["n:Q", "accuracy:Q", "center_distance_mm_mean:Q"],
            )
            .properties(width=700, height=280, title=f"Worst {slice_group.value} groups for {axis_titles[axis_key.value]}")
        )
        slice_panel = mo.vstack([slice_chart, mo.ui.table(slice_df.head(40), label="Failure slices")])
    slice_panel
    return (slice_df,)


@app.cell
def _(axis_key, error_examples, eval_df, example_mode, filter_mode, low_variation_threshold):
    example_df = error_examples(
        eval_df,
        axis_key=axis_key.value,
        threshold=float(low_variation_threshold.value),
        filter_mode=filter_mode.value,
        mode=example_mode.value,
    )
    return (example_df,)


@app.cell
def _(example_df):
    example_idx = mo.ui.slider(0, max(int(example_df.height) - 1, 0), value=0, step=1, label="Error example index")
    example_idx
    return (example_idx,)


@app.cell
def _(axis_order, cache, example_df, example_idx, example_pair_images):
    if cache is None or example_df.height == 0:
        example_panel = mo.md("No error examples for the selected settings.")
    else:
        row = example_df.row(int(example_idx.value), named=True)
        img_a, img_b = example_pair_images(cache, row)
        meta_df = pl.DataFrame(
            [
                {"metric": "scan_id", "value": str(row["scan_id_a"])},
                {"metric": "semantic_a", "value": str(row["semantic_target_key_a"])},
                {"metric": "semantic_b", "value": str(row["semantic_target_key_b"])},
                {"metric": "center_distance_mm", "value": f"{float(row['center_distance_mm']):.2f}"},
                {"metric": "std_a", "value": f"{float(row['normalized_patch_std_a']):.4f}"},
                {"metric": "std_b", "value": f"{float(row['normalized_patch_std_b']):.4f}"},
            ]
        )
        axis_df = pl.DataFrame(
            [
                {
                    "axis": axis,
                    "target": bool(row[f"target_{axis}"]),
                    "pred": bool(row[f"pred_{axis}"]),
                    "valid": bool(row[f"valid_{axis}"]),
                    "prob_pos": float(row[f"prob_{axis}"]),
                }
                for axis in axis_order
            ]
        )
        example_panel = mo.vstack(
            [
                mo.hstack([mo.vstack([mo.md("**View A**"), mo.image(img_a)]), mo.vstack([mo.md("**View B**"), mo.image(img_b)])]),
                mo.hstack([mo.ui.table(meta_df, label="Pair metadata"), mo.ui.table(axis_df, label="Axis predictions")]),
            ]
        )
    example_panel
    return


if __name__ == "__main__":
    app.run()
