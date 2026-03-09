import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


with app.setup:
    import os
    import sys
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.config import load_run_config_from_flat
    from prism_ssl.eval import (
        build_eval_batch,
        build_model_from_checkpoint,
        cosine_similarity_matrix,
        load_checkpoint_payload,
        masked_l1_per_view,
        nearest_neighbor_purity,
        pca_project,
        sample_study4_examples,
        within_between_cosine_gap,
    )
    from prism_ssl.eval.checkpoint_probe import VIEW_NAMES

    alt.data_transformers.disable_max_rows()

    def bucket_top_labels(values: list[object], top_k: int) -> list[str]:
        counts = Counter(values)
        keep = {
            label
            for label, _count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[: max(int(top_k), 0)]
        }
        return [str(value) if value in keep else "other" for value in values]

    def similarity_frame(matrix: np.ndarray, view_keys: list[str]) -> pl.DataFrame:
        sim = np.asarray(matrix, dtype=np.float32)
        coords = np.indices(sim.shape)
        flat_x = coords[1].reshape(-1).astype(np.int64)
        flat_y = coords[0].reshape(-1).astype(np.int64)
        return pl.DataFrame(
            {
                "x_idx": flat_x,
                "y_idx": flat_y,
                "x_key": [view_keys[idx] for idx in flat_x.tolist()],
                "y_key": [view_keys[idx] for idx in flat_y.tolist()],
                "similarity": sim.reshape(-1),
            }
        )

    def summarize_distribution(name: str, values: np.ndarray) -> dict[str, object]:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return {
                "metric": name,
                "count": 0,
                "mean": None,
                "median": None,
                "p25": None,
                "p75": None,
            }
        return {
            "metric": name,
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    def patch_grid(patches: np.ndarray, max_patches: int = 32, cols: int = 8) -> Image.Image:
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        n = min(int(max_patches), int(arr.shape[0]))
        if n <= 0:
            base = np.zeros((16, 16), dtype=np.uint8)
            return Image.fromarray(base)
        arr = arr[:n]
        rows = (n + cols - 1) // cols
        pad = rows * cols - n
        if pad > 0:
            arr = np.concatenate([arr, np.zeros((pad, arr.shape[1], arr.shape[2]), dtype=arr.dtype)], axis=0)
        row_imgs = [np.concatenate(arr[r * cols : (r + 1) * cols], axis=1) for r in range(rows)]
        grid = np.concatenate(row_imgs, axis=0)
        grid[0, 0] = -1.0
        if grid.shape[1] > 1:
            grid[0, 1] = 1.0
        gray = np.clip((grid + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(gray).resize((gray.shape[1] * 4, gray.shape[0] * 4), Image.NEAREST)

    def metric_display(value: float | None) -> str:
        if value is None:
            return "n/a"
        if np.isnan(value):
            return "n/a"
        return f"{value:.3f}"


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    intro = mo.md(
        """
# Prism SSL Study4 Checkpoint Probe

Load a local checkpoint, run deterministic `study4` sampling, and inspect:
- SupCon clustering by series label
- Anatomy/view CLS clustering by dominant TotalSegmentator label ID
- Masked-patch reconstruction quality for self/register/cross paths
"""
    )
    intro
    return (is_script_mode,)


@app.cell
def _(Path, mo, os):
    default_catalog = os.environ.get(
        "CATALOG_PATH",
        str(Path(__file__).resolve().parents[2] / "pmbb_catalog.csv.gz"),
    )
    checkpoint_path = mo.ui.text(label="Checkpoint path", value=os.environ.get("PRISM_NOTEBOOK_CKPT", ""))
    catalog_path = mo.ui.text(label="Catalog path", value=default_catalog)
    device = mo.ui.dropdown(options=["auto", "cpu", "cuda"], value="auto", label="Device")
    n_studies = mo.ui.slider(start=1, stop=128, step=1, value=32, label="Studies")
    eval_batch_size = mo.ui.slider(start=1, stop=16, step=1, value=4, label="Eval batch size")
    seed = mo.ui.number(label="Seed", value=42, step=1)
    include_background_label_0 = mo.ui.checkbox(label="Include anatomy label 0", value=False)

    mo.vstack(
        [
            mo.md("## Controls"),
            mo.hstack([checkpoint_path, catalog_path]),
            mo.hstack([device, n_studies, eval_batch_size, seed, include_background_label_0]),
        ]
    )
    return (
        catalog_path,
        checkpoint_path,
        device,
        eval_batch_size,
        include_background_label_0,
        n_studies,
        seed,
    )


@app.cell
def _(Path, checkpoint_path, load_checkpoint_payload, load_run_config_from_flat):
    checkpoint_hint = None
    hinted_config = None
    _ckpt_text = str(checkpoint_path.value).strip()
    if _ckpt_text:
        _hint_path = Path(_ckpt_text).expanduser()
        if _hint_path.is_file():
            _hint_payload = load_checkpoint_payload(_hint_path, device="cpu")
            _flat_cfg = _hint_payload.get("config")
            if isinstance(_flat_cfg, dict):
                hinted_config = load_run_config_from_flat(_flat_cfg)
        else:
            checkpoint_hint = f"Checkpoint not found: {_hint_path}"
    return checkpoint_hint, hinted_config


@app.cell
def _(checkpoint_hint, hinted_config, mo):
    _default_modalities = ",".join(hinted_config.data.modality_filter) if hinted_config is not None else "CT,MR"
    modality_filter = mo.ui.text(label="Modality filter", value=_default_modalities)
    _note = mo.callout(checkpoint_hint, kind="warn") if checkpoint_hint else mo.md("")
    mo.vstack([mo.hstack([modality_filter]), _note])
    return (modality_filter,)


@app.cell
def _(eval_batch_size, is_script_mode, n_studies):
    effective_n_studies = 2 if is_script_mode else int(n_studies.value)
    effective_eval_batch_size = 1 if is_script_mode else int(eval_batch_size.value)
    return effective_eval_batch_size, effective_n_studies


@app.cell
def _(Path, build_model_from_checkpoint, checkpoint_path, device, mo):
    _ckpt_text = str(checkpoint_path.value).strip()
    mo.stop(not _ckpt_text, mo.callout("Provide a local checkpoint path to begin.", kind="warn"))
    _ckpt_path = Path(_ckpt_text).expanduser()
    mo.stop(not _ckpt_path.is_file(), mo.callout(f"Checkpoint not found: {_ckpt_path}", kind="danger"))

    model, config, payload, resolved_device = build_model_from_checkpoint(_ckpt_path, device=device.value)
    return config, model, payload, resolved_device


@app.cell
def _(config, effective_eval_batch_size, effective_n_studies, mo, payload, resolved_device):
    _sample_unit = str(config.data.sample_unit).strip().lower()
    _ckpt_summary = pl.DataFrame(
        [
            {
                "step": int(payload.get("step", 0)),
                "sample_unit": _sample_unit,
                "resolved_device": str(resolved_device),
                "d_model": int(config.model.d_model),
                "proj_dim": int(config.model.proj_dim),
                "n_patches": int(config.data.n_patches),
                "patch_mm": float(config.data.patch_mm),
                "modalities": ",".join(config.data.modality_filter),
                "eval_studies": int(effective_n_studies),
                "eval_batch_size": int(effective_eval_batch_size),
            }
        ]
    )
    _status = (
        mo.callout("Checkpoint is ready for study4 analysis.", kind="success")
        if _sample_unit == "study4"
        else mo.callout(
            f"Checkpoint sample_unit='{config.data.sample_unit}' is not supported by this notebook.",
            kind="warn",
        )
    )
    mo.vstack([mo.md("## Checkpoint Summary"), _status, _ckpt_summary])
    return


@app.cell
def _(
    F,
    VIEW_NAMES,
    build_eval_batch,
    catalog_path,
    config,
    cosine_similarity_matrix,
    effective_eval_batch_size,
    effective_n_studies,
    masked_l1_per_view,
    modality_filter,
    model,
    mo,
    sample_study4_examples,
    seed,
    torch,
):
    mo.stop(
        str(config.data.sample_unit).strip().lower() != "study4",
        mo.callout("This notebook only supports `study4` checkpoints.", kind="warn"),
    )

    _modalities = tuple(m.strip().upper() for m in str(modality_filter.value).split(",") if m.strip())
    mo.stop(len(_modalities) == 0, mo.callout("Specify at least one modality.", kind="warn"))

    _examples = sample_study4_examples(
        catalog_path.value,
        config,
        n_studies=int(effective_n_studies),
        seed=int(seed.value),
        modality_filter=_modalities,
    )
    mo.stop(len(_examples) == 0, mo.callout("No valid study4 examples could be sampled.", kind="danger"))

    _all_view_rows: list[dict[str, object]] = []
    _all_sample_rows: list[dict[str, object]] = []
    _reconstruction_rows: list[dict[str, object]] = []
    _mim_rows: list[dict[str, object]] = []
    _supcon_chunks: list[torch.Tensor] = []
    _direction_chunks: list[torch.Tensor] = []
    _model_device = next(model.parameters()).device

    for _start in range(0, len(_examples), int(effective_eval_batch_size)):
        _chunk = _examples[_start : _start + int(effective_eval_batch_size)]
        _batch = build_eval_batch(_chunk, sample_offset=_start)
        with torch.no_grad():
            _outputs = model.forward_study4(
                _batch["patches_views"].to(_model_device),
                _batch["positions_views"].to(_model_device),
                _batch["cross_valid"].to(_model_device),
            )

        if _outputs.proj_views is None or _outputs.direction_cls_views is None:
            raise ValueError("study4 forward pass did not return the expected view embeddings")

        _supcon_chunks.append(_outputs.proj_views.detach().cpu().reshape(-1, _outputs.proj_views.shape[-1]))
        _direction_chunks.append(
            F.normalize(
                _outputs.direction_cls_views.detach().cpu().reshape(-1, _outputs.direction_cls_views.shape[-1]),
                dim=-1,
            )
        )
        _all_view_rows.extend(_batch["view_rows"])
        _all_sample_rows.extend(_batch["sample_rows"])

        _l1_bundle = masked_l1_per_view(_outputs, _batch["cross_valid"])
        _self_l1 = _l1_bundle["self"].detach().cpu().numpy()
        _register_l1 = _l1_bundle["register"].detach().cpu().numpy()
        _cross_l1 = _l1_bundle["cross"].detach().cpu().numpy()
        _cross_valid_mask = _l1_bundle["cross_valid_mask"].detach().cpu().numpy()

        for _local_sample_idx in range(_self_l1.shape[0]):
            for _view_idx, _view_name in enumerate(VIEW_NAMES):
                _row = _batch["view_rows"][_local_sample_idx * len(VIEW_NAMES) + _view_idx]
                _mim_rows.append(
                    {
                        **_row,
                        "self_l1": float(_self_l1[_local_sample_idx, _view_idx]),
                        "register_l1": float(_register_l1[_local_sample_idx, _view_idx]),
                        "cross_l1": float(_cross_l1[_local_sample_idx, _view_idx]),
                        "cross_valid_for_view": bool(_cross_valid_mask[_local_sample_idx, _view_idx]),
                    }
                )
                _reconstruction_rows.append(
                    {
                        **_row,
                        "self_l1": float(_self_l1[_local_sample_idx, _view_idx]),
                        "register_l1": float(_register_l1[_local_sample_idx, _view_idx]),
                        "cross_l1": float(_cross_l1[_local_sample_idx, _view_idx]),
                        "cross_valid_for_view": bool(_cross_valid_mask[_local_sample_idx, _view_idx]),
                        "target_patches": _outputs.mim_self_targets[_view_idx][_local_sample_idx].detach().cpu().numpy(),
                        "pred_self": _outputs.mim_self_preds[_view_idx][_local_sample_idx].detach().cpu().numpy(),
                        "pred_register": _outputs.mim_register_preds[_view_idx][_local_sample_idx].detach().cpu().numpy(),
                        "pred_cross": _outputs.mim_cross_preds[_view_idx][_local_sample_idx].detach().cpu().numpy(),
                    }
                )

    _supcon_embeddings = torch.cat(_supcon_chunks, dim=0)
    _direction_embeddings = torch.cat(_direction_chunks, dim=0)
    _supcon_similarity = cosine_similarity_matrix(_supcon_embeddings).detach().cpu().numpy()
    _view_df = pl.DataFrame(_all_view_rows)
    _sample_df = pl.DataFrame(_all_sample_rows)
    _mim_df = pl.DataFrame(_mim_rows)

    probe_state = {
        "examples": _examples,
        "view_df": _view_df,
        "sample_df": _sample_df,
        "mim_df": _mim_df,
        "reconstruction_rows": _reconstruction_rows,
        "supcon_embeddings": _supcon_embeddings,
        "direction_embeddings": _direction_embeddings,
        "supcon_similarity": _supcon_similarity,
        "effective_modalities": _modalities,
        "totalseg_resolved_count": int(sum(bool(row["totalseg_resolved"]) for row in _all_view_rows)),
    }
    return (probe_state,)


@app.cell
def _(mo, probe_state):
    _view_df = probe_state["view_df"]
    _eval_summary = pl.DataFrame(
        [
            {
                "samples": int(probe_state["sample_df"].height),
                "views": int(_view_df.height),
                "totalseg_resolved": f"{int(probe_state['totalseg_resolved_count'])}/{int(_view_df.height)}",
                "modalities": ",".join(probe_state["effective_modalities"]),
            }
        ]
    )
    mo.vstack([mo.md("## Evaluation Summary"), _eval_summary])
    return


@app.cell
def _(
    alt,
    bucket_top_labels,
    metric_display,
    mo,
    nearest_neighbor_purity,
    pca_project,
    probe_state,
    similarity_frame,
    within_between_cosine_gap,
):
    _view_df = probe_state["view_df"]
    _supcon_embeddings = probe_state["supcon_embeddings"]
    _coords, _explained = pca_project(_supcon_embeddings)
    _series_labels = _view_df["series_label_text"].to_list()
    _supcon_df = _view_df.with_columns(
        [
            pl.Series("pc1", _coords[:, 0]),
            pl.Series("pc2", _coords[:, 1]),
            pl.Series("series_bucket", bucket_top_labels(_series_labels, top_k=12)),
            pl.Series(
                "view_key",
                [f"s{sample_idx}:{view_name}" for sample_idx, view_name in zip(_view_df["sample_index"], _view_df["view_name"])],
            ),
        ]
    )
    _purity = nearest_neighbor_purity(_supcon_embeddings, _series_labels)
    _gap = within_between_cosine_gap(_supcon_embeddings, _series_labels)

    _scatter = (
        alt.Chart(alt.Data(values=_supcon_df.to_dicts()))
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("pc1:Q", title=f"PC1 ({_explained[0] * 100:.1f}%)"),
            y=alt.Y("pc2:Q", title=f"PC2 ({_explained[1] * 100:.1f}%)"),
            color=alt.Color("series_bucket:N", title="Series label"),
            tooltip=[
                alt.Tooltip("sample_index:Q", title="sample"),
                alt.Tooltip("view_name:N", title="view"),
                alt.Tooltip("series_label_text:N", title="series label"),
                alt.Tooltip("series_description:N", title="series description"),
                alt.Tooltip("study_id:N", title="study_id"),
                alt.Tooltip("series_path:N", title="series_path"),
            ],
        )
        .properties(title="SupCon PCA", height=360)
    )

    _sim_df = similarity_frame(probe_state["supcon_similarity"], _supcon_df["view_key"].to_list())
    _heatmap = (
        alt.Chart(alt.Data(values=_sim_df.to_dicts()))
        .mark_rect()
        .encode(
            x=alt.X("x_idx:O", title="view index"),
            y=alt.Y("y_idx:O", title="view index"),
            color=alt.Color("similarity:Q", scale=alt.Scale(domain=[-1.0, 1.0], scheme="redblue"), title="cosine"),
            tooltip=[
                alt.Tooltip("x_key:N", title="x"),
                alt.Tooltip("y_key:N", title="y"),
                alt.Tooltip("similarity:Q", format=".3f"),
            ],
        )
        .properties(title="SupCon cosine similarity", height=420, width=420)
    )

    _metrics = pl.DataFrame(
        [
            {"metric": "nearest_neighbor_purity", "value": metric_display(_purity)},
            {"metric": "within_between_cosine_gap", "value": metric_display(_gap)},
        ]
    )

    mo.vstack([mo.md("## SupCon Clustering"), _metrics, mo.ui.altair_chart(_scatter), mo.ui.altair_chart(_heatmap)])
    return


@app.cell
def _(
    alt,
    bucket_top_labels,
    include_background_label_0,
    metric_display,
    mo,
    nearest_neighbor_purity,
    pca_project,
    probe_state,
    within_between_cosine_gap,
):
    _view_df = probe_state["view_df"]
    _direction_embeddings = probe_state["direction_embeddings"]
    _coords, _explained = pca_project(_direction_embeddings)
    _anatomy_labels = _view_df["anatomy_label"].to_list()
    _ignore_labels = None if include_background_label_0.value else {0}
    _counts = Counter(label for label in _anatomy_labels if label != 0)
    _keep = {
        label
        for label, _count in sorted(_counts.items(), key=lambda item: (-item[1], str(item[0])))[:15]
    }
    _anatomy_buckets = [str(label) if (label in _keep or (label == 0 and include_background_label_0.value)) else "other" for label in _anatomy_labels]
    _anatomy_df = _view_df.with_columns(
        [
            pl.Series("pc1", _coords[:, 0]),
            pl.Series("pc2", _coords[:, 1]),
            pl.Series("anatomy_bucket", _anatomy_buckets),
        ]
    )
    _plot_df = _anatomy_df if include_background_label_0.value else _anatomy_df.filter(pl.col("anatomy_label") != 0)
    _purity = nearest_neighbor_purity(_direction_embeddings, _anatomy_labels, ignore_labels=_ignore_labels)
    _gap = within_between_cosine_gap(_direction_embeddings, _anatomy_labels, ignore_labels=_ignore_labels)
    _label_counts = (
        _anatomy_df.group_by("anatomy_label")
        .len()
        .rename({"len": "count"})
        .sort(["count", "anatomy_label"], descending=[True, False])
    )

    _scatter = (
        alt.Chart(alt.Data(values=_plot_df.to_dicts()))
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("pc1:Q", title=f"PC1 ({_explained[0] * 100:.1f}%)"),
            y=alt.Y("pc2:Q", title=f"PC2 ({_explained[1] * 100:.1f}%)"),
            color=alt.Color("anatomy_bucket:N", title="TS label"),
            tooltip=[
                alt.Tooltip("sample_index:Q", title="sample"),
                alt.Tooltip("view_name:N", title="view"),
                alt.Tooltip("anatomy_label:Q", title="anatomy label"),
                alt.Tooltip("study_id:N", title="study_id"),
                alt.Tooltip("series_path:N", title="series_path"),
                alt.Tooltip("cross_valid:N", title="cross_valid"),
            ],
        )
        .properties(title="Anatomy/View CLS PCA", height=360)
    )

    _metrics = pl.DataFrame(
        [
            {"metric": "nearest_neighbor_purity", "value": metric_display(_purity)},
            {"metric": "within_between_cosine_gap", "value": metric_display(_gap)},
            {"metric": "background_label_0_included", "value": str(bool(include_background_label_0.value)).lower()},
        ]
    )

    mo.vstack([mo.md("## Anatomy/View CLS Clustering"), _metrics, _label_counts, mo.ui.altair_chart(_scatter)])
    return


@app.cell
def _(alt, metric_display, mo, probe_state, summarize_distribution):
    _mim_df = probe_state["mim_df"]
    _self_values = _mim_df["self_l1"].to_numpy()
    _register_values = _mim_df["register_l1"].to_numpy()
    _cross_df = _mim_df.filter(pl.col("cross_valid_for_view"))
    _cross_values = _cross_df["cross_l1"].to_numpy()
    _excluded_cross = int(_mim_df.height - _cross_df.height)

    _summary_df = pl.DataFrame(
        [
            summarize_distribution("self", _self_values),
            summarize_distribution("register", _register_values),
            summarize_distribution("cross_valid_only", _cross_values),
        ]
    )

    _scatter_self_register = (
        alt.Chart(alt.Data(values=_mim_df.to_dicts()))
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X("self_l1:Q", title="self L1"),
            y=alt.Y("register_l1:Q", title="register L1"),
            color=alt.Color("view_name:N", title="view"),
            tooltip=[
                alt.Tooltip("sample_index:Q", title="sample"),
                alt.Tooltip("view_name:N", title="view"),
                alt.Tooltip("self_l1:Q", format=".4f"),
                alt.Tooltip("register_l1:Q", format=".4f"),
                alt.Tooltip("cross_mode:N", title="cross mode"),
            ],
        )
        .properties(title="Self vs register masked L1", height=340)
    )

    _scatter_self_cross = (
        alt.Chart(alt.Data(values=_cross_df.to_dicts()))
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X("self_l1:Q", title="self L1"),
            y=alt.Y("cross_l1:Q", title="cross L1"),
            color=alt.Color("view_name:N", title="view"),
            tooltip=[
                alt.Tooltip("sample_index:Q", title="sample"),
                alt.Tooltip("view_name:N", title="view"),
                alt.Tooltip("self_l1:Q", format=".4f"),
                alt.Tooltip("cross_l1:Q", format=".4f"),
                alt.Tooltip("cross_mode:N", title="cross mode"),
            ],
        )
        .properties(title="Self vs cross masked L1", height=340)
    )

    _note = mo.callout(f"Excluded {_excluded_cross} cross-view points where `cross_valid` was false.", kind="warn")
    mo.vstack(
        [
            mo.md("## MAE / MIM Performance"),
            _note,
            _summary_df,
            mo.ui.altair_chart(_scatter_self_register),
            mo.ui.altair_chart(_scatter_self_cross),
        ]
    )
    return


@app.cell
def _(mo, probe_state):
    _mim_df = probe_state["mim_df"]
    sample_options = [str(v) for v in _mim_df["sample_index"].unique().sort().to_list()]
    sample_picker = mo.ui.dropdown(options=sample_options, value=sample_options[0], label="Sample")
    view_picker = mo.ui.dropdown(options=list(VIEW_NAMES), value=VIEW_NAMES[0], label="View")
    mo.vstack([mo.md("## Reconstruction Browser"), mo.hstack([sample_picker, view_picker])])
    return sample_picker, view_picker


@app.cell
def _(mo, patch_grid, probe_state, sample_picker, view_picker):
    _target_sample = int(sample_picker.value)
    _target_view = str(view_picker.value)
    _match = next(
        row
        for row in probe_state["reconstruction_rows"]
        if int(row["sample_index"]) == _target_sample and str(row["view_name"]) == _target_view
    )

    _cross_note = (
        mo.callout("Cross reconstruction fell back because `cross_valid` was false for this sample.", kind="warn")
        if not bool(_match["cross_valid_for_view"])
        else mo.md("")
    )
    mo.vstack(
        [
            mo.md(
                f"Sample `{_target_sample}` view `{_target_view}` "
                f"(self={_match['self_l1']:.4f}, register={_match['register_l1']:.4f}, cross={_match['cross_l1']:.4f})"
            ),
            _cross_note,
            mo.hstack(
                [
                    mo.vstack([mo.md("Target masked patches"), mo.image(patch_grid(_match["target_patches"]), width=360)]),
                    mo.vstack([mo.md("Self prediction"), mo.image(patch_grid(_match["pred_self"]), width=360)]),
                    mo.vstack([mo.md("Register prediction"), mo.image(patch_grid(_match["pred_register"]), width=360)]),
                    mo.vstack([mo.md("Cross prediction"), mo.image(patch_grid(_match["pred_cross"]), width=360)]),
                ]
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
