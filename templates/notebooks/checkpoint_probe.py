import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


with app.setup:
    import os
    import sys
    import time
    from collections import Counter
    from pathlib import Path
    from typing import Mapping

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
        download_wandb_run_checkpoint,
        list_wandb_run_model_artifacts,
        load_checkpoint_payload,
        masked_l1_per_view,
        nearest_neighbor_purity,
        pca_project,
        sample_study4_examples,
        within_between_cosine_gap,
    )
    from prism_ssl.eval.checkpoint_probe import VIEW_NAMES
    from prism_ssl.validation import (
        build_eval_batch_from_ct_validation_cache,
        infer_contrast_bucket,
        infer_series_family,
        load_ct_validation_cache,
    )

    alt.data_transformers.disable_max_rows()

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

    def _series_display_text(row: Mapping[str, object]) -> str:
        description = str(row.get("series_description", "") or "").strip()
        if description:
            return description
        label = str(row.get("series_label_text", "") or "").strip()
        if label:
            return label
        series_path = str(row.get("series_path", "") or "").strip()
        return Path(series_path).name or "unknown"

    def _scan_label(row: Mapping[str, object]) -> str:
        description = _series_display_text(row)
        study_id = str(row.get("study_id", "") or "").strip()
        series_id = str(row.get("series_id", "") or "").strip()
        series_suffix = series_id.split("_")[-1] if series_id else "unknown"
        return " | ".join(part for part in [description, study_id, series_suffix] if part)

    def _axis_truth(delta_mm: float, *, negative_label: str, positive_label: str, threshold_mm: float = 1.0) -> tuple[str, bool, int]:
        delta_value = float(delta_mm)
        if abs(delta_value) < float(threshold_mm):
            return "ambiguous", False, 0
        is_positive = int(delta_value > 0.0)
        return (positive_label if is_positive else negative_label), True, is_positive

    def _axis_prediction(prob_positive: float, *, negative_label: str, positive_label: str) -> tuple[str, int, float]:
        prob_value = float(prob_positive)
        pred_positive = int(prob_value >= 0.5)
        confidence = prob_value if pred_positive == 1 else 1.0 - prob_value
        return (positive_label if pred_positive == 1 else negative_label), pred_positive, float(confidence)

    def build_relative_position_pair_rows(
        batch: Mapping[str, object],
        outputs,
    ) -> list[dict[str, object]]:
        if outputs.distance_logits_x is None or outputs.distance_logits_y is None:
            raise ValueError("study4 forward pass did not return relative-position logits")

        probs_x = torch.sigmoid(outputs.distance_logits_x[:, :3].detach().cpu().to(dtype=torch.float32)).numpy()
        probs_y = torch.sigmoid(outputs.distance_logits_y[:, :3].detach().cpu().to(dtype=torch.float32)).numpy()
        deltas_x = batch["center_delta_mm_x"].detach().cpu().to(dtype=torch.float32).numpy()
        deltas_y = batch["center_delta_mm_y"].detach().cpu().to(dtype=torch.float32).numpy()
        sample_rows = list(batch["sample_rows"])
        view_rows = list(batch["view_rows"])

        axis_specs = (
            ("left_right", "left", "right", 0),
            ("front_back", "back", "front", 1),
            ("top_bottom", "bottom", "top", 2),
        )

        pair_rows: list[dict[str, object]] = []
        for local_index, sample_row in enumerate(sample_rows):
            sample_view_rows = view_rows[local_index * 4 : (local_index + 1) * 4]
            if len(sample_view_rows) != 4:
                raise ValueError(
                    f"Expected 4 view rows per sample, got {len(sample_view_rows)} for sample_index={sample_row['sample_index']}"
                )

            pair_specs = (
                ("x", sample_view_rows[0], sample_view_rows[2], probs_x[local_index], deltas_x[local_index]),
                ("y", sample_view_rows[1], sample_view_rows[3], probs_y[local_index], deltas_y[local_index]),
            )

            for pair_name, anchor_view, target_view, axis_probs, axis_deltas in pair_specs:
                anchor_pt = np.asarray(anchor_view["prism_center_pt"], dtype=np.float32)
                target_pt = np.asarray(target_view["prism_center_pt"], dtype=np.float32)
                midpoint = 0.5 * (anchor_pt + target_pt)
                series_id = str(anchor_view["series_id"])
                base_row: dict[str, object] = {
                    "pair_key": f"{int(sample_row['sample_index'])}:{pair_name}",
                    "pair_name": str(pair_name),
                    "sample_index": int(sample_row["sample_index"]),
                    "study_sample_index": int(sample_row.get("study_sample_index", 0)),
                    "study_id": str(sample_row["study_id"]),
                    "series_id": series_id,
                    "scan_label": _scan_label(anchor_view),
                    "series_description_display": _series_display_text(anchor_view),
                    "series_path": str(anchor_view.get("series_path", "")),
                    "series_family": infer_series_family(anchor_view),
                    "contrast_bucket": infer_contrast_bucket(anchor_view),
                    "native_acquisition_plane": str(anchor_view.get("native_acquisition_plane", "unknown")),
                    "cross_valid": bool(sample_row.get("cross_valid", False)),
                    "cross_mode": str(sample_row.get("cross_mode", "")),
                    "anchor_view_name": str(anchor_view["view_name"]),
                    "target_view_name": str(target_view["view_name"]),
                    "anchor_r_mm": float(anchor_pt[0]),
                    "anchor_a_mm": float(anchor_pt[1]),
                    "anchor_s_mm": float(anchor_pt[2]),
                    "target_r_mm": float(target_pt[0]),
                    "target_a_mm": float(target_pt[1]),
                    "target_s_mm": float(target_pt[2]),
                    "mid_r_mm": float(midpoint[0]),
                    "mid_a_mm": float(midpoint[1]),
                    "mid_s_mm": float(midpoint[2]),
                    "delta_r_mm": float(axis_deltas[0]),
                    "delta_a_mm": float(axis_deltas[1]),
                    "delta_s_mm": float(axis_deltas[2]),
                    "pair_distance_mm": float(np.linalg.norm(np.asarray(axis_deltas[:3], dtype=np.float32))),
                }

                valid_axes = 0
                correct_axes = 0
                for axis_name, negative_label, positive_label, axis_index in axis_specs:
                    truth_label, is_valid, truth_positive = _axis_truth(
                        float(axis_deltas[axis_index]),
                        negative_label=negative_label,
                        positive_label=positive_label,
                    )
                    pred_label, pred_positive, confidence = _axis_prediction(
                        float(axis_probs[axis_index]),
                        negative_label=negative_label,
                        positive_label=positive_label,
                    )
                    is_correct = bool(pred_positive == truth_positive) if is_valid else None
                    if is_valid:
                        valid_axes += 1
                        correct_axes += int(bool(is_correct))
                    base_row[f"{axis_name}_truth"] = truth_label
                    base_row[f"{axis_name}_pred"] = pred_label
                    base_row[f"{axis_name}_valid"] = bool(is_valid)
                    base_row[f"{axis_name}_correct"] = is_correct
                    base_row[f"{axis_name}_prob_positive"] = float(axis_probs[axis_index])
                    base_row[f"{axis_name}_confidence"] = float(confidence)

                base_row["n_valid_axes"] = int(valid_axes)
                base_row["n_incorrect_axes"] = int(valid_axes - correct_axes) if valid_axes > 0 else None
                base_row["pair_accuracy"] = float(correct_axes / valid_axes) if valid_axes > 0 else None
                pair_rows.append(base_row)

        return pair_rows


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    intro = mo.md(
        """
# Prism SSL Study4 Checkpoint Probe

Load a local checkpoint or a W&B run artifact, run deterministic `study4` sampling, and inspect:
- Trained relative-position decoder accuracy for left/right, front/back, and top/bottom
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
        str(Path(__file__).resolve().parents[1] / "data" / "pmbb_catalog.csv.gz"),
    )
    default_validation_cache = os.environ.get("PRISM_VALIDATION_CACHE", "")
    checkpoint_path = mo.ui.text(label="Checkpoint path (optional)", value=os.environ.get("PRISM_NOTEBOOK_CKPT", ""))
    wandb_run_ref = mo.ui.text(label="W&B run URL (optional)", value=os.environ.get("PRISM_NOTEBOOK_WANDB_RUN", ""))
    wandb_force_refresh = mo.ui.checkbox(label="Refresh W&B artifacts", value=False)
    catalog_path = mo.ui.text(label="Catalog path (live sampling only)", value=default_catalog)
    validation_cache_dir = mo.ui.text(label="Validation cache dir (optional)", value=default_validation_cache)
    device = mo.ui.dropdown(options=["auto", "cpu", "cuda"], value="auto", label="Device")
    n_studies = mo.ui.slider(start=1, stop=8192, step=1, value=32, label="Eval samples (cache) / studies (live)")
    eval_batch_size = mo.ui.slider(start=1, stop=32, step=1, value=4, label="Eval batch size")
    seed = mo.ui.number(label="Seed", value=42, step=1)
    include_background_label_0 = mo.ui.checkbox(label="Include anatomy label 0", value=False)
    _source_note = mo.callout(
        "Provide either a local checkpoint path or a W&B run URL. Local paths take priority. "
        "W&B artifacts are downloaded into session-local /tmp storage and removed when the notebook process exits. "
        "If a validation cache dir is set, the notebook reuses cached CT patch samples instead of live NIfTI sampling.",
        kind="info",
    )

    mo.vstack(
        [
            mo.md("## Controls"),
            _source_note,
            mo.hstack([checkpoint_path, wandb_run_ref]),
            mo.hstack([catalog_path, validation_cache_dir]),
            mo.hstack([wandb_force_refresh]),
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
        validation_cache_dir,
        wandb_force_refresh,
        wandb_run_ref,
    )


@app.cell
def _(checkpoint_path, list_wandb_run_model_artifacts, wandb_force_refresh, wandb_run_ref):
    wandb_artifacts = []
    wandb_artifact_error = None
    _ckpt_text = str(checkpoint_path.value).strip()
    _run_text = str(wandb_run_ref.value).strip()
    if not _ckpt_text and _run_text:
        try:
            wandb_artifacts = list_wandb_run_model_artifacts(
                _run_text,
                force_refresh=bool(wandb_force_refresh.value),
            )
        except Exception as exc:
            wandb_artifact_error = str(exc)
    return wandb_artifact_error, wandb_artifacts


@app.cell
def _(checkpoint_path, mo, pl, wandb_artifact_error, wandb_artifacts, wandb_run_ref):
    artifact_label_to_ref = {}
    wandb_artifact_picker = None
    _ckpt_text = str(checkpoint_path.value).strip()
    _run_text = str(wandb_run_ref.value).strip()
    if _ckpt_text:
        _artifact_ui = mo.md("")
    elif wandb_artifact_error:
        _artifact_ui = mo.callout(wandb_artifact_error, kind="danger")
    elif not _run_text:
        _artifact_ui = mo.md("")
    elif len(wandb_artifacts) == 0:
        _artifact_ui = mo.callout("No model artifacts were found for this run.", kind="warn")
    else:
        _artifact_labels = [str(item["display_name"]) for item in wandb_artifacts]
        artifact_label_to_ref = {
            str(item["display_name"]): str(item["artifact_ref"]) for item in wandb_artifacts
        }
        wandb_artifact_picker = mo.ui.dropdown(
            options=_artifact_labels,
            value=_artifact_labels[0],
            label="W&B checkpoint",
        )
        _artifact_df = pl.DataFrame(
            [
                {
                    "checkpoint": str(item["display_name"]),
                    "version": str(item["version"]),
                    "aliases": ",".join(str(alias) for alias in item["aliases"]),
                    "artifact_ref": str(item["artifact_ref"]),
                }
                for item in wandb_artifacts
            ]
        )
        _artifact_ui = mo.vstack([mo.md("## W&B Checkpoints"), wandb_artifact_picker, _artifact_df])
    _artifact_ui
    return artifact_label_to_ref, wandb_artifact_picker


@app.cell
def _(
    Path,
    artifact_label_to_ref,
    checkpoint_path,
    download_wandb_run_checkpoint,
    mo,
    wandb_artifact_picker,
    wandb_force_refresh,
    wandb_run_ref,
):
    _ckpt_text = str(checkpoint_path.value).strip()
    if _ckpt_text:
        resolved_checkpoint_path = Path(_ckpt_text).expanduser()
        mo.stop(not resolved_checkpoint_path.is_file(), mo.callout(f"Checkpoint not found: {resolved_checkpoint_path}", kind="danger"))
        checkpoint_source = {
            "source": "local",
            "run_ref": "",
            "artifact_ref": "",
            "label": str(resolved_checkpoint_path),
        }
    else:
        _run_text = str(wandb_run_ref.value).strip()
        mo.stop(not _run_text, mo.callout("Provide a local checkpoint path or a W&B run URL.", kind="warn"))
        mo.stop(
            wandb_artifact_picker is None or str(wandb_artifact_picker.value).strip() not in artifact_label_to_ref,
            mo.callout("Select a W&B checkpoint artifact to continue.", kind="warn"),
        )
        _artifact_label = str(wandb_artifact_picker.value)
        _artifact_ref = artifact_label_to_ref[_artifact_label]
        resolved_checkpoint_path = download_wandb_run_checkpoint(
            _run_text,
            _artifact_ref,
            force_refresh=bool(wandb_force_refresh.value),
        )
        checkpoint_source = {
            "source": "wandb_artifact",
            "run_ref": _run_text,
            "artifact_ref": _artifact_ref,
            "label": _artifact_label,
        }
    return checkpoint_source, resolved_checkpoint_path


@app.cell
def _(load_checkpoint_payload, load_run_config_from_flat, resolved_checkpoint_path):
    hinted_config = None
    _hint_payload = load_checkpoint_payload(resolved_checkpoint_path, device="cpu")
    _flat_cfg = _hint_payload.get("config")
    if isinstance(_flat_cfg, dict):
        hinted_config = load_run_config_from_flat(_flat_cfg)
    return (hinted_config,)


@app.cell
def _(checkpoint_source, hinted_config, mo):
    _default_modalities = ",".join(hinted_config.data.modality_filter) if hinted_config is not None else "CT,MR"
    modality_filter = mo.ui.text(label="Modality filter", value=_default_modalities)
    _source_text = (
        f"Using local checkpoint `{checkpoint_source['label']}`."
        if checkpoint_source["source"] == "local"
        else f"Using W&B artifact `{checkpoint_source['label']}` from `{checkpoint_source['run_ref']}`."
    )
    _note = mo.callout(_source_text, kind="info")
    mo.vstack([mo.hstack([modality_filter]), _note])
    return (modality_filter,)


@app.cell
def _(eval_batch_size, is_script_mode, n_studies):
    effective_n_studies = 2 if is_script_mode else int(n_studies.value)
    effective_eval_batch_size = 1 if is_script_mode else int(eval_batch_size.value)
    return effective_eval_batch_size, effective_n_studies


@app.cell
def _(build_model_from_checkpoint, device, resolved_checkpoint_path):
    model, config, payload, resolved_device = build_model_from_checkpoint(resolved_checkpoint_path, device=device.value)
    return config, model, payload, resolved_device


@app.cell
def _(checkpoint_source, config, effective_eval_batch_size, effective_n_studies, mo, payload, resolved_checkpoint_path, resolved_device):
    _sample_unit = str(config.data.sample_unit).strip().lower()
    _ckpt_summary = pl.DataFrame(
        [
            {
                "step": int(payload.get("step", 0)),
                "checkpoint_source": str(checkpoint_source["source"]),
                "checkpoint_label": str(checkpoint_source["label"]),
                "artifact_ref": str(checkpoint_source["artifact_ref"]),
                "resolved_checkpoint_path": str(resolved_checkpoint_path),
                "sample_unit": _sample_unit,
                "resolved_device": str(resolved_device),
                "d_model": int(config.model.d_model),
                "proj_dim": int(config.model.proj_dim),
                "n_patches": int(config.data.n_patches),
                "patch_mm": float(config.data.patch_mm),
                "modalities": ",".join(config.data.modality_filter),
                "eval_units_requested": int(effective_n_studies),
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
    Path,
    VIEW_NAMES,
    build_eval_batch,
    build_eval_batch_from_ct_validation_cache,
    build_relative_position_pair_rows,
    catalog_path,
    config,
    effective_eval_batch_size,
    effective_n_studies,
    is_script_mode,
    load_ct_validation_cache,
    masked_l1_per_view,
    modality_filter,
    model,
    mo,
    sample_study4_examples,
    seed,
    time,
    torch,
    validation_cache_dir,
):
    mo.stop(
        str(config.data.sample_unit).strip().lower() != "study4",
        mo.callout("This notebook only supports `study4` checkpoints.", kind="warn"),
    )

    _requested_modalities = tuple(m.strip().upper() for m in str(modality_filter.value).split(",") if m.strip())
    _cache_dir_text = str(validation_cache_dir.value).strip()
    if not _cache_dir_text:
        mo.stop(len(_requested_modalities) == 0, mo.callout("Specify at least one modality.", kind="warn"))

    _modalities = _requested_modalities
    _sampling_seconds = 0.0
    _cache_load_seconds = 0.0
    _forward_seconds = 0.0
    _pack_seconds = 0.0
    _finalize_seconds = 0.0
    _target_studies = int(effective_n_studies)
    _cache = None
    _examples = None
    _data_source = "live_sampling"

    if _cache_dir_text:
        _cache_root = Path(_cache_dir_text).expanduser()
        mo.stop(
            not _cache_root.is_dir(),
            mo.callout(f"Validation cache directory not found: {_cache_root}", kind="danger"),
        )
        _cache_progress_state = {"percent": 0}
        _cache_load_t0 = time.perf_counter()
        with mo.status.progress_bar(
            total=100,
            title="Loading validation cache",
            subtitle="Reading cached CT patch shards from disk...",
            completion_title="Validation cache loaded",
            show_rate=False,
            show_eta=False,
            disabled=bool(is_script_mode),
        ) as _cache_bar:

            def _on_cache_progress(event: dict[str, object]) -> None:
                _stage = str(event.get("stage", "loading"))
                _status = str(event.get("status", "running"))
                _n_shards = max(int(event.get("n_shards", 0)), 1)
                _loaded_shards = int(event.get("loaded_shards", 0))
                if _stage == "metadata" and _status == "complete":
                    _target_percent = 10
                elif _stage == "shards":
                    _target_percent = 10 + int(round(80.0 * (_loaded_shards / _n_shards)))
                elif _stage == "finalize":
                    _target_percent = 100
                else:
                    _target_percent = int(_cache_progress_state["percent"])
                _delta = max(0, _target_percent - int(_cache_progress_state["percent"]))
                _cache_progress_state["percent"] = _target_percent
                _subtitle = (
                    f"stage={_stage} status={_status}"
                    f"{f' | shards {_loaded_shards}/{_n_shards}' if _stage == 'shards' else ''}"
                )
                _cache_bar.update(increment=_delta, subtitle=_subtitle)

            _cache = load_ct_validation_cache(_cache_root, progress=_on_cache_progress)
        _cache_load_seconds = float(time.perf_counter() - _cache_load_t0)
        _available_studies = int(_cache["patches_views"].shape[0])
        _selected_studies = min(_target_studies, _available_studies)
        mo.stop(_selected_studies <= 0, mo.callout("Validation cache is empty.", kind="danger"))
        _modalities = (
            tuple(sorted({str(value).upper() for value in _cache["view_df"]["modality"].to_list()}))
            if "modality" in _cache["view_df"].columns
            else ("CT",)
        ) or ("CT",)
        _data_source = "validation_cache"
    else:
        _sample_progress_state = {"accepted": 0}
        _sampling_t0 = time.perf_counter()
        with mo.status.progress_bar(
            total=max(_target_studies, 1),
            title="Sampling studies",
            subtitle="Loading scans and drawing deterministic study4 views...",
            completion_title="Sampling complete",
            show_rate=True,
            show_eta=True,
            disabled=bool(is_script_mode),
        ) as _sample_bar:

            def _on_sampling_progress(event: dict[str, object]) -> None:
                _accepted = int(event.get("accepted_examples", 0))
                _visited = int(event.get("visited_studies", 0))
                _total = int(event.get("total_candidates", 0))
                _status = str(event.get("status", "running"))
                _study_id = str(event.get("study_id", ""))
                _delta = max(0, _accepted - int(_sample_progress_state["accepted"]))
                _sample_progress_state["accepted"] = _accepted
                _subtitle = (
                    f"accepted {_accepted}/{_target_studies} | visited {_visited}/{_total} candidates | "
                    f"status={_status}{f' | { _study_id }' if _study_id else ''}"
                )
                _sample_bar.update(increment=_delta, subtitle=_subtitle)

            _examples = sample_study4_examples(
                catalog_path.value,
                config,
                n_studies=_target_studies,
                seed=int(seed.value),
                modality_filter=_modalities,
                progress=_on_sampling_progress,
            )
        _sampling_seconds = float(time.perf_counter() - _sampling_t0)
        mo.stop(len(_examples) == 0, mo.callout("No valid study4 examples could be sampled.", kind="danger"))
        _selected_studies = int(len(_examples))

    _all_view_rows: list[dict[str, object]] = []
    _all_sample_rows: list[dict[str, object]] = []
    _all_pair_rows: list[dict[str, object]] = []
    _reconstruction_rows: list[dict[str, object]] = []
    _mim_rows: list[dict[str, object]] = []
    _direction_chunks: list[torch.Tensor] = []
    _model_device = next(model.parameters()).device
    _num_chunks = (_selected_studies + int(effective_eval_batch_size) - 1) // int(effective_eval_batch_size)

    with mo.status.progress_bar(
        total=max(_num_chunks, 1),
        title="Running checkpoint eval",
        subtitle="Building batches and forwarding through the model...",
        completion_title="Checkpoint eval complete",
        show_rate=True,
        show_eta=True,
        disabled=bool(is_script_mode),
    ) as _eval_bar:
        for _chunk_index, _start in enumerate(range(0, _selected_studies, int(effective_eval_batch_size)), start=1):
            _pack_t0 = time.perf_counter()
            _stop = min(_start + int(effective_eval_batch_size), _selected_studies)
            if _cache is not None:
                _batch = build_eval_batch_from_ct_validation_cache(_cache, start=_start, stop=_stop)
            else:
                _chunk = _examples[_start:_stop]
                _batch = build_eval_batch(_chunk, sample_offset=_start)
            _pack_seconds += float(time.perf_counter() - _pack_t0)
            _forward_t0 = time.perf_counter()
            with torch.no_grad():
                _outputs = model.forward_study4(
                    _batch["patches_views"].to(_model_device, dtype=torch.float32),
                    _batch["positions_views"].to(_model_device, dtype=torch.float32),
                    _batch["cross_valid"].to(_model_device),
                )
            _forward_seconds += float(time.perf_counter() - _forward_t0)

            if (
                _outputs.distance_logits_x is None
                or _outputs.distance_logits_y is None
                or _outputs.direction_cls_views is None
            ):
                raise ValueError("study4 forward pass did not return the expected position-decoder outputs")

            _pack_t1 = time.perf_counter()
            _direction_chunks.append(
                F.normalize(
                    _outputs.direction_cls_views.detach().cpu().reshape(-1, _outputs.direction_cls_views.shape[-1]),
                    dim=-1,
                )
            )
            _all_view_rows.extend(_batch["view_rows"])
            _all_sample_rows.extend(_batch["sample_rows"])
            _all_pair_rows.extend(build_relative_position_pair_rows(_batch, _outputs))

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
            _pack_seconds += float(time.perf_counter() - _pack_t1)
            _eval_bar.update(
                increment=1,
                subtitle=(
                    f"chunk {_chunk_index}/{_num_chunks} | studies {_stop}/{_selected_studies} | "
                    f"forward {_forward_seconds:.1f}s | pack {_pack_seconds:.1f}s"
                ),
            )

    _finalize_t0 = time.perf_counter()
    _direction_embeddings = torch.cat(_direction_chunks, dim=0)
    _view_df = pl.DataFrame(_all_view_rows)
    _sample_df = pl.DataFrame(_all_sample_rows)
    _position_pair_df = pl.DataFrame(_all_pair_rows)
    _mim_df = pl.DataFrame(_mim_rows)
    _finalize_seconds = float(time.perf_counter() - _finalize_t0)
    _total_seconds = float(_cache_load_seconds + _sampling_seconds + _forward_seconds + _pack_seconds + _finalize_seconds)

    probe_state = {
        "examples": _examples,
        "view_df": _view_df,
        "sample_df": _sample_df,
        "position_pair_df": _position_pair_df,
        "mim_df": _mim_df,
        "reconstruction_rows": _reconstruction_rows,
        "direction_embeddings": _direction_embeddings,
        "effective_modalities": _modalities,
        "data_source": _data_source,
        "cache_summary": dict(_cache["summary"]) if _cache is not None else None,
        "totalseg_resolved_count": int(sum(bool(row["totalseg_resolved"]) for row in _all_view_rows)),
        "performance_rows": [
            {"stage": "cache_load", "seconds": _cache_load_seconds},
            {"stage": "sampling", "seconds": _sampling_seconds},
            {"stage": "batch_and_metrics_pack", "seconds": _pack_seconds},
            {"stage": "model_forward", "seconds": _forward_seconds},
            {"stage": "finalize_probe_state", "seconds": _finalize_seconds},
            {"stage": "total", "seconds": _total_seconds},
        ],
    }
    return (probe_state,)


@app.cell
def _(mo, probe_state):
    _view_df = probe_state["view_df"]
    _sample_df = probe_state["sample_df"]
    _perf_df = (
        pl.DataFrame(probe_state["performance_rows"])
        .filter(pl.col("seconds") > 0)
        .with_columns(pl.col("seconds").cast(pl.Float64).round(2))
    )
    _cache_summary = probe_state["cache_summary"] or {}
    _eval_summary = pl.DataFrame(
        [
            {
                "samples": int(_sample_df.height),
                "unique_studies": int(_sample_df["study_id"].n_unique()) if _sample_df.height > 0 else 0,
                "views": int(_view_df.height),
                "totalseg_resolved": f"{int(probe_state['totalseg_resolved_count'])}/{int(_view_df.height)}",
                "modalities": ",".join(probe_state["effective_modalities"]),
                "data_source": str(probe_state["data_source"]),
                "cache_dir": str(_cache_summary.get("cache_dir", "")),
                "cache_unique_studies_total": int(_cache_summary.get("n_studies", 0)) if _cache_summary else 0,
                "cache_samples_total": int(_cache_summary.get("n_samples", _cache_summary.get("n_studies", 0)))
                if _cache_summary
                else 0,
                "cache_samples_per_study": int(_cache_summary.get("samples_per_study", 1)) if _cache_summary else 0,
            }
        ]
    )
    mo.vstack([mo.md("## Evaluation Summary"), _eval_summary, _perf_df])
    return


@app.cell
def _(alt, metric_display, mo, pl, probe_state):
    position_pair_df = probe_state["position_pair_df"].filter(pl.col("n_valid_axes") > 0)
    mo.stop(
        position_pair_df.height == 0,
        mo.callout("The trained relative-position decoder did not produce any valid left/right, front/back, or top/bottom targets.", kind="warn"),
    )

    position_series_summary = (
        position_pair_df.group_by(
            [
                "scan_label",
                "series_id",
                "study_id",
                "series_description_display",
                "series_family",
                "native_acquisition_plane",
                "contrast_bucket",
                "series_path",
            ]
        )
        .agg(
            [
                pl.len().alias("pair_count"),
                pl.col("sample_index").n_unique().alias("sample_count"),
                pl.col("pair_accuracy").mean().alias("mean_relative_accuracy"),
                pl.col("pair_accuracy").median().alias("median_relative_accuracy"),
                pl.col("pair_accuracy").min().alias("worst_relative_accuracy"),
                pl.col("left_right_correct").cast(pl.Float64).mean().alias("left_right_accuracy"),
                pl.col("front_back_correct").cast(pl.Float64).mean().alias("front_back_accuracy"),
                pl.col("top_bottom_correct").cast(pl.Float64).mean().alias("top_bottom_accuracy"),
                pl.col("pair_distance_mm").mean().alias("mean_pair_distance_mm"),
            ]
        )
        .sort(
            ["mean_relative_accuracy", "worst_relative_accuracy", "scan_label"],
            descending=[False, False, False],
        )
    )
    _series_ranked = position_series_summary.with_row_index("scan_rank", offset=1)
    _series_table = position_series_summary.select(
        [
            "scan_label",
            "series_family",
            "native_acquisition_plane",
            "contrast_bucket",
            "pair_count",
            "sample_count",
            "mean_relative_accuracy",
            "worst_relative_accuracy",
            "left_right_accuracy",
            "front_back_accuracy",
            "top_bottom_accuracy",
        ]
    ).with_columns(
        [
            pl.col("mean_relative_accuracy").round(3),
            pl.col("worst_relative_accuracy").round(3),
            pl.col("left_right_accuracy").round(3),
            pl.col("front_back_accuracy").round(3),
            pl.col("top_bottom_accuracy").round(3),
        ]
    )
    _scan_rows_table = mo.ui.table(
        _series_table.to_dicts(),
        pagination=True,
        page_size=12,
        wrapped_columns=["scan_label", "series_family"],
        show_download=False,
        max_height=420,
        label="Scan summary (worst to best)",
    )
    _scan_options = position_series_summary["scan_label"].to_list()
    position_scan_picker = mo.ui.dropdown(
        options=_scan_options,
        value=_scan_options[0],
        label="Inspect scan",
    )

    _overall_metrics = pl.DataFrame(
        [
            {"metric": "unique_scans", "value": str(int(position_series_summary.height))},
            {"metric": "evaluated_pairs", "value": str(int(position_pair_df.height))},
            {"metric": "unique_studies", "value": str(int(position_pair_df["study_id"].n_unique()))},
            {"metric": "mean_relative_accuracy", "value": metric_display(float(position_pair_df["pair_accuracy"].mean()))},
            {
                "metric": "left_right_accuracy",
                "value": metric_display(float(position_pair_df["left_right_correct"].drop_nulls().cast(pl.Float64).mean())),
            },
            {
                "metric": "front_back_accuracy",
                "value": metric_display(float(position_pair_df["front_back_correct"].drop_nulls().cast(pl.Float64).mean())),
            },
            {
                "metric": "top_bottom_accuracy",
                "value": metric_display(float(position_pair_df["top_bottom_correct"].drop_nulls().cast(pl.Float64).mean())),
            },
        ]
    )
    _worst_pairs = (
        position_pair_df.sort(
            ["pair_accuracy", "pair_distance_mm", "scan_label", "sample_index"],
            descending=[False, True, False, False],
        )
        .select(
            [
                "scan_label",
                "sample_index",
                "pair_name",
                "pair_accuracy",
                "left_right_truth",
                "left_right_pred",
                "left_right_correct",
                "front_back_truth",
                "front_back_pred",
                "front_back_correct",
                "top_bottom_truth",
                "top_bottom_pred",
                "top_bottom_correct",
                "pair_distance_mm",
            ]
        )
        .with_columns(
            [
                pl.col("pair_accuracy").round(3),
                pl.col("pair_distance_mm").round(1),
            ]
        )
        .head(40)
    )
    _chart = (
        alt.Chart(alt.Data(values=_series_ranked.to_dicts()))
        .mark_circle(size=80, opacity=0.85)
        .encode(
            x=alt.X("scan_rank:Q", title="scan rank (worst to best)"),
            y=alt.Y("mean_relative_accuracy:Q", title="mean relative-position accuracy", scale=alt.Scale(domain=[0.0, 1.0])),
            color=alt.Color(
                "worst_relative_accuracy:Q",
                title="worst pair accuracy",
                scale=alt.Scale(domain=[0.0, 1.0], range=["#b2182b", "#fddbc7", "#1a9850"]),
            ),
            tooltip=[
                alt.Tooltip("scan_label:N", title="scan"),
                alt.Tooltip("series_family:N", title="series family"),
                alt.Tooltip("native_acquisition_plane:N", title="plane"),
                alt.Tooltip("contrast_bucket:N", title="contrast"),
                alt.Tooltip("pair_count:Q", title="pairs"),
                alt.Tooltip("sample_count:Q", title="samples"),
                alt.Tooltip("mean_relative_accuracy:Q", format=".3f"),
                alt.Tooltip("worst_relative_accuracy:Q", format=".3f"),
                alt.Tooltip("left_right_accuracy:Q", format=".3f"),
                alt.Tooltip("front_back_accuracy:Q", format=".3f"),
                alt.Tooltip("top_bottom_accuracy:Q", format=".3f"),
            ],
        )
        .properties(title="Scan-level relative-position accuracy", height=340)
    )
    _note = mo.callout(
        "This section uses the trained distance head directly on the `(a,b)` and `(ap,bp)` pairs. "
        "Accuracy is computed only on axes where the true center delta magnitude is at least 1 mm.",
        kind="info",
    )
    mo.vstack(
        [
            mo.md("## Relative Position Decoder"),
            _note,
            _overall_metrics,
            mo.ui.altair_chart(_chart),
            _scan_rows_table,
            mo.md("### Worst sampled pairs"),
            _worst_pairs,
            position_scan_picker,
        ]
    )
    return position_pair_df, position_scan_picker, position_series_summary


@app.cell
def _(alt, mo, pl, position_pair_df, position_scan_picker, position_series_summary):
    _selected_scan = str(position_scan_picker.value)
    _selected_summary = position_series_summary.filter(pl.col("scan_label") == _selected_scan)
    _selected_pairs = position_pair_df.filter(pl.col("scan_label") == _selected_scan).sort(
        ["pair_accuracy", "sample_index", "pair_name"],
        descending=[False, False, False],
    )
    mo.stop(_selected_pairs.height == 0, mo.callout(f"No sampled pairs found for `{_selected_scan}`.", kind="warn"))

    _summary_table = _selected_summary.select(
        [
            "scan_label",
            "series_family",
            "native_acquisition_plane",
            "contrast_bucket",
            "pair_count",
            "sample_count",
            "mean_relative_accuracy",
            "median_relative_accuracy",
            "worst_relative_accuracy",
            "left_right_accuracy",
            "front_back_accuracy",
            "top_bottom_accuracy",
            "mean_pair_distance_mm",
        ]
    ).with_columns(
        [
            pl.col("mean_relative_accuracy").round(3),
            pl.col("median_relative_accuracy").round(3),
            pl.col("worst_relative_accuracy").round(3),
            pl.col("left_right_accuracy").round(3),
            pl.col("front_back_accuracy").round(3),
            pl.col("top_bottom_accuracy").round(3),
            pl.col("mean_pair_distance_mm").round(1),
        ]
    )

    def _projection_chart(
        pair_df: pl.DataFrame,
        *,
        title: str,
        x_start: str,
        x_end: str,
        y_start: str,
        y_end: str,
        x_mid: str,
        y_mid: str,
        x_title: str,
        y_title: str,
    ):
        _fields = [
            "sample_index",
            "pair_name",
            "pair_accuracy",
            "series_description_display",
            "anchor_view_name",
            "target_view_name",
            "delta_r_mm",
            "delta_a_mm",
            "delta_s_mm",
            "left_right_truth",
            "left_right_pred",
            "left_right_correct",
            "front_back_truth",
            "front_back_pred",
            "front_back_correct",
            "top_bottom_truth",
            "top_bottom_pred",
            "top_bottom_correct",
            x_start,
            x_end,
            y_start,
            y_end,
            x_mid,
            y_mid,
        ]
        _data = pair_df.select(_fields).to_dicts()
        _base = alt.Chart(alt.Data(values=_data))
        _rules = _base.mark_rule(strokeWidth=4, opacity=0.8).encode(
            x=alt.X(f"{x_start}:Q", title=x_title),
            x2=alt.X2(f"{x_end}:Q"),
            y=alt.Y(f"{y_start}:Q", title=y_title),
            y2=alt.Y2(f"{y_end}:Q"),
            color=alt.Color(
                "pair_accuracy:Q",
                title="pair accuracy",
                scale=alt.Scale(domain=[0.0, 1.0], range=["#b2182b", "#fddbc7", "#1a9850"]),
            ),
            tooltip=[
                alt.Tooltip("sample_index:Q", title="sample"),
                alt.Tooltip("pair_name:N", title="pair"),
                alt.Tooltip("pair_accuracy:Q", format=".3f"),
                alt.Tooltip("anchor_view_name:N", title="anchor"),
                alt.Tooltip("target_view_name:N", title="target"),
                alt.Tooltip("delta_r_mm:Q", title="delta R", format=".1f"),
                alt.Tooltip("delta_a_mm:Q", title="delta A", format=".1f"),
                alt.Tooltip("delta_s_mm:Q", title="delta S", format=".1f"),
                alt.Tooltip("left_right_truth:N", title="L/R truth"),
                alt.Tooltip("left_right_pred:N", title="L/R pred"),
                alt.Tooltip("left_right_correct:N", title="L/R ok"),
                alt.Tooltip("front_back_truth:N", title="F/B truth"),
                alt.Tooltip("front_back_pred:N", title="F/B pred"),
                alt.Tooltip("front_back_correct:N", title="F/B ok"),
                alt.Tooltip("top_bottom_truth:N", title="T/B truth"),
                alt.Tooltip("top_bottom_pred:N", title="T/B pred"),
                alt.Tooltip("top_bottom_correct:N", title="T/B ok"),
            ],
        )
        _midpoints = _base.mark_circle(size=70, opacity=0.9, stroke="black", strokeWidth=0.3).encode(
            x=alt.X(f"{x_mid}:Q", title=x_title),
            y=alt.Y(f"{y_mid}:Q", title=y_title),
            color=alt.Color(
                "pair_accuracy:Q",
                title="pair accuracy",
                scale=alt.Scale(domain=[0.0, 1.0], range=["#b2182b", "#fddbc7", "#1a9850"]),
            ),
            tooltip=[
                alt.Tooltip("sample_index:Q", title="sample"),
                alt.Tooltip("pair_name:N", title="pair"),
                alt.Tooltip("pair_accuracy:Q", format=".3f"),
            ],
        )
        return alt.layer(_rules, _midpoints).properties(title=title, height=280)

    _axial_chart = _projection_chart(
        _selected_pairs,
        title="Axial projection (right/left vs front/back)",
        x_start="anchor_r_mm",
        x_end="target_r_mm",
        y_start="anchor_a_mm",
        y_end="target_a_mm",
        x_mid="mid_r_mm",
        y_mid="mid_a_mm",
        x_title="R/L position (mm)",
        y_title="F/B position (mm)",
    )
    _coronal_chart = _projection_chart(
        _selected_pairs,
        title="Coronal projection (right/left vs top/bottom)",
        x_start="anchor_r_mm",
        x_end="target_r_mm",
        y_start="anchor_s_mm",
        y_end="target_s_mm",
        x_mid="mid_r_mm",
        y_mid="mid_s_mm",
        x_title="R/L position (mm)",
        y_title="T/B position (mm)",
    )
    _sagittal_chart = _projection_chart(
        _selected_pairs,
        title="Sagittal projection (front/back vs top/bottom)",
        x_start="anchor_a_mm",
        x_end="target_a_mm",
        y_start="anchor_s_mm",
        y_end="target_s_mm",
        x_mid="mid_a_mm",
        y_mid="mid_s_mm",
        x_title="F/B position (mm)",
        y_title="T/B position (mm)",
    )
    _worst_selected_pairs = _selected_pairs.select(
        [
            "sample_index",
            "pair_name",
            "pair_accuracy",
            "left_right_truth",
            "left_right_pred",
            "left_right_correct",
            "front_back_truth",
            "front_back_pred",
            "front_back_correct",
            "top_bottom_truth",
            "top_bottom_pred",
            "top_bottom_correct",
            "delta_r_mm",
            "delta_a_mm",
            "delta_s_mm",
            "pair_distance_mm",
        ]
    ).with_columns(
        [
            pl.col("pair_accuracy").round(3),
            pl.col("delta_r_mm").round(1),
            pl.col("delta_a_mm").round(1),
            pl.col("delta_s_mm").round(1),
            pl.col("pair_distance_mm").round(1),
        ]
    ).head(24)

    mo.vstack(
        [
            mo.md("## Relative Position Details"),
            _summary_table,
            mo.ui.altair_chart(_axial_chart),
            mo.ui.altair_chart(_coronal_chart),
            mo.ui.altair_chart(_sagittal_chart),
            mo.md("### Worst pairs for the selected scan"),
            _worst_selected_pairs,
        ]
    )
    return


@app.cell
def _(
    alt,
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
