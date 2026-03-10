import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


with app.setup:
    import json
    import os
    import sys
    import time
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

    SUPCON_GROUP_OPTIONS = [
        "series_label_text",
        "series_family",
        "native_acquisition_plane",
        "contrast_bucket",
        "series_family|plane",
        "series_family|contrast",
        "series_family|plane|contrast",
    ]

    def _normalize_group_part(value: object, *, fallback: str) -> str:
        text = str(value or "").strip()
        return text if text else fallback

    def prepare_supcon_view_df(view_df: pl.DataFrame) -> pl.DataFrame:
        series_labels = []
        series_families = []
        contrast_buckets = []
        acquisition_planes = []
        family_plane = []
        family_contrast = []
        family_plane_contrast = []
        for row in view_df.to_dicts():
            series_label = _normalize_group_part(
                row.get("series_label_text") or row.get("series_description"),
                fallback=infer_series_family(row),
            )
            series_family = _normalize_group_part(row.get("series_family"), fallback=infer_series_family(row))
            contrast_bucket = _normalize_group_part(row.get("contrast_bucket"), fallback=infer_contrast_bucket(row))
            acquisition_plane = _normalize_group_part(
                row.get("native_acquisition_plane"),
                fallback="unknown_plane",
            )
            series_labels.append(series_label)
            series_families.append(series_family)
            contrast_buckets.append(contrast_bucket)
            acquisition_planes.append(acquisition_plane)
            family_plane.append(f"{series_family} | {acquisition_plane}")
            family_contrast.append(f"{series_family} | {contrast_bucket}")
            family_plane_contrast.append(f"{series_family} | {acquisition_plane} | {contrast_bucket}")

        return view_df.with_columns(
            [
                pl.Series("series_label_text", series_labels),
                pl.Series("series_family", series_families),
                pl.Series("contrast_bucket", contrast_buckets),
                pl.Series("native_acquisition_plane", acquisition_planes),
                pl.Series("series_family_plane", family_plane),
                pl.Series("series_family_contrast", family_contrast),
                pl.Series("series_family_plane_contrast", family_plane_contrast),
            ]
        )


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    intro = mo.md(
        """
# Prism SSL Study4 Checkpoint Probe

Load a local checkpoint or a W&B run artifact, run deterministic `study4` sampling, and inspect:
- SupCon clustering by series label
- Anatomy/view CLS clustering by dominant TotalSegmentator label ID
- Masked-patch reconstruction quality for self/register/cross paths
"""
    )
    intro
    return (is_script_mode,)


@app.cell
def _(Path, SUPCON_GROUP_OPTIONS, mo, os):
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
    n_studies = mo.ui.slider(start=1, stop=512, step=1, value=32, label="Studies")
    eval_batch_size = mo.ui.slider(start=1, stop=32, step=1, value=4, label="Eval batch size")
    seed = mo.ui.number(label="Seed", value=42, step=1)
    supcon_group_by = mo.ui.dropdown(
        options=SUPCON_GROUP_OPTIONS,
        value="series_family|plane|contrast",
        label="SupCon group by",
    )
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
            mo.hstack([device, n_studies, eval_batch_size, seed, supcon_group_by, include_background_label_0]),
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
        supcon_group_by,
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
    Path,
    VIEW_NAMES,
    build_eval_batch,
    build_eval_batch_from_ct_validation_cache,
    catalog_path,
    config,
    cosine_similarity_matrix,
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
    _reconstruction_rows: list[dict[str, object]] = []
    _mim_rows: list[dict[str, object]] = []
    _supcon_chunks: list[torch.Tensor] = []
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

            if _outputs.proj_views is None or _outputs.direction_cls_views is None:
                raise ValueError("study4 forward pass did not return the expected view embeddings")

            _pack_t1 = time.perf_counter()
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
            _pack_seconds += float(time.perf_counter() - _pack_t1)
            _eval_bar.update(
                increment=1,
                subtitle=(
                    f"chunk {_chunk_index}/{_num_chunks} | studies {_stop}/{_selected_studies} | "
                    f"forward {_forward_seconds:.1f}s | pack {_pack_seconds:.1f}s"
                ),
            )

    _finalize_t0 = time.perf_counter()
    _supcon_embeddings = torch.cat(_supcon_chunks, dim=0)
    _direction_embeddings = torch.cat(_direction_chunks, dim=0)
    _supcon_similarity = cosine_similarity_matrix(_supcon_embeddings).detach().cpu().numpy()
    _view_df = pl.DataFrame(_all_view_rows)
    _sample_df = pl.DataFrame(_all_sample_rows)
    _mim_df = pl.DataFrame(_mim_rows)
    _finalize_seconds = float(time.perf_counter() - _finalize_t0)
    _total_seconds = float(_cache_load_seconds + _sampling_seconds + _forward_seconds + _pack_seconds + _finalize_seconds)

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
        "data_source": _data_source,
        "cache_summary": dict(_cache["summary"]) if _cache is not None else None,
        "totalseg_resolved_count": int(sum(bool(row["totalseg_resolved"]) for row in _all_view_rows)),
        "performance_rows": [
            {"stage": "cache_load", "seconds": _cache_load_seconds},
            {"stage": "sampling", "seconds": _sampling_seconds},
            {"stage": "batch_and_metrics_pack", "seconds": _pack_seconds},
            {"stage": "model_forward", "seconds": _forward_seconds},
            {"stage": "finalize_similarity_tables", "seconds": _finalize_seconds},
            {"stage": "total", "seconds": _total_seconds},
        ],
    }
    return (probe_state,)


@app.cell
def _(mo, probe_state):
    _view_df = probe_state["view_df"]
    _perf_df = (
        pl.DataFrame(probe_state["performance_rows"])
        .filter(pl.col("seconds") > 0)
        .with_columns(pl.col("seconds").cast(pl.Float64).round(2))
    )
    _cache_summary = probe_state["cache_summary"] or {}
    _eval_summary = pl.DataFrame(
        [
            {
                "samples": int(probe_state["sample_df"].height),
                "views": int(_view_df.height),
                "totalseg_resolved": f"{int(probe_state['totalseg_resolved_count'])}/{int(_view_df.height)}",
                "modalities": ",".join(probe_state["effective_modalities"]),
                "data_source": str(probe_state["data_source"]),
                "cache_dir": str(_cache_summary.get("cache_dir", "")),
                "cache_studies_total": int(_cache_summary.get("n_studies", 0)) if _cache_summary else 0,
            }
        ]
    )
    mo.vstack([mo.md("## Evaluation Summary"), _eval_summary, _perf_df])
    return


@app.cell
def _(
    alt,
    bucket_top_labels,
    metric_display,
    mo,
    nearest_neighbor_purity,
    pca_project,
    prepare_supcon_view_df,
    probe_state,
    similarity_frame,
    supcon_group_by,
    within_between_cosine_gap,
):
    _view_df = prepare_supcon_view_df(probe_state["view_df"])
    _supcon_embeddings = probe_state["supcon_embeddings"]
    _coords, _explained = pca_project(_supcon_embeddings)
    _group_by = str(supcon_group_by.value)
    _group_column = {
        "series_label_text": "series_label_text",
        "series_family": "series_family",
        "native_acquisition_plane": "native_acquisition_plane",
        "contrast_bucket": "contrast_bucket",
        "series_family|plane": "series_family_plane",
        "series_family|contrast": "series_family_contrast",
        "series_family|plane|contrast": "series_family_plane_contrast",
    }[_group_by]
    _series_labels = _view_df[_group_column].to_list()
    _supcon_df = _view_df.with_columns(
        [
            pl.Series("pc1", _coords[:, 0]),
            pl.Series("pc2", _coords[:, 1]),
            pl.Series("supcon_group_label", _series_labels),
            pl.Series("series_bucket", bucket_top_labels(_series_labels, top_k=12)),
            pl.Series(
                "view_key",
                [f"s{sample_idx}:{view_name}" for sample_idx, view_name in zip(_view_df["sample_index"], _view_df["view_name"])],
            ),
        ]
    )
    _purity = nearest_neighbor_purity(_supcon_embeddings, _series_labels)
    _gap = within_between_cosine_gap(_supcon_embeddings, _series_labels)
    _breakdown_df = (
        _supcon_df.group_by(
            [
                "supcon_group_label",
                "series_family",
                "native_acquisition_plane",
                "contrast_bucket",
                "series_label_text",
            ]
        )
        .agg(
            [
                pl.len().alias("view_count"),
                pl.col("sample_index").n_unique().alias("sample_count"),
                pl.col("series_description").first().alias("example_series_description"),
            ]
        )
        .sort(["sample_count", "view_count", "supcon_group_label"], descending=[True, True, False])
    )

    _scatter = (
        alt.Chart(alt.Data(values=_supcon_df.to_dicts()))
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("pc1:Q", title=f"PC1 ({_explained[0] * 100:.1f}%)"),
            y=alt.Y("pc2:Q", title=f"PC2 ({_explained[1] * 100:.1f}%)"),
            color=alt.Color("series_bucket:N", title="Series group"),
            tooltip=[
                alt.Tooltip("sample_index:Q", title="sample"),
                alt.Tooltip("view_name:N", title="view"),
                alt.Tooltip("supcon_group_label:N", title="selected group"),
                alt.Tooltip("series_label_text:N", title="series label"),
                alt.Tooltip("series_family:N", title="series family"),
                alt.Tooltip("native_acquisition_plane:N", title="plane"),
                alt.Tooltip("contrast_bucket:N", title="contrast"),
                alt.Tooltip("series_description:N", title="series description"),
                alt.Tooltip("study_id:N", title="study_id"),
                alt.Tooltip("series_path:N", title="series_path"),
            ],
        )
        .properties(title=f"SupCon PCA ({_group_by})", height=360)
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
            {"metric": "group_by", "value": _group_by},
            {"metric": "nearest_neighbor_purity", "value": metric_display(_purity)},
            {"metric": "within_between_cosine_gap", "value": metric_display(_gap)},
        ]
    )

    mo.vstack(
        [
            mo.md("## SupCon Clustering"),
            _metrics,
            _breakdown_df,
            mo.ui.altair_chart(_scatter),
            mo.ui.altair_chart(_heatmap),
        ]
    )
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
