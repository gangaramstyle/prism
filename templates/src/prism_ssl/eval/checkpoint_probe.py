"""Checkpoint evaluation helpers for study4 notebook analysis."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import nibabel as nib
import numpy as np
import polars as pl
import torch

from prism_ssl.config import RunConfig, load_run_config_from_flat
from prism_ssl.data import collate_prism_batch, load_catalog, load_nifti_scan
from prism_ssl.data.catalog import build_scan_id, series_id_from_row, study_id_from_row
from prism_ssl.data.filters import filter_modalities, filter_nonempty_series_path
from prism_ssl.data.preflight import resolve_totalseg_total_ct_path, world_points_to_voxel
from prism_ssl.data.sample_contract import build_study4_dataset_item
from prism_ssl.model import PrismModelOutput, PrismSSLModel
from prism_ssl.utils.hashing import stable_int_hash

VIEW_NAMES: tuple[str, ...] = ("a", "ap", "b", "bp")
_TOTALSEG_CACHE: dict[str, np.ndarray | None] = {}
_WANDB_ARTIFACT_LIST_CACHE: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
_WANDB_CHECKPOINT_CACHE: dict[str, Path] = {}
_SESSION_TMP_DIR = Path(
    tempfile.mkdtemp(
        prefix="prism_ssl_checkpoint_probe_",
        dir=os.environ.get("TMPDIR", tempfile.gettempdir()),
    )
)
atexit.register(lambda: shutil.rmtree(_SESSION_TMP_DIR, ignore_errors=True))


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    key = str(device).strip().lower()
    if key in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(key)


def _ensure_wandb_tmp_env() -> None:
    for key, relative in (
        ("WANDB_DIR", "wandb"),
        ("WANDB_CACHE_DIR", "wandb_cache"),
        ("WANDB_ARTIFACT_DIR", "wandb_artifacts"),
        ("TMPDIR", "tmp"),
    ):
        os.environ.setdefault(key, str(_SESSION_TMP_DIR / relative))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _load_wandb_module() -> Any:
    _ensure_wandb_tmp_env()
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised via caller-facing error paths
        raise RuntimeError("wandb package is required for W&B checkpoint loading") from exc
    return wandb


def parse_wandb_run_ref(run_ref: str) -> tuple[str, str, str]:
    """Parse a W&B run URL or entity/project/run_id reference."""
    text = str(run_ref).strip()
    if not text:
        raise ValueError("W&B run reference is empty")

    if "://" in text:
        parsed = urlparse(text)
        parts = [part for part in parsed.path.split("/") if part]
    else:
        parts = [part for part in text.split("/") if part]

    if len(parts) < 3:
        raise ValueError(f"Could not parse W&B run reference: {text}")

    if len(parts) >= 4 and parts[-2] == "runs":
        entity, project, run_id = parts[-4], parts[-3], parts[-1]
    else:
        entity, project, run_id = parts[-3], parts[-2], parts[-1]

    run_id = run_id.strip()
    if not entity or not project or not run_id:
        raise ValueError(f"Could not parse W&B run reference: {text}")
    return entity, project, run_id


def _artifact_version_number(version: str) -> int:
    match = re.fullmatch(r"v(\d+)", str(version).strip())
    return int(match.group(1)) if match is not None else -1


def _artifact_step_from_aliases(aliases: Sequence[str]) -> int | None:
    steps = []
    for alias in aliases:
        match = re.fullmatch(r"step-(\d+)", str(alias).strip())
        if match is not None:
            steps.append(int(match.group(1)))
    if not steps:
        return None
    return max(steps)


def _artifact_display_name(name: str, version: str, step: int | None) -> str:
    if step is not None:
        return f"step-{step} | {name}"
    if version:
        return f"{version} | {name}"
    return name


def list_wandb_run_model_artifacts(
    run_ref: str,
    *,
    timeout: int = 60,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """List model artifacts logged by a W&B run, newest checkpoint first."""
    entity, project, run_id = parse_wandb_run_ref(run_ref)
    cache_key = (entity, project, run_id)
    if not force_refresh and cache_key in _WANDB_ARTIFACT_LIST_CACHE:
        return [dict(row) for row in _WANDB_ARTIFACT_LIST_CACHE[cache_key]]

    wandb = _load_wandb_module()
    api = wandb.Api(timeout=int(timeout))
    run = api.run(f"{entity}/{project}/{run_id}")

    artifacts: list[dict[str, Any]] = []
    for artifact in run.logged_artifacts():
        if str(getattr(artifact, "type", "")) != "model":
            continue
        name = str(getattr(artifact, "name", ""))
        version = str(getattr(artifact, "version", ""))
        aliases = [str(alias) for alias in list(getattr(artifact, "aliases", []) or [])]
        step = _artifact_step_from_aliases(aliases)
        artifacts.append(
            {
                "entity": entity,
                "project": project,
                "run_id": run_id,
                "artifact_name": name,
                "artifact_ref": f"{entity}/{project}/{name}",
                "version": version,
                "aliases": aliases,
                "step": step,
                "display_name": _artifact_display_name(name, version, step),
            }
        )

    artifacts.sort(
        key=lambda item: (
            item["step"] is None,
            -(int(item["step"]) if item["step"] is not None else -1),
            -_artifact_version_number(str(item["version"])),
            str(item["artifact_name"]),
        )
    )
    _WANDB_ARTIFACT_LIST_CACHE[cache_key] = [dict(row) for row in artifacts]
    return [dict(row) for row in artifacts]


def download_wandb_run_checkpoint(
    run_ref: str,
    artifact_ref: str,
    *,
    timeout: int = 60,
    force_refresh: bool = False,
) -> Path:
    """Download a W&B model artifact checkpoint into a session-local tmp dir."""
    entity, project, run_id = parse_wandb_run_ref(run_ref)
    full_artifact_ref = str(artifact_ref).strip()
    if not full_artifact_ref:
        raise ValueError("artifact_ref must be a non-empty W&B artifact reference")
    if "/" not in full_artifact_ref:
        full_artifact_ref = f"{entity}/{project}/{full_artifact_ref}"

    cache_key = f"{run_id}|{full_artifact_ref}"
    cached = _WANDB_CHECKPOINT_CACHE.get(cache_key)
    if not force_refresh and cached is not None and cached.is_file():
        return cached

    wandb = _load_wandb_module()
    api = wandb.Api(timeout=int(timeout))
    artifact = api.artifact(full_artifact_ref, type="model")

    safe_name = full_artifact_ref.replace("/", "__").replace(":", "__")
    download_root = _SESSION_TMP_DIR / "downloads" / run_id / safe_name
    if force_refresh and download_root.exists():
        shutil.rmtree(download_root, ignore_errors=True)
    download_root.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(artifact.download(root=str(download_root)))
    candidates = sorted(artifact_dir.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files were found in W&B artifact {full_artifact_ref}")

    ckpt_path = candidates[0].resolve()
    _WANDB_CHECKPOINT_CACHE[cache_key] = ckpt_path
    return ckpt_path


def load_checkpoint_payload(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a training checkpoint payload onto the requested device."""
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=_resolve_device(device))


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[PrismSSLModel, RunConfig, dict[str, Any], torch.device]:
    """Rebuild a PrismSSLModel from a saved checkpoint."""
    resolved_device = _resolve_device(device)
    payload = load_checkpoint_payload(checkpoint_path, device=resolved_device)
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Checkpoint payload is missing a flat 'config' dictionary")
    if "model_state_dict" not in payload:
        raise ValueError("Checkpoint payload is missing 'model_state_dict'")

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
    model.to(resolved_device)
    model.eval()
    return model, config, payload, resolved_device


def _pair_curriculum(config: RunConfig, sample_index: int) -> tuple[float, float]:
    steps = max(0, int(config.data.pair_local_curriculum_steps))
    if steps <= 0 or float(config.data.pair_local_final_prob) <= 0.0:
        return 0.0, float(config.data.pair_local_start_radius_mm)
    progress = min(max(int(sample_index), 0) / float(steps), 1.0)
    local_prob = float(config.data.pair_local_final_prob) * progress
    radius_mm = float(config.data.pair_local_start_radius_mm) + (
        float(config.data.pair_local_end_radius_mm) - float(config.data.pair_local_start_radius_mm)
    ) * progress
    return float(local_prob), float(max(radius_mm, 0.0))


def _sample_pair_center_vox(
    scan: Any,
    center_a_vox: np.ndarray,
    *,
    seed: int,
    sample_index: int,
    config: RunConfig,
) -> tuple[np.ndarray, bool]:
    rng = np.random.default_rng(int(seed))
    center_a = np.asarray(center_a_vox, dtype=np.int64).reshape(3)
    local_prob, local_radius_mm = _pair_curriculum(config, sample_index)
    if local_prob > 0.0 and float(rng.random()) < local_prob:
        local_center, local_from_body = scan.sample_center_near_with_source(
            rng,
            center_a,
            radius_mm=local_radius_mm,
            patch_vox=scan.patch_shape_vox,
        )
        if np.any(local_center != center_a):
            return local_center, bool(local_from_body)

    for _ in range(32):
        center_b, center_b_from_body = scan.sample_prism_center_with_source(rng, patch_vox=scan.patch_shape_vox)
        if np.any(center_b != center_a):
            return center_b, bool(center_b_from_body)

    return scan.sample_center_near_with_source(
        rng,
        center_a,
        radius_mm=max(local_radius_mm, float(config.data.patch_mm) * 2.0),
        patch_vox=scan.patch_shape_vox,
    )


def _try_same_world_view(scan: Any, world_center_pt: np.ndarray, *, seed: int, n_patches: int) -> Mapping[str, Any] | None:
    center_vox = np.rint(world_points_to_voxel(np.asarray(world_center_pt, dtype=np.float32), scan.affine)[0]).astype(np.int64)
    if not scan._patch_has_overlap(center_vox, scan.patch_shape_vox):  # noqa: SLF001
        return None
    min_idx, max_idx = scan._center_bounds_for_full_patch(scan.patch_shape_vox)  # noqa: SLF001
    center_vox = np.clip(center_vox, min_idx, max_idx)
    if not scan._patch_has_overlap(center_vox, scan.patch_shape_vox):  # noqa: SLF001
        return None
    return scan.train_sample(
        n_patches,
        seed=seed,
        subset_center_vox=center_vox,
        sampled_body_center=scan.contains_body_center_vox(center_vox),
    )


def _row_to_scan_record(row: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(row)
    return {
        "scan_id": build_scan_id(record),
        "series_id": series_id_from_row(record),
        "study_id": study_id_from_row(record),
        "modality": str(record.get("modality", "CT")).upper(),
        "series_path": str(record.get("series_path", "")),
        "nifti_path": str(record.get("nifti_path", "") or ""),
    }


def _series_label_text(row: Mapping[str, Any]) -> str:
    description = str(row.get("series_description", "")).strip()
    if description:
        return description
    return Path(str(row.get("series_path", ""))).name or "unknown"


def _select_series_pair(group_rows: Sequence[Mapping[str, Any]], *, seed: int, study_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    ordered_rows = sorted(
        group_rows,
        key=lambda row: stable_int_hash(
            f"{seed}|{study_id}|{row.get('series_path', '')}|{row.get('series_description', '')}"
        ),
    )
    row_x = ordered_rows[0]
    series_x = series_id_from_row(dict(row_x))
    distinct = [row for row in ordered_rows if series_id_from_row(dict(row)) != series_x]
    if distinct:
        return row_x, distinct[0], "paired_world"
    return row_x, row_x, "duplicate"


def sample_study4_examples(
    catalog: str | Path | pl.DataFrame,
    config: RunConfig,
    *,
    n_studies: int,
    seed: int,
    modality_filter: tuple[str, ...] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Sample deterministic same-study examples with training-like 4-view logic."""
    sample_unit = str(config.data.sample_unit).strip().lower()
    if sample_unit != "study4":
        raise ValueError(f"sample_study4_examples requires data.sample_unit='study4', got '{config.data.sample_unit}'")

    df = load_catalog(str(catalog)) if isinstance(catalog, (str, Path)) else catalog
    modalities = tuple(str(m).upper() for m in (modality_filter or config.data.modality_filter))
    filtered = filter_nonempty_series_path(filter_modalities(df, modalities))
    if len(filtered) == 0 or int(n_studies) <= 0:
        return []

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in filtered.to_dicts():
        groups.setdefault(study_id_from_row(row), []).append(dict(row))

    ordered_studies = sorted(groups, key=lambda study_id: stable_int_hash(f"{seed}|{study_id}"))
    examples: list[dict[str, Any]] = []
    if progress is not None:
        progress(
            {
                "event": "start",
                "visited_studies": 0,
                "accepted_examples": 0,
                "target_examples": int(n_studies),
                "total_candidates": int(len(ordered_studies)),
                "study_id": "",
                "status": "starting",
            }
        )
    visited_studies = 0
    for study_order, study_id in enumerate(ordered_studies):
        if len(examples) >= int(n_studies):
            break
        visited_studies = int(study_order + 1)

        row_x, row_y, fallback_mode = _select_series_pair(groups[study_id], seed=seed, study_id=study_id)
        record_x = _row_to_scan_record(row_x)
        record_y = _row_to_scan_record(row_y)

        try:
            scan_x, resolved_nifti_path_x = load_nifti_scan(
                record_x,
                base_patch_mm=float(config.data.patch_mm),
                use_totalseg_body_centers=bool(config.data.use_totalseg_body_centers),
            )
            if record_y["series_id"] == record_x["series_id"]:
                scan_y = scan_x
                resolved_nifti_path_y = resolved_nifti_path_x
            else:
                scan_y, resolved_nifti_path_y = load_nifti_scan(
                    record_y,
                    base_patch_mm=float(config.data.patch_mm),
                    use_totalseg_body_centers=bool(config.data.use_totalseg_body_centers),
                )
        except Exception:
            if progress is not None:
                progress(
                    {
                        "event": "study",
                        "visited_studies": visited_studies,
                        "accepted_examples": int(len(examples)),
                        "target_examples": int(n_studies),
                        "total_candidates": int(len(ordered_studies)),
                        "study_id": str(study_id),
                        "status": "load_failed",
                    }
                )
            continue

        study_seed = stable_int_hash(f"{seed}|{study_id}") % 2_147_483_647
        result_a = scan_x.train_sample(int(config.data.n_patches), seed=study_seed * 4)
        pair_center_b_vox, pair_center_b_from_body = _sample_pair_center_vox(
            scan_x,
            np.asarray(result_a["prism_center_vox"], dtype=np.int64),
            seed=study_seed * 4 + 1,
            sample_index=study_order,
            config=config,
        )
        result_b = scan_x.train_sample(
            int(config.data.n_patches),
            seed=study_seed * 4 + 2,
            subset_center_vox=pair_center_b_vox,
            sampled_body_center=pair_center_b_from_body,
        )

        cross_valid = False
        cross_mode = str(fallback_mode)
        if record_y["series_id"] != record_x["series_id"]:
            result_ap = _try_same_world_view(
                scan_y,
                np.asarray(result_a["prism_center_pt"], dtype=np.float32),
                seed=study_seed * 4 + 3,
                n_patches=int(config.data.n_patches),
            )
            result_bp = _try_same_world_view(
                scan_y,
                np.asarray(result_b["prism_center_pt"], dtype=np.float32),
                seed=study_seed * 4 + 4,
                n_patches=int(config.data.n_patches),
            )
            if result_ap is not None and result_bp is not None:
                cross_valid = True
                cross_mode = "paired_world"
            else:
                result_ap = scan_y.train_sample(int(config.data.n_patches), seed=study_seed * 4 + 5)
                result_bp = scan_y.train_sample(int(config.data.n_patches), seed=study_seed * 4 + 6)
                cross_mode = "unpaired"
        else:
            result_ap = scan_y.train_sample(int(config.data.n_patches), seed=study_seed * 4 + 5)
            result_bp = scan_y.train_sample(int(config.data.n_patches), seed=study_seed * 4 + 6)

        views = [
            {
                "view_name": "a",
                "series_id": record_x["series_id"],
                "series_path": record_x["series_path"],
                "series_description": str(row_x.get("series_description", "")),
                "series_label_text": _series_label_text(row_x),
                "resolved_nifti_path": resolved_nifti_path_x,
                "totalseg_path": resolve_totalseg_total_ct_path(record_x["series_path"]),
                "result": result_a,
            },
            {
                "view_name": "ap",
                "series_id": record_y["series_id"],
                "series_path": record_y["series_path"],
                "series_description": str(row_y.get("series_description", "")),
                "series_label_text": _series_label_text(row_y),
                "resolved_nifti_path": resolved_nifti_path_y,
                "totalseg_path": resolve_totalseg_total_ct_path(record_y["series_path"]),
                "result": result_ap,
            },
            {
                "view_name": "b",
                "series_id": record_x["series_id"],
                "series_path": record_x["series_path"],
                "series_description": str(row_x.get("series_description", "")),
                "series_label_text": _series_label_text(row_x),
                "resolved_nifti_path": resolved_nifti_path_x,
                "totalseg_path": resolve_totalseg_total_ct_path(record_x["series_path"]),
                "result": result_b,
            },
            {
                "view_name": "bp",
                "series_id": record_y["series_id"],
                "series_path": record_y["series_path"],
                "series_description": str(row_y.get("series_description", "")),
                "series_label_text": _series_label_text(row_y),
                "resolved_nifti_path": resolved_nifti_path_y,
                "totalseg_path": resolve_totalseg_total_ct_path(record_y["series_path"]),
                "result": result_bp,
            },
        ]
        examples.append(
            {
                "study_id": study_id,
                "series_id_x": record_x["series_id"],
                "series_id_y": record_y["series_id"],
                "series_path_x": record_x["series_path"],
                "series_path_y": record_y["series_path"],
                "series_description_x": str(row_x.get("series_description", "")),
                "series_description_y": str(row_y.get("series_description", "")),
                "cross_valid": bool(cross_valid),
                "cross_mode": str(cross_mode),
                "views": views,
            }
        )
        if progress is not None:
            progress(
                {
                    "event": "study",
                    "visited_studies": visited_studies,
                    "accepted_examples": int(len(examples)),
                    "target_examples": int(n_studies),
                    "total_candidates": int(len(ordered_studies)),
                    "study_id": str(study_id),
                    "status": "accepted",
                }
            )
    if progress is not None:
        progress(
            {
                "event": "done",
                "visited_studies": visited_studies,
                "accepted_examples": int(len(examples)),
                "target_examples": int(n_studies),
                "total_candidates": int(len(ordered_studies)),
                "study_id": "",
                "status": "complete",
            }
        )
    return examples


def _dominant_label(values: np.ndarray) -> int:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    if unique.size == 0:
        return 0
    return int(unique[np.argmax(counts)])


def _load_totalseg_volume(ts_path: str) -> np.ndarray | None:
    cached = _TOTALSEG_CACHE.get(ts_path)
    if ts_path in _TOTALSEG_CACHE:
        return cached
    if not ts_path or not Path(ts_path).is_file():
        _TOTALSEG_CACHE[ts_path] = None
        return None
    try:
        raw = nib.load(ts_path)
        try:
            img = nib.as_closest_canonical(raw)
        except Exception:
            img = raw
        seg = np.asarray(img.dataobj)
        if seg.ndim == 4:
            seg = seg[..., 0]
        if seg.ndim != 3:
            _TOTALSEG_CACHE[ts_path] = None
            return None
        out = np.asarray(seg, dtype=np.int16)
        _TOTALSEG_CACHE[ts_path] = out
        return out
    except Exception:
        _TOTALSEG_CACHE[ts_path] = None
        return None


def _view_field(view: Mapping[str, Any], key: str) -> Any:
    if key in view:
        return view[key]
    result = view.get("result")
    if isinstance(result, Mapping) and key in result:
        return result[key]
    return None


def _labels_at_voxels(seg: np.ndarray, coords: np.ndarray) -> np.ndarray:
    points = np.asarray(coords, dtype=np.int64)
    if points.ndim == 1:
        points = points[np.newaxis, :]
    if points.size == 0:
        return np.zeros(0, dtype=np.int64)
    max_idx = np.asarray(seg.shape, dtype=np.int64) - 1
    clipped = np.clip(points, 0, max_idx)
    return seg[clipped[:, 0], clipped[:, 1], clipped[:, 2]].astype(np.int64, copy=False)


def dominant_totalseg_label_for_view(view: Mapping[str, Any]) -> int:
    """Vote a dominant TS label from patch centers, falling back to prism center."""
    series_path = str(view.get("series_path", ""))
    ts_path = str(view.get("totalseg_path") or resolve_totalseg_total_ct_path(series_path))
    seg = _load_totalseg_volume(ts_path)
    if seg is None:
        return 0

    patch_centers = _view_field(view, "patch_centers_vox")
    if patch_centers is not None:
        patch_labels = _labels_at_voxels(seg, np.asarray(patch_centers, dtype=np.int64))
        patch_labels = patch_labels[patch_labels > 0]
        if patch_labels.size > 0:
            return _dominant_label(patch_labels)

    prism_center = _view_field(view, "prism_center_vox")
    if prism_center is None:
        return 0
    prism_labels = _labels_at_voxels(seg, np.asarray(prism_center, dtype=np.int64))
    if prism_labels.size == 0:
        return 0
    prism_label = int(prism_labels[0])
    return prism_label if prism_label > 0 else 0


def build_eval_batch(examples: Sequence[Mapping[str, Any]], *, sample_offset: int = 0) -> dict[str, Any]:
    """Convert sampled study4 examples into one collated batch plus metadata rows."""
    items: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for local_index, example in enumerate(examples):
        views = list(example["views"])
        item = build_study4_dataset_item(
            result_a=views[0]["result"],
            result_ap=views[1]["result"],
            result_b=views[2]["result"],
            result_bp=views[3]["result"],
            study_id=str(example["study_id"]),
            series_id_x=str(example["series_id_x"]),
            series_id_y=str(example["series_id_y"]),
            cross_valid=bool(example["cross_valid"]),
            cross_mode=str(example["cross_mode"]),
        )
        items.append(item)

        sample_index = int(sample_offset + local_index)
        sample_rows.append(
            {
                "sample_index": sample_index,
                "study_id": str(example["study_id"]),
                "series_id_x": str(example["series_id_x"]),
                "series_id_y": str(example["series_id_y"]),
                "series_description_x": str(example.get("series_description_x", "")),
                "series_description_y": str(example.get("series_description_y", "")),
                "cross_valid": bool(example["cross_valid"]),
                "cross_mode": str(example["cross_mode"]),
            }
        )

        for view_index, view in enumerate(views):
            result = view["result"]
            view_rows.append(
                {
                    "sample_index": sample_index,
                    "view_index": int(view_index),
                    "view_name": str(view["view_name"]),
                    "study_id": str(example["study_id"]),
                    "series_id": str(view["series_id"]),
                    "series_path": str(view["series_path"]),
                    "series_description": str(view.get("series_description", "")),
                    "series_label_text": str(view.get("series_label_text", "")),
                    "resolved_nifti_path": str(view.get("resolved_nifti_path", "")),
                    "totalseg_path": str(view.get("totalseg_path", "")),
                    "totalseg_resolved": bool(view.get("totalseg_path")),
                    "anatomy_label": int(dominant_totalseg_label_for_view(view)),
                    "cross_valid": bool(example["cross_valid"]),
                    "cross_mode": str(example["cross_mode"]),
                    "sampled_body_center": bool(result.get("sampled_body_center", False)),
                    "prism_center_vox": tuple(int(v) for v in np.asarray(result["prism_center_vox"]).tolist()),
                    "prism_center_pt": tuple(float(v) for v in np.asarray(result["prism_center_pt"]).tolist()),
                }
            )

    batch = collate_prism_batch(items)
    batch["view_rows"] = view_rows
    batch["sample_rows"] = sample_rows
    return batch


def pca_project(embeddings: torch.Tensor | np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Project embeddings with PCA using numpy SVD."""
    x = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 embeddings, got shape={x.shape}")
    if x.shape[0] == 0:
        return np.zeros((0, int(n_components)), dtype=np.float32), np.zeros(int(n_components), dtype=np.float32)

    x_centered = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 1:
        return np.zeros((1, int(n_components)), dtype=np.float32), np.zeros(int(n_components), dtype=np.float32)

    _, s, vh = np.linalg.svd(x_centered, full_matrices=False)
    k = min(int(n_components), vh.shape[0])
    proj = x_centered @ vh[:k].T
    explained = (s**2) / max(x.shape[0] - 1, 1)
    explained_ratio = explained / explained.sum() if float(explained.sum()) > 0.0 else np.zeros_like(explained)

    if k < int(n_components):
        proj = np.pad(proj, ((0, 0), (0, int(n_components) - k)))
        explained_ratio = np.pad(explained_ratio[:k], (0, int(n_components) - k))
    else:
        explained_ratio = explained_ratio[: int(n_components)]
    return np.asarray(proj, dtype=np.float32), np.asarray(explained_ratio, dtype=np.float32)


def _normalized_embeddings(embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
    x = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 embeddings, got shape={x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-6, None)


def _label_mask(labels: Sequence[Any], ignore_labels: set[Any] | None = None) -> np.ndarray:
    ignored = ignore_labels or set()
    return np.asarray([label not in ignored for label in labels], dtype=bool)


def nearest_neighbor_purity(
    embeddings: torch.Tensor | np.ndarray,
    labels: Sequence[Any],
    *,
    ignore_labels: set[Any] | None = None,
) -> float:
    """Compute top-1 nearest-neighbor purity for the valid subset."""
    mask = _label_mask(labels, ignore_labels=ignore_labels)
    x = _normalized_embeddings(embeddings)[mask]
    labels_arr = np.asarray(list(labels), dtype=object)[mask]
    if x.shape[0] < 2:
        return float("nan")
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argmax(sim, axis=1)
    return float(np.mean(labels_arr == labels_arr[nn_idx]))


def within_between_cosine_gap(
    embeddings: torch.Tensor | np.ndarray,
    labels: Sequence[Any],
    *,
    ignore_labels: set[Any] | None = None,
) -> float:
    """Return mean(within-label cosine) - mean(between-label cosine)."""
    mask = _label_mask(labels, ignore_labels=ignore_labels)
    x = _normalized_embeddings(embeddings)[mask]
    labels_arr = np.asarray(list(labels), dtype=object)[mask]
    if x.shape[0] < 2:
        return float("nan")
    sim = x @ x.T
    eye = np.eye(x.shape[0], dtype=bool)
    same = labels_arr[:, None] == labels_arr[None, :]
    within = sim[same & ~eye]
    between = sim[~same & ~eye]
    if within.size == 0 or between.size == 0:
        return float("nan")
    return float(within.mean() - between.mean())


def _stack_masked_l1(preds: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(preds) != len(targets):
        raise ValueError("Prediction and target tuple lengths must match")
    values = []
    for pred, target in zip(preds, targets):
        if pred.shape != target.shape:
            raise ValueError(f"Prediction shape {tuple(pred.shape)} does not match target shape {tuple(target.shape)}")
        values.append(torch.mean(torch.abs(pred - target).reshape(pred.shape[0], -1), dim=1))
    if not values:
        return torch.empty((0, 0))
    return torch.stack(values, dim=1)


def masked_l1_per_view(
    outputs: PrismModelOutput,
    cross_valid: torch.Tensor | Sequence[bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-sample, per-view masked reconstruction L1 values."""
    self_l1 = _stack_masked_l1(outputs.mim_self_preds, outputs.mim_self_targets)
    register_l1 = _stack_masked_l1(outputs.mim_register_preds, outputs.mim_register_targets)
    cross_l1 = _stack_masked_l1(outputs.mim_cross_preds, outputs.mim_cross_targets)

    if cross_valid is None:
        cross_valid_tensor = torch.ones(self_l1.shape[0], dtype=torch.bool, device=self_l1.device)
    elif torch.is_tensor(cross_valid):
        cross_valid_tensor = cross_valid.to(device=self_l1.device, dtype=torch.bool).reshape(-1)
    else:
        cross_valid_tensor = torch.as_tensor(list(cross_valid), dtype=torch.bool, device=self_l1.device).reshape(-1)

    cross_valid_mask = cross_valid_tensor[:, None].expand(-1, cross_l1.shape[1]) if cross_l1.numel() > 0 else torch.empty((0, 0), dtype=torch.bool)
    return {
        "self": self_l1,
        "register": register_l1,
        "cross": cross_l1,
        "cross_valid_mask": cross_valid_mask,
    }


__all__ = [
    "VIEW_NAMES",
    "load_checkpoint_payload",
    "build_model_from_checkpoint",
    "parse_wandb_run_ref",
    "list_wandb_run_model_artifacts",
    "download_wandb_run_checkpoint",
    "sample_study4_examples",
    "build_eval_batch",
    "dominant_totalseg_label_for_view",
    "pca_project",
    "nearest_neighbor_purity",
    "within_between_cosine_gap",
    "masked_l1_per_view",
]
