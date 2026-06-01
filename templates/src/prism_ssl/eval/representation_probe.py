"""Representation probing helpers for marimo evaluation notebooks."""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from prism_ssl.config import RunConfig, apply_overrides, load_run_config
from prism_ssl.config.schema import ScanRecord
from prism_ssl.data.catalog import build_scan_id, load_catalog
from prism_ssl.data.preflight import load_nifti_scan
from prism_ssl.data.sample_contract import compute_pair_targets, tensorize_sample_view
from prism_ssl.model import PrismSSLModel
from prism_ssl.utils.hashing import stable_int_hash


@dataclass
class LoadedProbeModel:
    model: PrismSSLModel
    config: RunConfig
    checkpoint_path: Path
    step: int
    device: torch.device


@dataclass
class RepresentationBatch:
    metadata: pl.DataFrame
    cls_embeddings: np.ndarray
    projection_embeddings: np.ndarray
    broken: pl.DataFrame


@dataclass
class PairwisePredictionBatch:
    predictions: pl.DataFrame
    metrics: pl.DataFrame
    broken: pl.DataFrame


_PAIR_FIELD_SPECS = (
    ("center_delta_mm", ("x", "y", "z"), "mm"),
    ("rotation_delta_deg", ("x", "y", "z"), "deg"),
    ("window_delta", ("wc", "ww"), "intensity"),
)


def resolve_device(device_key: str) -> torch.device:
    key = str(device_key).strip().lower()
    if key in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(key)


def run_config_from_flat(flat_config: dict[str, Any]) -> RunConfig:
    """Rehydrate the flattened config stored in PRISM checkpoints."""
    return apply_overrides(RunConfig(), dict(flat_config))


def download_wandb_artifact_checkpoint(artifact_ref: str, root_dir: str | Path | None = None) -> Path:
    """Download a model artifact and return the first checkpoint file."""
    import wandb

    ref = str(artifact_ref).strip()
    if not ref:
        raise ValueError("artifact_ref is empty")
    root = Path(root_dir) if root_dir else Path(tempfile.mkdtemp(prefix="prism_probe_wandb_"))
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(root / "wandb_cache"))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(root / "wandb_artifacts"))

    artifact = wandb.Api(timeout=60).artifact(ref, type="model")
    artifact_dir = Path(artifact.download(root=str(root / "downloads" / _safe_path_token(ref))))
    candidates = sorted(artifact_dir.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in artifact {ref}")
    return candidates[0].resolve()


def resolve_checkpoint_path(
    *,
    checkpoint_path: str,
    artifact_ref: str,
    download_root: str | Path | None = None,
) -> Path:
    path_text = str(checkpoint_path).strip()
    if path_text:
        path = Path(path_text).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        return path
    return download_wandb_artifact_checkpoint(artifact_ref, root_dir=download_root)


def load_probe_model(checkpoint_path: str | Path, device_key: str = "auto") -> LoadedProbeModel:
    device = resolve_device(device_key)
    ckpt = Path(checkpoint_path).expanduser().resolve()
    # Load checkpoint payloads on CPU first so optimizer/cache tensors do not
    # consume MIG/GPU memory during notebook probing.
    payload = torch.load(ckpt, map_location="cpu")
    flat_config = payload.get("config")
    if not isinstance(flat_config, dict):
        raise ValueError(f"Checkpoint missing flat config payload: {ckpt}")
    config = run_config_from_flat(flat_config)
    state_dict = payload["model_state_dict"]
    patch_dim = _checkpoint_patch_dim(state_dict)
    model = PrismSSLModel(
        patch_dim=patch_dim,
        model_name=config.model.name,
        d_model=config.model.d_model,
        proj_dim=config.model.proj_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return LoadedProbeModel(
        model=model,
        config=config,
        checkpoint_path=ckpt,
        step=int(payload.get("step", 0)),
        device=device,
    )


def load_config_or_checkpoint_config(config_path: str, loaded: LoadedProbeModel | None) -> RunConfig:
    if loaded is not None:
        return loaded.config
    return load_run_config(config_path)


def deterministic_catalog_rows(
    catalog_path: str,
    *,
    n_scans: int,
    seed: int,
    modality_filter: tuple[str, ...] = ("CT", "MR"),
) -> list[dict[str, Any]]:
    df = load_catalog(catalog_path)
    if "modality" in df.columns:
        df = df.filter(pl.col("modality").cast(pl.Utf8).str.to_uppercase().is_in([m.upper() for m in modality_filter]))
    if "series_path" in df.columns:
        df = df.filter(pl.col("series_path").is_not_null() & (pl.col("series_path").cast(pl.Utf8).str.len_chars() > 0))
    key_cols = [col for col in ("pmbb_id", "series_path", "series_description") if col in df.columns]
    if not key_cols:
        df = df.with_row_index("_row_idx")
        key_expr = pl.col("_row_idx").cast(pl.Utf8)
    else:
        key_expr = pl.concat_str([pl.col(col).cast(pl.Utf8).fill_null("") for col in key_cols], separator="|")
    df = df.with_columns(key_expr.alias("_probe_key")).with_columns(pl.col("_probe_key").hash(seed=seed).alias("_h"))
    df = df.sort("_h")
    if n_scans > 0:
        df = df.head(min(int(n_scans), df.height))
    return df.drop([col for col in ("_probe_key", "_h") if col in df.columns]).to_dicts()


def scan_record_from_row(row: dict[str, Any]) -> ScanRecord:
    series_path = str(row.get("series_path", ""))
    return ScanRecord(
        scan_id=build_scan_id(row),
        series_id=f"series_{stable_int_hash(series_path) & 0xFFFFFFFF:08x}",
        modality=str(row.get("modality", "CT")).upper(),
        series_path=series_path,
        nifti_path=str(row.get("nifti_path", "") or ""),
    )


def metadata_from_row(row: dict[str, Any], *, view_index: int, scan_id: str, series_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    series_description = _first_text(row, ["series_description", "SeriesDescription", "protocol_name", "ProtocolName"])
    body_part = _first_text(row, ["body_part", "BodyPartExamined", "body_part_examined", "anatomic_region"])
    exam_type = _first_text(row, ["exam_type", "study_description", "StudyDescription", "procedure_description"])
    manufacturer = _first_text(row, ["manufacturer", "Manufacturer"])
    report_text = _first_text(row, ["report", "report_text", "findings", "impression", "rad_report", "ReportText"])
    return {
        "view_id": f"{scan_id}:{view_index}",
        "scan_id": scan_id,
        "series_id": series_id,
        "pmbb_id": str(row.get("pmbb_id", "")),
        "modality": str(row.get("modality", "")).upper(),
        "series_description": series_description,
        "series_family": normalize_series_family(series_description),
        "body_part": normalize_body_part(body_part, series_description, exam_type, report_text),
        "contrast_bucket": infer_contrast_bucket(row, series_description, exam_type, report_text),
        "manufacturer": manufacturer or "unknown",
        "exam_type": exam_type,
        "report_snippet": report_text[:320],
        "native_acquisition_plane": str(sample.get("native_acquisition_plane", "")),
        "sampling_radius_mm": float(sample.get("sampling_radius_mm", 0.0)),
        "wc": float(sample.get("wc", 0.0)),
        "ww": float(sample.get("ww", 0.0)),
    }


def collect_representations(
    *,
    loaded: LoadedProbeModel,
    catalog_path: str,
    n_scans: int,
    views_per_scan: int,
    n_patches: int,
    seed: int,
    batch_size: int,
    modality_filter: tuple[str, ...] = ("CT", "MR"),
) -> RepresentationBatch:
    config = loaded.config
    rows = deterministic_catalog_rows(
        catalog_path,
        n_scans=int(n_scans),
        seed=int(seed),
        modality_filter=modality_filter,
    )
    view_tensors: list[dict[str, torch.Tensor]] = []
    metadata_rows: list[dict[str, Any]] = []
    broken_rows: list[dict[str, Any]] = []
    target_patch_size = _model_target_patch_size(loaded.model)

    for row_idx, row in enumerate(rows):
        record = scan_record_from_row(row)
        try:
            scan, _ = load_nifti_scan(record, base_patch_mm=float(config.data.patch_mm))
            for view_idx in range(int(views_per_scan)):
                view_seed = int(seed) + (row_idx * 10_000) + view_idx
                sample = scan.train_sample(
                    int(n_patches),
                    seed=view_seed,
                    method=config.data.method,
                    apply_native_orientation_hint=config.data.apply_native_orientation_hint,
                    rotation_augmentation_max_degrees=config.data.rotation_augmentation_max_degrees,
                    target_patch_size=target_patch_size,
                )
                tensors = tensorize_sample_view(sample, position_frame=config.data.position_frame_for_model)
                view_tensors.append(tensors)
                metadata_rows.append(
                    metadata_from_row(
                        row,
                        view_index=view_idx,
                        scan_id=record.scan_id,
                        series_id=record.series_id,
                        sample=sample,
                    )
                )
        except Exception as exc:
            broken_rows.append(
                {
                    "scan_id": record.scan_id,
                    "series_path": record.series_path,
                    "error": str(exc),
                }
            )

    if not view_tensors:
        return RepresentationBatch(
            metadata=pl.DataFrame(metadata_rows),
            cls_embeddings=np.zeros((0, int(config.model.d_model)), dtype=np.float32),
            projection_embeddings=np.zeros((0, int(config.model.proj_dim)), dtype=np.float32),
            broken=pl.DataFrame(broken_rows),
        )

    cls_chunks: list[np.ndarray] = []
    proj_chunks: list[np.ndarray] = []
    model = loaded.model
    device = loaded.device
    with torch.inference_mode():
        for start in range(0, len(view_tensors), max(int(batch_size), 1)):
            chunk = view_tensors[start : start + max(int(batch_size), 1)]
            patches = torch.stack([item["patches"] for item in chunk]).to(device=device, dtype=torch.float32)
            positions = torch.stack([item["positions"] for item in chunk]).to(device=device, dtype=torch.float32)
            cls = model.encoder(patches, positions)
            proj = F.normalize(model.proj_head(cls), dim=1)
            cls_chunks.append(cls.detach().cpu().numpy().astype(np.float32, copy=False))
            proj_chunks.append(proj.detach().cpu().numpy().astype(np.float32, copy=False))

    return RepresentationBatch(
        metadata=pl.DataFrame(metadata_rows),
        cls_embeddings=np.concatenate(cls_chunks, axis=0),
        projection_embeddings=np.concatenate(proj_chunks, axis=0),
        broken=pl.DataFrame(broken_rows),
    )


def collect_pairwise_predictions(
    *,
    loaded: LoadedProbeModel,
    catalog_path: str,
    n_scans: int,
    pairs_per_scan: int,
    n_patches: int,
    seed: int,
    batch_size: int,
    modality_filter: tuple[str, ...] = ("CT", "MR"),
) -> PairwisePredictionBatch:
    """Sample A/B views through the training path and compare labels to heads."""
    config = loaded.config
    rows = deterministic_catalog_rows(
        catalog_path,
        n_scans=int(n_scans),
        seed=int(seed),
        modality_filter=modality_filter,
    )
    views_a: list[dict[str, torch.Tensor]] = []
    views_b: list[dict[str, torch.Tensor]] = []
    prediction_rows: list[dict[str, Any]] = []
    broken_rows: list[dict[str, Any]] = []
    target_patch_size = _model_target_patch_size(loaded.model)

    for row_idx, row in enumerate(rows):
        record = scan_record_from_row(row)
        try:
            scan, _ = load_nifti_scan(record, base_patch_mm=float(config.data.patch_mm))
            for pair_idx in range(int(pairs_per_scan)):
                sample_seed = int(seed) + row_idx * 1_000_000 + pair_idx
                sample_kwargs = {
                    "method": config.data.method,
                    "apply_native_orientation_hint": config.data.apply_native_orientation_hint,
                    "rotation_augmentation_max_degrees": config.data.rotation_augmentation_max_degrees,
                    "target_patch_size": target_patch_size,
                }
                result_a = scan.train_sample(int(n_patches), seed=sample_seed * 2, **sample_kwargs)
                result_b = scan.train_sample(int(n_patches), seed=sample_seed * 2 + 1, **sample_kwargs)
                view_a = tensorize_sample_view(result_a, position_frame=config.data.position_frame_for_model)
                view_b = tensorize_sample_view(result_b, position_frame=config.data.position_frame_for_model)
                pair_targets = compute_pair_targets(view_a, view_b)
                row_base = metadata_from_row(
                    row,
                    view_index=pair_idx,
                    scan_id=record.scan_id,
                    series_id=record.series_id,
                    sample=result_a,
                )
                row_base["pair_id"] = f"{record.scan_id}:{pair_idx}"
                row_base["view_a_seed"] = int(sample_seed * 2)
                row_base["view_b_seed"] = int(sample_seed * 2 + 1)
                _add_pair_values(row_base, "target", pair_targets)
                views_a.append(view_a)
                views_b.append(view_b)
                prediction_rows.append(row_base)
        except Exception as exc:
            broken_rows.append(
                {
                    "scan_id": record.scan_id,
                    "series_path": record.series_path,
                    "error": str(exc),
                }
            )

    if not views_a:
        empty = pl.DataFrame(prediction_rows)
        return PairwisePredictionBatch(predictions=empty, metrics=pl.DataFrame([]), broken=pl.DataFrame(broken_rows))

    model = loaded.model
    device = loaded.device
    with torch.inference_mode():
        for start in range(0, len(views_a), max(int(batch_size), 1)):
            end = start + max(int(batch_size), 1)
            chunk_a = views_a[start:end]
            chunk_b = views_b[start:end]
            patches_a = torch.stack([item["patches"] for item in chunk_a]).to(device=device, dtype=torch.float32)
            positions_a = torch.stack([item["positions"] for item in chunk_a]).to(device=device, dtype=torch.float32)
            patches_b = torch.stack([item["patches"] for item in chunk_b]).to(device=device, dtype=torch.float32)
            positions_b = torch.stack([item["positions"] for item in chunk_b]).to(device=device, dtype=torch.float32)
            outputs = model(patches_a, positions_a, patches_b, positions_b)
            pred_values = {
                "center_delta_mm": outputs.center_delta_mm.detach().cpu(),
                "rotation_delta_deg": outputs.rotation_delta_deg.detach().cpu(),
                "window_delta": outputs.window_delta.detach().cpu(),
            }
            for local_idx in range(patches_a.shape[0]):
                row = prediction_rows[start + local_idx]
                _add_pair_values(
                    row,
                    "pred",
                    {key: value[local_idx] for key, value in pred_values.items()},
                )

    for row in prediction_rows:
        for field, axes, _unit in _PAIR_FIELD_SPECS:
            for axis in axes:
                key = f"{field}_{axis}"
                row[f"abs_error_{key}"] = abs(float(row[f"pred_{key}"]) - float(row[f"target_{key}"]))

    predictions = pl.DataFrame(prediction_rows)
    return PairwisePredictionBatch(
        predictions=predictions,
        metrics=pair_prediction_metric_table(predictions),
        broken=pl.DataFrame(broken_rows),
    )


def pca_2d(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    denom = float(np.sum(s**2))
    explained = (s[:2] ** 2) / denom if denom > 0 else np.zeros(2, dtype=np.float32)
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
    return coords[:, :2].astype(np.float32, copy=False), np.asarray(explained[:2], dtype=np.float32)


def projection_table(metadata: pl.DataFrame, embeddings: np.ndarray) -> tuple[pl.DataFrame, np.ndarray]:
    coords, explained = pca_2d(embeddings)
    if metadata.height == 0:
        return metadata, explained
    return metadata.with_columns(
        [
            pl.Series("pc1", coords[:, 0].tolist()),
            pl.Series("pc2", coords[:, 1].tolist()),
        ]
    ), explained


def knn_probe_table(
    metadata: pl.DataFrame,
    embeddings: np.ndarray,
    *,
    label_columns: list[str],
    k: int = 5,
    min_count: int = 2,
) -> pl.DataFrame:
    rows = []
    for label_col in label_columns:
        if label_col not in metadata.columns:
            continue
        labels = [str(v) if v is not None and str(v) else "unknown" for v in metadata[label_col].to_list()]
        metrics = knn_leave_one_out(embeddings, labels, k=k, min_count=min_count)
        rows.append({"label": label_col, **metrics})
    return pl.DataFrame(rows)


def knn_leave_one_out(embeddings: np.ndarray, labels: list[str], *, k: int = 5, min_count: int = 2) -> dict[str, Any]:
    x = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    labels_arr = np.asarray(labels, dtype=object)
    if x.shape[0] <= 1:
        return {"n_eval": 0, "accuracy": None, "macro_recall": None}
    counts = {str(label): int(np.sum(labels_arr == label)) for label in set(labels)}
    eligible = np.asarray([counts[str(label)] >= int(min_count) for label in labels_arr], dtype=bool)
    if not bool(np.any(eligible)):
        return {"n_eval": 0, "accuracy": None, "macro_recall": None}

    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(max(int(k), 1), max(x.shape[0] - 1, 1))
    top = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    preds = []
    for row in top:
        neigh_labels = [str(labels_arr[idx]) for idx in row]
        vote_counts: dict[str, int] = {}
        for label in neigh_labels:
            vote_counts[label] = vote_counts.get(label, 0) + 1
        preds.append(max(vote_counts.items(), key=lambda item: (item[1], item[0]))[0])
    preds_arr = np.asarray(preds, dtype=object)
    correct = (preds_arr == labels_arr) & eligible

    recalls = []
    for label, count in counts.items():
        if count < int(min_count):
            continue
        mask = (labels_arr == label) & eligible
        recalls.append(float(np.mean(correct[mask])) if np.any(mask) else 0.0)
    return {
        "n_eval": int(np.sum(eligible)),
        "accuracy": float(np.mean(correct[eligible])),
        "macro_recall": float(np.mean(recalls)) if recalls else None,
    }


def nearest_neighbor_table(
    metadata: pl.DataFrame,
    embeddings: np.ndarray,
    *,
    anchor_index: int,
    k: int = 12,
) -> pl.DataFrame:
    if metadata.height == 0:
        return pl.DataFrame([])
    x = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    idx = int(np.clip(anchor_index, 0, max(x.shape[0] - 1, 0)))
    sim = x @ x[idx]
    order = np.argsort(-sim)
    rows = []
    for rank, row_idx in enumerate(order[: max(int(k), 1)]):
        base = metadata.row(int(row_idx), named=True)
        base["rank"] = int(rank)
        base["anchor"] = bool(row_idx == idx)
        base["cosine_similarity"] = float(sim[row_idx])
        rows.append(base)
    return pl.DataFrame(rows)


def embedding_similarity_pair_table(
    metadata: pl.DataFrame,
    embeddings: np.ndarray,
    *,
    max_pairs: int = 50_000,
    seed: int = 0,
) -> pl.DataFrame:
    """Sample pairwise cosine similarities with interpretable relationship labels."""
    if metadata.height < 2:
        return pl.DataFrame(
            schema={
                "left_view_id": pl.Utf8,
                "right_view_id": pl.Utf8,
                "left_series_family": pl.Utf8,
                "right_series_family": pl.Utf8,
                "left_body_part": pl.Utf8,
                "right_body_part": pl.Utf8,
                "left_modality": pl.Utf8,
                "right_modality": pl.Utf8,
                "pair_type": pl.Utf8,
                "cosine_similarity": pl.Float64,
            }
        )
    x = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    n = int(x.shape[0])
    total_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(int(seed))
    rows_as_dicts = metadata.to_dicts()

    if total_pairs <= int(max_pairs):
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        chosen: set[tuple[int, int]] = set()
        target = min(int(max_pairs), total_pairs)
        while len(chosen) < target:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1
            chosen.add((min(i, j), max(i, j)))
        pairs = sorted(chosen)

    out = []
    for i, j in pairs:
        left = rows_as_dicts[i]
        right = rows_as_dicts[j]
        out.append(
            {
                "left_view_id": str(left.get("view_id", i)),
                "right_view_id": str(right.get("view_id", j)),
                "left_series_family": str(left.get("series_family", "unknown")),
                "right_series_family": str(right.get("series_family", "unknown")),
                "left_body_part": str(left.get("body_part", "unknown")),
                "right_body_part": str(right.get("body_part", "unknown")),
                "left_modality": str(left.get("modality", "unknown")),
                "right_modality": str(right.get("modality", "unknown")),
                "pair_type": _metadata_pair_type(left, right),
                "cosine_similarity": float(np.dot(x[i], x[j])),
            }
        )
    return pl.DataFrame(out)


def embedding_similarity_summary_table(pair_table: pl.DataFrame) -> pl.DataFrame:
    if pair_table.height == 0 or "pair_type" not in pair_table.columns:
        return pl.DataFrame(
            schema={
                "pair_type": pl.Utf8,
                "n_pairs": pl.UInt32,
                "mean_cosine": pl.Float64,
                "median_cosine": pl.Float64,
                "p10_cosine": pl.Float64,
                "p90_cosine": pl.Float64,
            }
        )
    return (
        pair_table.group_by("pair_type")
        .agg(
            [
                pl.len().alias("n_pairs"),
                pl.col("cosine_similarity").mean().alias("mean_cosine"),
                pl.col("cosine_similarity").median().alias("median_cosine"),
                pl.col("cosine_similarity").quantile(0.10).alias("p10_cosine"),
                pl.col("cosine_similarity").quantile(0.90).alias("p90_cosine"),
            ]
        )
        .sort("mean_cosine", descending=True)
    )


def pair_prediction_long_table(predictions: pl.DataFrame) -> pl.DataFrame:
    if predictions.height == 0:
        return pl.DataFrame(
            schema={
                "pair_id": pl.Utf8,
                "scan_id": pl.Utf8,
                "series_family": pl.Utf8,
                "body_part": pl.Utf8,
                "field": pl.Utf8,
                "axis": pl.Utf8,
                "unit": pl.Utf8,
                "target": pl.Float64,
                "pred": pl.Float64,
                "abs_error": pl.Float64,
            }
        )
    rows = []
    for row in predictions.to_dicts():
        for field, axes, unit in _PAIR_FIELD_SPECS:
            for axis in axes:
                key = f"{field}_{axis}"
                rows.append(
                    {
                        "pair_id": row.get("pair_id", ""),
                        "scan_id": row.get("scan_id", ""),
                        "series_family": row.get("series_family", "unknown"),
                        "body_part": row.get("body_part", "unknown"),
                        "field": field,
                        "axis": axis,
                        "unit": unit,
                        "target": float(row.get(f"target_{key}", 0.0)),
                        "pred": float(row.get(f"pred_{key}", 0.0)),
                        "abs_error": float(row.get(f"abs_error_{key}", 0.0)),
                    }
                )
    return pl.DataFrame(rows)


def pair_prediction_metric_table(predictions: pl.DataFrame) -> pl.DataFrame:
    long = pair_prediction_long_table(predictions)
    if long.height == 0:
        return pl.DataFrame(
            schema={
                "field": pl.Utf8,
                "axis": pl.Utf8,
                "n": pl.Int64,
                "mae": pl.Float64,
                "rmse": pl.Float64,
                "bias": pl.Float64,
                "target_std": pl.Float64,
                "pred_std": pl.Float64,
                "pred_to_target_std": pl.Float64,
                "pearson": pl.Float64,
                "sign_accuracy": pl.Float64,
            }
        )
    rows = []
    for (field, axis), group in long.group_by(["field", "axis"], maintain_order=True):
        target = np.asarray(group["target"].to_list(), dtype=np.float64)
        pred = np.asarray(group["pred"].to_list(), dtype=np.float64)
        err = pred - target
        target_std = float(np.std(target))
        pred_std = float(np.std(pred))
        if target.size > 1 and target_std > 1e-8 and pred_std > 1e-8:
            corr = float(np.corrcoef(target, pred)[0, 1])
        else:
            corr = None
        nonzero = np.abs(target) > 1e-6
        sign_acc = float(np.mean(np.sign(pred[nonzero]) == np.sign(target[nonzero]))) if bool(np.any(nonzero)) else None
        rows.append(
            {
                "field": str(field),
                "axis": str(axis),
                "n": int(target.size),
                "mae": float(np.mean(np.abs(err))) if target.size else None,
                "rmse": float(np.sqrt(np.mean(err**2))) if target.size else None,
                "bias": float(np.mean(err)) if target.size else None,
                "target_std": target_std,
                "pred_std": pred_std,
                "pred_to_target_std": pred_std / max(target_std, 1e-8),
                "pearson": corr,
                "sign_accuracy": sign_acc,
            }
        )
    return pl.DataFrame(rows)


def label_count_table(metadata: pl.DataFrame, label_columns: list[str], *, top_k: int = 20) -> pl.DataFrame:
    rows = []
    for col in label_columns:
        if col not in metadata.columns:
            continue
        vc = metadata.group_by(col).len().sort("len", descending=True).head(int(top_k))
        for item in vc.to_dicts():
            rows.append({"label": col, "value": str(item.get(col, "unknown")), "count": int(item["len"])})
    return pl.DataFrame(rows)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-8, None)


def _checkpoint_patch_dim(state_dict: dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("encoder.patch_proj.weight")
    if weight is None or getattr(weight, "ndim", 0) != 2:
        return 256
    return int(weight.shape[1])


def _model_target_patch_size(model: PrismSSLModel) -> int:
    patch_dim = int(model.encoder.patch_proj.in_features)
    side = int(round(math.sqrt(patch_dim)))
    if side * side != patch_dim:
        raise ValueError(f"Model patch_dim must be square for 2D patch sampling, got {patch_dim}")
    return side


def _add_pair_values(row: dict[str, Any], prefix: str, values: dict[str, torch.Tensor]) -> None:
    for field, axes, _unit in _PAIR_FIELD_SPECS:
        arr = values[field]
        if isinstance(arr, torch.Tensor):
            arr_np = arr.detach().cpu().float().numpy().reshape(-1)
        else:
            arr_np = np.asarray(arr, dtype=np.float32).reshape(-1)
        for idx, axis in enumerate(axes):
            row[f"{prefix}_{field}_{axis}"] = float(arr_np[idx])


def _metadata_pair_type(left: dict[str, Any], right: dict[str, Any]) -> str:
    if str(left.get("scan_id", "")) and str(left.get("scan_id", "")) == str(right.get("scan_id", "")):
        return "same_scan"
    if str(left.get("series_id", "")) and str(left.get("series_id", "")) == str(right.get("series_id", "")):
        return "same_series_id"
    left_family = str(left.get("series_family", "unknown"))
    right_family = str(right.get("series_family", "unknown"))
    if left_family != "unknown" and left_family == right_family:
        return "same_series_family"
    left_body = str(left.get("body_part", "unknown"))
    right_body = str(right.get("body_part", "unknown"))
    if left_body != "unknown" and left_body == right_body:
        return "same_body_part"
    if str(left.get("modality", "")) and str(left.get("modality", "")) == str(right.get("modality", "")):
        return "same_modality_only"
    return "different"


def _safe_path_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", text)


def _first_text(row: dict[str, Any], columns: list[str]) -> str:
    for col in columns:
        value = row.get(col)
        if value is not None and str(value).strip() and str(value).lower() != "null":
            return str(value).strip()
    return ""


def normalize_series_family(series_description: str) -> str:
    text = _normalize_text(series_description)
    if not text:
        return "unknown"
    text = re.sub(r"\b\d+([._-]\d+)?\b", "", text)
    text = re.sub(r"\b(i|b|br|bv|q|kernel|thin|mm|axial|ax|cor|sag)\d*\w*\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:80] if text else "unknown"


def normalize_body_part(*texts: str) -> str:
    joined = " ".join(_normalize_text(text) for text in texts if text)
    if not joined:
        return "unknown"
    buckets = [
        ("head_neck", ["head", "brain", "neck", "face", "sinus", "cspine", "c spine"]),
        ("chest", ["chest", "thorax", "lung", "pulmonary", "cardiac", "heart", "coronary"]),
        ("abdomen", ["abdomen", "abd", "liver", "kidney", "renal", "pancreas"]),
        ("pelvis", ["pelvis", "hip", "bladder", "prostate"]),
        ("spine", ["spine", "lumbar", "thoracic", "sacrum"]),
        ("extremity", ["knee", "ankle", "foot", "hand", "wrist", "shoulder", "elbow", "arm", "leg"]),
    ]
    for bucket, needles in buckets:
        if any(needle in joined for needle in needles):
            return bucket
    return joined.split(" ")[0][:40]


def infer_contrast_bucket(row: dict[str, Any], *texts: str) -> str:
    explicit = _first_text(row, ["contrast", "contrast_bolus_agent", "ContrastBolusAgent", "contrast_bucket"])
    joined = " ".join([explicit, *(_normalize_text(text) for text in texts if text)]).lower()
    if any(token in joined for token in ["non contrast", "noncontrast", "without contrast", "wo contrast", "w/o contrast"]):
        return "non_contrast"
    if any(token in joined for token in ["with contrast", "w contrast", "venous", "arterial", "post contrast", "cta", "angio"]):
        return "contrast"
    return "unknown"


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
