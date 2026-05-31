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
from prism_ssl.data.sample_contract import tensorize_sample_view
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
    payload = torch.load(ckpt, map_location=device)
    flat_config = payload.get("config")
    if not isinstance(flat_config, dict):
        raise ValueError(f"Checkpoint missing flat config payload: {ckpt}")
    config = run_config_from_flat(flat_config)
    target_patch_size = int(math.sqrt(256))
    model = PrismSSLModel(
        patch_dim=target_patch_size * target_patch_size,
        model_name=config.model.name,
        d_model=config.model.d_model,
        proj_dim=config.model.proj_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    )
    model.load_state_dict(payload["model_state_dict"])
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
                    target_patch_size=16,
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
