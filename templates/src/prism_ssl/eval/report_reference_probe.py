"""Helpers for exploring report-reference weak labels in notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import polars as pl


@dataclass(frozen=True)
class SlicePreview:
    image_rgb: np.ndarray
    axis: int
    axis_name: str
    slice_index: int
    window_center: float
    window_width: float
    shape: tuple[int, int, int]
    note: str


def default_report_ref_manifest_candidates(repo_root: str | Path) -> list[Path]:
    root = Path(repo_root).expanduser().resolve()
    return [
        root / "results" / "report_refs" / "pmbb_report_refs.parquet",
        root / "results" / "report_refs" / "report_refs_smoke_default.parquet",
        root / "results" / "report_refs" / "report_refs_smoke_small.parquet",
    ]


def resolve_report_ref_manifest_path(path_text: str, repo_root: str | Path) -> Path:
    explicit = str(path_text).strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Report-reference manifest not found: {path}")
    for candidate in default_report_ref_manifest_candidates(repo_root):
        if candidate.exists():
            return candidate.resolve()
    candidates = "\n".join(str(p) for p in default_report_ref_manifest_candidates(repo_root))
    raise FileNotFoundError(f"No report-reference manifest found. Tried:\n{candidates}")


def load_report_ref_manifest(path: str | Path) -> pl.DataFrame:
    p = Path(path).expanduser()
    if p.suffix.lower() == ".parquet":
        return pl.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pl.read_csv(p)
    if p.suffix.lower() == ".tsv":
        return pl.read_csv(p, separator="\t")
    raise ValueError(f"Unsupported report-reference manifest suffix: {p.suffix}")


def filter_report_refs(
    df: pl.DataFrame,
    *,
    confidence_filter: str = "high_exact_instance",
    modality_filter: str = "ALL",
    organ_filter: str = "ALL",
    section_contains: str = "",
) -> pl.DataFrame:
    out = df
    if confidence_filter == "mapped":
        out = out.filter(pl.col("slice_mapping_confidence") != "unmapped")
    elif confidence_filter != "all":
        out = out.filter(pl.col("slice_mapping_confidence") == confidence_filter)

    modality = str(modality_filter).strip().upper()
    if modality and modality != "ALL":
        out = out.filter(pl.col("modality").cast(pl.Utf8).str.to_uppercase() == modality)

    organ = str(organ_filter).strip()
    if organ and organ != "ALL":
        out = out.filter(pl.col("organ_hint") == organ)

    section = str(section_contains).strip().lower()
    if section:
        out = out.filter(pl.col("report_section").cast(pl.Utf8).str.to_lowercase().str.contains(section, literal=True))
    return out


def count_table(df: pl.DataFrame, columns: list[str], *, top_k: int = 40) -> pl.DataFrame:
    available = [col for col in columns if col in df.columns]
    if not available or df.height == 0:
        return pl.DataFrame({"count": []})
    return df.group_by(available).len(name="count").sort("count", descending=True).head(int(top_k))


def row_at(df: pl.DataFrame, index: int) -> dict[str, Any]:
    if df.height == 0:
        raise IndexError("Cannot select from an empty report-reference table")
    idx = int(np.clip(index, 0, df.height - 1))
    return df.row(idx, named=True)


@lru_cache(maxsize=4096)
def read_report_text(report_path: str, max_chars: int = 0) -> str:
    path = Path(str(report_path)).expanduser()
    text = path.read_text(encoding="utf-8", errors="replace")
    if int(max_chars) > 0 and len(text) > int(max_chars):
        return text[: int(max_chars)] + "\n\n[TRUNCATED]"
    return text


def report_ref_table_dataframe(
    df: pl.DataFrame,
    *,
    limit: int = 500,
    full_report_max_chars: int = 12_000,
) -> pl.DataFrame:
    """Create a UI-table friendly report-ref dataframe with full report text."""
    if df.height == 0:
        return pl.DataFrame([])
    base = df.with_row_index("_filtered_row_index").head(max(int(limit), 1))
    wanted = [
        "_filtered_row_index",
        "modality",
        "organ_hint",
        "report_section",
        "sentence",
        "full_report",
        "slice_mapping_confidence",
        "series_match_confidence",
        "series_number_reported",
        "image_number_reported",
        "dicom_instance_number",
        "slice_axis_name",
        "canonical_slice_index",
        "series_description",
        "report_path",
        "nifti_path",
    ]
    rows: list[dict[str, Any]] = []
    for row in base.to_dicts():
        report_path = str(row.get("report_path") or "")
        row["full_report"] = read_report_text(report_path, int(full_report_max_chars)) if report_path else ""
        rows.append({col: row.get(col) for col in wanted})
    return pl.DataFrame(rows)


def selected_table_row(table_value: Any, fallback_df: pl.DataFrame) -> dict[str, Any]:
    """Normalize marimo table.value to a selected row dict."""
    if table_value is None:
        return row_at(fallback_df, 0)
    if isinstance(table_value, pl.DataFrame):
        if table_value.height > 0:
            return table_value.row(0, named=True)
        return row_at(fallback_df, 0)
    if isinstance(table_value, dict):
        # marimo may return column-oriented data for dataframe-backed tables.
        if table_value:
            lengths = [len(v) for v in table_value.values() if isinstance(v, list)]
            if lengths and min(lengths) > 0:
                return {k: (v[0] if isinstance(v, list) else v) for k, v in table_value.items()}
        return row_at(fallback_df, 0)
    if isinstance(table_value, list) and table_value:
        first = table_value[0]
        if isinstance(first, dict):
            return first
    return row_at(fallback_df, 0)


def load_report_ref_slice_preview(
    row: dict[str, Any],
    *,
    window_center: float | None = None,
    window_width: float | None = None,
    max_display_px: int = 640,
    show_slice_anchor: bool = False,
) -> SlicePreview:
    nifti_path = str(row.get("nifti_path") or "")
    if not nifti_path:
        raise FileNotFoundError("Selected report-reference row has no nifti_path")
    axis = _as_int(row.get("slice_axis"), default=2)
    slice_index = _as_int(row.get("canonical_slice_index"), default=-1)
    if slice_index < 0:
        raise ValueError("Selected report-reference row has no canonical_slice_index")

    img = nib.as_closest_canonical(nib.load(nifti_path))
    volume = np.asarray(img.dataobj, dtype=np.float32)
    shape = tuple(int(v) for v in volume.shape[:3])
    axis = int(np.clip(axis, 0, 2))
    slice_index = int(np.clip(slice_index, 0, shape[axis] - 1))
    slice_2d = _take_slice(volume, axis, slice_index)

    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=np.float32)
    if window_center is None or window_width is None or float(window_width) <= 0:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        wc = float((lo + hi) * 0.5)
        ww = float(max(hi - lo, 1e-3))
    else:
        wc = float(window_center)
        ww = float(window_width)

    image = _window_to_rgb(slice_2d, wc, ww)
    if show_slice_anchor:
        voxel = _row_voxel(row)
        if voxel is not None:
            _draw_cross_on_slice(image, axis=axis, voxel=voxel, color=(255, 64, 64), radius=4)
    image = _resize_nearest(image, max_display_px=max_display_px)
    axis_name = ("x", "y", "z")[axis]
    note = (
        "This is the referenced DICOM slice. The optional marker is the DICOM "
        "ImagePositionPatient slice anchor, not a lesion coordinate."
    )
    return SlicePreview(
        image_rgb=image,
        axis=axis,
        axis_name=axis_name,
        slice_index=slice_index,
        window_center=wc,
        window_width=ww,
        shape=shape,
        note=note,
    )


def _as_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return default
        return int(value)
    except Exception:
        return default


def _take_slice(volume: np.ndarray, axis: int, slice_index: int) -> np.ndarray:
    if axis == 0:
        return np.asarray(volume[slice_index, :, :], dtype=np.float32)
    if axis == 1:
        return np.asarray(volume[:, slice_index, :], dtype=np.float32)
    return np.asarray(volume[:, :, slice_index], dtype=np.float32)


def _window_to_rgb(slice_2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
    width = max(float(ww), 1e-6)
    low = float(wc) - 0.5 * width
    high = float(wc) + 0.5 * width
    clipped = np.clip(slice_2d, low, high)
    gray = ((clipped - low) / max(high - low, 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _row_voxel(row: dict[str, Any]) -> tuple[int, int, int] | None:
    keys = ("canonical_voxel_x", "canonical_voxel_y", "canonical_voxel_z")
    vals = []
    for key in keys:
        value = row.get(key)
        if value is None:
            return None
        try:
            vals.append(int(value))
        except Exception:
            return None
    return tuple(vals)  # type: ignore[return-value]


def _draw_cross_on_slice(
    image: np.ndarray,
    *,
    axis: int,
    voxel: tuple[int, int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    if axis == 0:
        row, col = voxel[1], voxel[2]
    elif axis == 1:
        row, col = voxel[0], voxel[2]
    else:
        row, col = voxel[0], voxel[1]
    h, w = image.shape[:2]
    if row < 0 or row >= h or col < 0 or col >= w:
        return
    r0 = max(0, int(row) - radius)
    r1 = min(h, int(row) + radius + 1)
    c0 = max(0, int(col) - radius)
    c1 = min(w, int(col) + radius + 1)
    image[r0:r1, int(col)] = np.asarray(color, dtype=np.uint8)
    image[int(row), c0:c1] = np.asarray(color, dtype=np.uint8)


def _resize_nearest(image: np.ndarray, *, max_display_px: int) -> np.ndarray:
    max_px = max(int(max_display_px), 64)
    h, w = image.shape[:2]
    scale = min(max_px / max(h, w), 1.0)
    if scale >= 1.0:
        return image
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    row_idx = np.linspace(0, h - 1, new_h).round().astype(np.int64)
    col_idx = np.linspace(0, w - 1, new_w).round().astype(np.int64)
    return image[row_idx][:, col_idx]
