"""Build weak report-reference labels from PMBB reports and DICOM metadata.

The target use case is report text such as "lesion on series 5 image 23".
When the series can be matched to a NIfTI-producing series directory and the
image number can be matched to a DICOM instance, this module emits a weak
slice-level caption row with a canonical RAS voxel coordinate.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha1
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import polars as pl

from prism_ssl.data.catalog import load_catalog
from prism_ssl.data.filters import filter_modalities, filter_nonempty_series_path


REPORT_REF_COLUMNS = [
    "pmbb_id",
    "study_dir",
    "report_path",
    "report_section",
    "organ_hint",
    "sentence",
    "reference_text",
    "series_number_reported",
    "series_suffix_reported",
    "image_number_reported",
    "image_number_raw_text",
    "series_match_confidence",
    "series_match_count",
    "series_match_rank",
    "series_number_matched",
    "series_description",
    "modality",
    "body_part",
    "series_path",
    "nifti_path",
    "tree_path",
    "slice_mapping_confidence",
    "confidence_reason",
    "dicom_instance_number",
    "slice_axis",
    "slice_axis_name",
    "canonical_voxel_x",
    "canonical_voxel_y",
    "canonical_voxel_z",
    "canonical_slice_index",
    "canonical_shape_x",
    "canonical_shape_y",
    "canonical_shape_z",
]
REPORT_REF_INT_COLUMNS = {
    "series_number_reported",
    "image_number_reported",
    "series_match_count",
    "series_match_rank",
    "dicom_instance_number",
    "slice_axis",
    "canonical_voxel_x",
    "canonical_voxel_y",
    "canonical_voxel_z",
    "canonical_slice_index",
    "canonical_shape_x",
    "canonical_shape_y",
    "canonical_shape_z",
}
REPORT_REF_SCHEMA = {
    col: (pl.Int64 if col in REPORT_REF_INT_COLUMNS else pl.Utf8)
    for col in REPORT_REF_COLUMNS
}


SERIES_IMAGE_RE = re.compile(
    r"\bseries\s+(?P<series>\d+[A-Za-z]?)\s*[,;:]?\s*"
    r"(?:and\s+)?(?:images?|imgs?|im)\s+(?P<images>\d+(?:\s*(?:,|and|&|to|-)\s*\d+)*)",
    flags=re.IGNORECASE,
)
SECTION_RE = re.compile(r"(?im)^\s*([A-Za-z][A-Za-z0-9 /_-]{1,80})\s*:")

ORGAN_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("liver", ("liver", "hepatic")),
    ("pancreas", ("pancreas", "pancreatic")),
    ("kidney", ("kidney", "renal")),
    ("spleen", ("spleen", "splenic")),
    ("adrenal", ("adrenal",)),
    ("gallbladder", ("gallbladder", "gall bladder")),
    ("bile_duct", ("bile duct", "biliary", "common duct", "cbd")),
    ("bowel", ("bowel", "small bowel", "intestine", "intestinal")),
    ("colon", ("colon", "colonic", "rectum", "rectal")),
    ("appendix", ("appendix", "appendiceal")),
    ("bladder", ("bladder", "urinary bladder")),
    ("prostate", ("prostate", "prostatic")),
    ("uterus", ("uterus", "uterine", "endometrium")),
    ("ovary", ("ovary", "ovarian", "adnexa", "adnexal")),
    ("lung", ("lung", "pulmonary", "pleural")),
    ("heart", ("heart", "cardiac", "pericard")),
    ("aorta", ("aorta", "aortic")),
    ("vessel", ("vessel", "vascular", "artery", "arterial", "vein", "venous")),
    ("lymph_node", ("lymph node", "lymphadenopathy", "node", "nodal")),
    ("bone", ("bone", "osseous", "rib", "pelvis", "femur")),
    ("spine", ("spine", "spinal", "vertebra", "vertebral")),
    ("brain", ("brain", "intracranial", "cerebral")),
    ("thyroid", ("thyroid",)),
    ("breast", ("breast",)),
)


@dataclass(frozen=True)
class ReportReference:
    report_path: str
    section: str
    organ_hint: str
    sentence: str
    reference_text: str
    series_number: int
    series_suffix: str
    image_number: int
    image_number_raw_text: str


@dataclass(frozen=True)
class SeriesInfo:
    series_path: str
    metadata_path: str
    tree_path: str
    nifti_path: str
    series_number: int | None
    series_number_text: str
    series_suffix: str
    series_description: str
    modality: str


@dataclass(frozen=True)
class SliceMapping:
    confidence: str
    reason: str
    dicom_instance_number: int | None
    slice_axis: int | None
    slice_axis_name: str
    canonical_voxel: tuple[int | None, int | None, int | None]
    canonical_slice_index: int | None
    canonical_shape: tuple[int | None, int | None, int | None]


def _stable_shard(value: str, num_shards: int) -> int:
    digest = sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(num_shards)


def _dicom_value(value: Any) -> Any:
    """Unwrap pydicom-json style tag dicts to their actual Value payload."""
    if isinstance(value, dict) and "Value" in value:
        value = value.get("Value")
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _as_int(value: Any) -> int | None:
    value = _dicom_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _parse_series_token(value: str) -> tuple[int | None, str, str]:
    match = re.match(r"\s*(\d+)\s*([A-Za-z]*)\s*$", str(value))
    if not match:
        return None, str(value), ""
    number = int(match.group(1))
    suffix = match.group(2).upper()
    return number, f"{number}{suffix}", suffix


def _parse_float_sequence(value: Any) -> tuple[float, ...] | None:
    value = _dicom_value(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
        return tuple(vals)
    text = str(value).strip()
    if not text:
        return None
    text = text.strip("[]()")
    parts = [p for p in re.split(r"[\\, ]+", text) if p]
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sentence_for_offset(text: str, start: int, end: int) -> str:
    left_candidates = [text.rfind(token, 0, start) for token in ("\n\n", "\n", ".", "!", "?")]
    left = max(left_candidates)
    left = 0 if left < 0 else left + 1
    right_candidates = [idx for token in ("\n\n", "\n", ".", "!", "?") if (idx := text.find(token, end)) >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return _normalize_text(text[left:right])


def _section_for_offset(text: str, offset: int) -> str:
    prefix_start = max(0, offset - 4000)
    prefix = text[prefix_start:offset]
    matches = list(SECTION_RE.finditer(prefix))
    if not matches:
        return ""
    return _normalize_text(matches[-1].group(1)).strip(": ")


def _organ_hint(section: str, sentence: str) -> str:
    haystacks = [section.lower(), sentence.lower()]
    for organ, terms in ORGAN_TERMS:
        for term in terms:
            if any(re.search(rf"\b{re.escape(term)}\b", h) for h in haystacks):
                return organ
    return "unknown"


def _parse_image_numbers(value: str, max_range: int = 25) -> list[int]:
    text = value.lower().replace("&", " and ")
    text = re.sub(r"\band\b", ",", text)
    chunks = [c.strip() for c in text.split(",") if c.strip()]
    numbers: list[int] = []
    for chunk in chunks:
        range_match = re.match(r"^(\d+)\s*(?:-|to)\s*(\d+)$", chunk)
        if range_match:
            start = int(range_match.group(1))
            stop = int(range_match.group(2))
            lo, hi = sorted((start, stop))
            if hi - lo <= max_range:
                numbers.extend(range(lo, hi + 1))
            else:
                numbers.extend([start, stop])
            continue
        numbers.extend(int(m.group(0)) for m in re.finditer(r"\d+", chunk))
    deduped: list[int] = []
    seen: set[int] = set()
    for number in numbers:
        if number not in seen:
            seen.add(number)
            deduped.append(number)
    return deduped


def find_report_files(study_dir: str | Path) -> list[Path]:
    """Return report text files in a study directory."""
    root = Path(study_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    reports = [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt" and "report" in p.name.lower()
    ]
    return sorted(reports)


def parse_report_references(report_path: str | Path) -> list[ReportReference]:
    """Extract series/image references from a plain-text report."""
    p = Path(report_path).expanduser()
    text = p.read_text(encoding="utf-8", errors="replace")
    refs: list[ReportReference] = []
    for match in SERIES_IMAGE_RE.finditer(text):
        series_number, series_text, suffix = _parse_series_token(match.group("series"))
        if series_number is None:
            continue
        image_raw = match.group("images")
        image_numbers = _parse_image_numbers(image_raw)
        if not image_numbers:
            continue
        sentence = _sentence_for_offset(text, match.start(), match.end())
        section = _section_for_offset(text, match.start())
        organ = _organ_hint(section, sentence)
        for image_number in image_numbers:
            refs.append(
                ReportReference(
                    report_path=str(p),
                    section=section,
                    organ_hint=organ,
                    sentence=sentence,
                    reference_text=_normalize_text(match.group(0)),
                    series_number=series_number,
                    series_suffix=suffix,
                    image_number=int(image_number),
                    image_number_raw_text=image_raw,
                )
            )
    return refs


@lru_cache(maxsize=16384)
def _read_json_cached(path_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path_text).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_cached(str(path))


def _candidate_metadata_json(series_dir: Path) -> Path | None:
    exact = series_dir / f"{series_dir.name}.json"
    if exact.exists():
        return exact
    jsons = [p for p in series_dir.glob("*.json") if not p.name.endswith("_series_tree.json")]
    return sorted(jsons)[0] if jsons else None


def _candidate_tree_json(series_dir: Path) -> Path | None:
    exact = series_dir / f"{series_dir.name}_series_tree.json"
    if exact.exists():
        return exact
    trees = sorted(series_dir.glob("*_series_tree.json"))
    return trees[0] if trees else None


def _candidate_nifti(series_dir: Path) -> Path | None:
    nii_gz = sorted(series_dir.glob("*.nii.gz"))
    if nii_gz:
        return nii_gz[0]
    nii = sorted(series_dir.glob("*.nii"))
    return nii[0] if nii else None


def discover_study_series(
    study_dir: str | Path,
    allowed_series_paths: set[str] | None = None,
) -> list[SeriesInfo]:
    """Discover per-series metadata under one PMBB study directory."""
    root = Path(study_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    allowed = {str(Path(p).expanduser()) for p in allowed_series_paths} if allowed_series_paths is not None else None
    out: list[SeriesInfo] = []
    for series_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        series_path = str(series_dir)
        if allowed is not None and series_path not in allowed:
            continue
        meta_path = _candidate_metadata_json(series_dir)
        tree_path = _candidate_tree_json(series_dir)
        nifti_path = _candidate_nifti(series_dir)
        if meta_path is None:
            metadata: dict[str, Any] = {}
        else:
            metadata = _read_json(meta_path)
        number_value = _dicom_value(metadata.get("SeriesNumber", metadata.get("series_number", "")))
        series_number, series_text, suffix = _parse_series_token(str(number_value))
        if series_number is None:
            series_number = _as_int(number_value)
            series_text = str(series_number) if series_number is not None else str(number_value)
        out.append(
            SeriesInfo(
                series_path=series_path,
                metadata_path=str(meta_path) if meta_path else "",
                tree_path=str(tree_path) if tree_path else "",
                nifti_path=str(nifti_path) if nifti_path else "",
                series_number=series_number,
                series_number_text=series_text,
                series_suffix=suffix,
                series_description=str(
                    _dicom_value(metadata.get("SeriesDescription", metadata.get("series_description", series_dir.name)))
                ),
                modality=str(_dicom_value(metadata.get("Modality", metadata.get("modality", "")))).upper(),
            )
        )
    return out


def _match_series_infos(reference: ReportReference, series_infos: list[SeriesInfo]) -> tuple[list[SeriesInfo], str]:
    exact_text = [
        s
        for s in series_infos
        if s.series_number_text.upper() == f"{reference.series_number}{reference.series_suffix}".upper()
    ]
    if exact_text:
        return exact_text, "exact_series_text" if len(exact_text) == 1 else "ambiguous_exact_series_text"

    numeric = [s for s in series_infos if s.series_number == reference.series_number]
    if not numeric:
        return [], "unmatched_series_number"
    if reference.series_suffix:
        return numeric, "suffix_ignored_numeric_match" if len(numeric) == 1 else "ambiguous_suffix_ignored_numeric_match"
    return numeric, "exact_series_number" if len(numeric) == 1 else "ambiguous_exact_series_number"


def _find_instance_list(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        value = obj.get("InstanceList")
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        for child in obj.values():
            found = _find_instance_list(child)
            if found:
                return found
    elif isinstance(obj, list):
        for child in obj:
            found = _find_instance_list(child)
            if found:
                return found
    return []


def _instance_number(instance: dict[str, Any]) -> int | None:
    return _as_int(instance.get("InstanceNumber", instance.get("instance_number")))


@lru_cache(maxsize=8192)
def _canonical_affine_and_shape(nifti_path: str) -> tuple[np.ndarray, tuple[int, int, int]]:
    img = nib.as_closest_canonical(nib.load(nifti_path))
    shape = tuple(int(v) for v in img.shape[:3])
    return np.asarray(img.affine, dtype=np.float64), shape


def _instance_voxel(
    instance: dict[str, Any],
    inv_affine: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    ipp = _parse_float_sequence(instance.get("ImagePositionPatient"))
    if ipp is None or len(ipp) < 3:
        return None
    ras = np.asarray([-float(ipp[0]), -float(ipp[1]), float(ipp[2]), 1.0], dtype=np.float64)
    vox_f = inv_affine @ ras
    vox = np.rint(vox_f[:3]).astype(np.int64)
    clipped = np.clip(vox, np.zeros(3, dtype=np.int64), np.asarray(shape, dtype=np.int64) - 1)
    return tuple(int(v) for v in clipped.tolist())


def _slice_axis_from_instances(
    instances: list[dict[str, Any]],
    inv_affine: np.ndarray,
    shape: tuple[int, int, int],
) -> int | None:
    coords = [_instance_voxel(instance, inv_affine, shape) for instance in instances]
    good = np.asarray([c for c in coords if c is not None], dtype=np.float64)
    if good.ndim != 2 or good.shape[0] < 2:
        return None
    return int(np.argmax(np.ptp(good, axis=0)))


def _sort_instances_for_ordinal(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with_instance = [(idx, _instance_number(inst), inst) for idx, inst in enumerate(instances)]
    if any(number is not None for _, number, _ in with_instance):
        return [inst for _, _, inst in sorted(with_instance, key=lambda x: (x[1] is None, x[1] or 0, x[0]))]
    with_location = []
    for idx, inst in enumerate(instances):
        location = None
        try:
            location = float(_dicom_value(inst.get("SliceLocation")))
        except Exception:
            pass
        with_location.append((idx, location, inst))
    if any(location is not None for _, location, _ in with_location):
        return [inst for _, _, inst in sorted(with_location, key=lambda x: (x[1] is None, x[1] or 0.0, x[0]))]
    return instances


@lru_cache(maxsize=8192)
def _series_mapping_context(
    tree_path: str,
    nifti_path: str,
) -> tuple[
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    list[dict[str, Any]],
    np.ndarray,
    tuple[int, int, int],
    int | None,
]:
    tree = _read_json(Path(tree_path))
    instances = _find_instance_list(tree)
    affine, shape = _canonical_affine_and_shape(nifti_path)
    inv_affine = np.linalg.inv(affine)
    slice_axis = _slice_axis_from_instances(instances, inv_affine, shape) if instances else None
    by_number: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for inst in instances:
        number = _instance_number(inst)
        if number is not None:
            by_number[number].append(inst)
    sorted_instances = _sort_instances_for_ordinal(instances)
    return instances, dict(by_number), sorted_instances, inv_affine, shape, slice_axis


def map_reported_image_to_slice(series: SeriesInfo, image_number: int) -> SliceMapping:
    """Map a reported image number to a canonical RAS voxel/slice coordinate."""
    empty_shape: tuple[int | None, int | None, int | None] = (None, None, None)
    empty_voxel: tuple[int | None, int | None, int | None] = (None, None, None)
    if not series.tree_path:
        return SliceMapping(
            "unmapped",
            "missing_series_tree_json",
            None,
            None,
            "",
            empty_voxel,
            None,
            empty_shape,
        )
    if not series.nifti_path:
        return SliceMapping("unmapped", "missing_nifti", None, None, "", empty_voxel, None, empty_shape)

    try:
        instances, by_number, sorted_instances, inv_affine, shape, slice_axis = _series_mapping_context(
            series.tree_path,
            series.nifti_path,
        )
    except Exception as exc:
        return SliceMapping("unmapped", f"series_mapping_context_error:{type(exc).__name__}", None, None, "", empty_voxel, None, empty_shape)

    if not instances:
        return SliceMapping("unmapped", "missing_instance_list", None, None, "", empty_voxel, None, empty_shape)
    axis_names = ("x", "y", "z")
    slice_axis_name = axis_names[slice_axis] if slice_axis is not None else ""

    selected: dict[str, Any] | None = None
    exact = by_number.get(int(image_number), [])
    if exact:
        selected = exact[0]
        base_confidence = "high_exact_instance" if len(exact) == 1 else "medium_ambiguous_exact_instance"
        reason = "matched_dicom_instance_number"
    else:
        if 1 <= int(image_number) <= len(sorted_instances):
            selected = sorted_instances[int(image_number) - 1]
            base_confidence = "medium_ordinal"
            reason = "reported_image_number_used_as_1_based_ordinal"
        else:
            return SliceMapping(
                "unmapped",
                "image_number_outside_instance_range",
                None,
                slice_axis,
                slice_axis_name,
                empty_voxel,
                None,
                tuple(int(v) for v in shape),
            )

    voxel = _instance_voxel(selected, inv_affine, shape)
    instance_number = _instance_number(selected)
    if voxel is None:
        return SliceMapping(
            base_confidence.replace("high_", "medium_").replace("medium_", "low_"),
            f"{reason};missing_image_position_patient",
            instance_number,
            slice_axis,
            slice_axis_name,
            empty_voxel,
            None,
            tuple(int(v) for v in shape),
        )
    canonical_slice = int(voxel[slice_axis]) if slice_axis is not None else None
    return SliceMapping(
        base_confidence,
        reason,
        instance_number,
        slice_axis,
        slice_axis_name,
        tuple(int(v) for v in voxel),
        canonical_slice,
        tuple(int(v) for v in shape),
    )


def _empty_row(study_dir: str, reference: ReportReference, series_match_confidence: str, series_match_count: int) -> dict[str, Any]:
    row = {col: None for col in REPORT_REF_COLUMNS}
    row.update(
        {
            "study_dir": study_dir,
            "report_path": reference.report_path,
            "report_section": reference.section,
            "organ_hint": reference.organ_hint,
            "sentence": reference.sentence,
            "reference_text": reference.reference_text,
            "series_number_reported": reference.series_number,
            "series_suffix_reported": reference.series_suffix,
            "image_number_reported": reference.image_number,
            "image_number_raw_text": reference.image_number_raw_text,
            "series_match_confidence": series_match_confidence,
            "series_match_count": series_match_count,
            "series_match_rank": None,
            "slice_mapping_confidence": "unmapped",
            "confidence_reason": series_match_confidence,
        }
    )
    return row


def _row_for_match(
    study_dir: str,
    study_catalog_rows: list[dict[str, Any]],
    catalog_row_by_series_path: dict[str, dict[str, Any]],
    reference: ReportReference,
    series: SeriesInfo,
    series_match_confidence: str,
    series_match_count: int,
    series_match_rank: int,
) -> dict[str, Any]:
    mapping = map_reported_image_to_slice(series, reference.image_number)
    matched_catalog = catalog_row_by_series_path.get(series.series_path, {})
    fallback_catalog = study_catalog_rows[0] if study_catalog_rows else {}
    catalog = matched_catalog or fallback_catalog
    voxel = mapping.canonical_voxel
    shape = mapping.canonical_shape
    row = {col: None for col in REPORT_REF_COLUMNS}
    row.update(
        {
            "pmbb_id": str(catalog.get("pmbb_id", "")),
            "study_dir": study_dir,
            "report_path": reference.report_path,
            "report_section": reference.section,
            "organ_hint": reference.organ_hint,
            "sentence": reference.sentence,
            "reference_text": reference.reference_text,
            "series_number_reported": int(reference.series_number),
            "series_suffix_reported": reference.series_suffix,
            "image_number_reported": int(reference.image_number),
            "image_number_raw_text": reference.image_number_raw_text,
            "series_match_confidence": series_match_confidence,
            "series_match_count": int(series_match_count),
            "series_match_rank": int(series_match_rank),
            "series_number_matched": series.series_number_text,
            "series_description": series.series_description or str(catalog.get("series_description", "")),
            "modality": (series.modality or str(catalog.get("modality", ""))).upper(),
            "body_part": str(catalog.get("body_part", "")),
            "series_path": series.series_path,
            "nifti_path": series.nifti_path,
            "tree_path": series.tree_path,
            "slice_mapping_confidence": mapping.confidence,
            "confidence_reason": mapping.reason,
            "dicom_instance_number": None if mapping.dicom_instance_number is None else int(mapping.dicom_instance_number),
            "slice_axis": None if mapping.slice_axis is None else int(mapping.slice_axis),
            "slice_axis_name": mapping.slice_axis_name,
            "canonical_voxel_x": None if voxel[0] is None else int(voxel[0]),
            "canonical_voxel_y": None if voxel[1] is None else int(voxel[1]),
            "canonical_voxel_z": None if voxel[2] is None else int(voxel[2]),
            "canonical_slice_index": None if mapping.canonical_slice_index is None else int(mapping.canonical_slice_index),
            "canonical_shape_x": None if shape[0] is None else int(shape[0]),
            "canonical_shape_y": None if shape[1] is None else int(shape[1]),
            "canonical_shape_z": None if shape[2] is None else int(shape[2]),
        }
    )
    return row


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))


def build_report_reference_manifest(
    catalog_path: str,
    modalities: tuple[str, ...] = ("CT", "MR"),
    max_rows: int = 0,
    max_reports_per_study: int = 0,
    matched_series_scope: str = "catalog",
    include_unmapped: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
    progress_every: int = 250,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build a weak report-reference manifest from a catalog or manifest CSV."""
    if matched_series_scope not in {"catalog", "study"}:
        raise ValueError("matched_series_scope must be 'catalog' or 'study'")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    df = filter_nonempty_series_path(filter_modalities(load_catalog(catalog_path), modalities))
    if max_rows > 0:
        df = df.head(int(max_rows))

    catalog_rows = df.to_dicts()
    study_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    catalog_row_by_series_path: dict[str, dict[str, Any]] = {}
    for row in catalog_rows:
        series_path = str(Path(str(row.get("series_path", ""))).expanduser())
        if not series_path:
            continue
        study_dir = str(Path(series_path).parent)
        study_rows[study_dir].append(row)
        catalog_row_by_series_path[series_path] = row

    selected_studies = [
        study_dir for study_dir in sorted(study_rows) if num_shards == 1 or _stable_shard(study_dir, num_shards) == shard_index
    ]
    counters: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    series_match_counts: Counter[str] = Counter()
    organ_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for idx, study_dir in enumerate(selected_studies, start=1):
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0):
            print(f"[report-refs] study {idx}/{len(selected_studies)} rows={len(rows)}", flush=True)

        counters["studies_considered"] += 1
        report_files = find_report_files(study_dir)
        if max_reports_per_study > 0:
            report_files = report_files[: int(max_reports_per_study)]
        if not report_files:
            counters["studies_without_report_txt"] += 1
            continue

        allowed_paths = None
        if matched_series_scope == "catalog":
            allowed_paths = {str(Path(str(r.get("series_path", ""))).expanduser()) for r in study_rows[study_dir]}
        series_infos = discover_study_series(study_dir, allowed_paths)
        if not series_infos:
            counters["studies_without_series_metadata"] += 1

        references: list[ReportReference] = []
        for report_path in report_files:
            counters["reports_considered"] += 1
            refs = parse_report_references(report_path)
            if not refs:
                counters["reports_without_references"] += 1
            references.extend(refs)

        if not references:
            counters["studies_without_references"] += 1
            continue

        for reference in references:
            counters["references_extracted"] += 1
            organ_counts[reference.organ_hint] += 1
            matches, match_confidence = _match_series_infos(reference, series_infos)
            series_match_counts[match_confidence] += 1
            if not matches:
                counters["references_without_series_match"] += 1
                if include_unmapped:
                    rows.append(_empty_row(study_dir, reference, match_confidence, 0))
                    confidence_counts["unmapped"] += 1
                continue
            for rank, series in enumerate(matches):
                out_row = _row_for_match(
                    study_dir=study_dir,
                    study_catalog_rows=study_rows[study_dir],
                    catalog_row_by_series_path=catalog_row_by_series_path,
                    reference=reference,
                    series=series,
                    series_match_confidence=match_confidence,
                    series_match_count=len(matches),
                    series_match_rank=rank,
                )
                rows.append(out_row)
                confidence_counts[str(out_row["slice_mapping_confidence"])] += 1
                modality = str(out_row.get("modality", "")).upper() or "unknown"
                modality_counts[modality] += 1

    out_df = pl.DataFrame(rows, schema=REPORT_REF_SCHEMA, strict=False)
    summary = {
        "catalog_path": str(Path(catalog_path).expanduser()),
        "modalities": list(modalities),
        "max_rows": int(max_rows),
        "max_reports_per_study": int(max_reports_per_study),
        "matched_series_scope": matched_series_scope,
        "include_unmapped": bool(include_unmapped),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "input_catalog_rows": int(len(catalog_rows)),
        "unique_studies_total": int(len(study_rows)),
        "unique_studies_selected": int(len(selected_studies)),
        "rows_written": int(len(rows)),
        "counters": _counter_dict(counters),
        "slice_mapping_confidence_counts": _counter_dict(confidence_counts),
        "series_match_confidence_counts": _counter_dict(series_match_counts),
        "organ_hint_counts": _counter_dict(organ_counts),
        "modality_counts": _counter_dict(modality_counts),
    }
    return out_df, summary
