"""Convert a local NLST DICOM tree into NIfTI series plus a minimal catalog."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--catalog-path", required=True, type=Path)
    parser.add_argument("--modalities", default="CT", type=str)
    parser.add_argument("--min-dicom-files", default=2, type=int)
    parser.add_argument("--limit-series", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_optional_dependencies() -> tuple[Any, Any]:
    try:
        import pydicom
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "This script requires pydicom and SimpleITK. "
            "Run it with: uv run --with pydicom --with SimpleITK python scripts/convert_local_nlst_dicoms.py ..."
        ) from exc
    return pydicom, sitk


def _series_dirs(dicom_root: Path) -> list[Path]:
    return sorted(path for path in dicom_root.glob("*/*/*") if path.is_dir())


def _read_dicom_metadata(pydicom: Any, series_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    dicom_files = sorted(path for path in series_dir.iterdir() if path.is_file() and path.suffix.lower() == ".dcm")
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True, force=True)
    metadata = {
        "patient_id": str(getattr(ds, "PatientID", series_dir.parents[1].name)),
        "study_date": str(getattr(ds, "StudyDate", "")),
        "study_description": str(getattr(ds, "StudyDescription", "")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "modality": str(getattr(ds, "Modality", "")),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "kernel": str(getattr(ds, "ConvolutionKernel", "")),
        "slice_thickness": str(getattr(ds, "SliceThickness", "")),
    }
    return metadata, dicom_files


def _nifti_stats(nifti_path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    img = nib.load(str(nifti_path))
    shape = tuple(int(v) for v in img.shape[:3])
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return shape, spacing


def _convert_series(sitk: Any, series_dir: Path, output_nifti: Path, overwrite: bool) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if output_nifti.exists() and not overwrite:
        return _nifti_stats(output_nifti)

    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not file_names:
        raise ValueError(f"SimpleITK could not resolve a DICOM series in {series_dir}")
    reader.SetFileNames(file_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_nifti), useCompression=True)
    return _nifti_stats(output_nifti)


def main() -> None:
    args = parse_args()
    pydicom, sitk = _load_optional_dependencies()

    dicom_root = args.dicom_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    catalog_path = args.catalog_path.expanduser().resolve()
    modalities = {token.strip().upper() for token in str(args.modalities).split(",") if token.strip()}

    output_root.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    converted = 0
    skipped = 0

    for index, series_dir in enumerate(_series_dirs(dicom_root)):
        if int(args.limit_series) > 0 and index >= int(args.limit_series):
            break

        try:
            metadata, dicom_files = _read_dicom_metadata(pydicom, series_dir)
            modality = str(metadata["modality"]).upper()
            if modalities and modality not in modalities:
                skipped += 1
                continue
            if len(dicom_files) < int(args.min_dicom_files):
                skipped += 1
                continue

            relative = series_dir.relative_to(dicom_root)
            output_series_dir = output_root / relative
            output_nifti = output_series_dir / "image.nii.gz"
            shape, spacing = _convert_series(sitk, series_dir, output_nifti, overwrite=bool(args.overwrite))

            sidecar = {
                **metadata,
                "source_dicom_dir": str(series_dir),
                "n_dicom_files": int(len(dicom_files)),
                "shape": list(shape),
                "spacing": list(spacing),
                "nifti_path": str(output_nifti),
            }
            (output_series_dir / "conversion.json").write_text(json.dumps(sidecar, indent=2))

            rows.append(
                {
                    "pmbb_id": str(metadata["patient_id"]),
                    "modality": modality,
                    "series_description": str(metadata["series_description"]) or series_dir.name,
                    "study_description": str(metadata["study_description"]),
                    "study_date": str(metadata["study_date"]),
                    "series_path": str(output_series_dir),
                    "nifti_path": str(output_nifti),
                    "source_dicom_dir": str(series_dir),
                    "study_instance_uid": str(metadata["study_instance_uid"]),
                    "series_instance_uid": str(metadata["series_instance_uid"]),
                    "manufacturer": str(metadata["manufacturer"]),
                    "kernel": str(metadata["kernel"]),
                    "slice_thickness": str(metadata["slice_thickness"]),
                    "n_dicom_files": int(len(dicom_files)),
                    "shape_z": int(shape[2]) if len(shape) > 2 else 1,
                    "shape_y": int(shape[1]) if len(shape) > 1 else 1,
                    "shape_x": int(shape[0]) if len(shape) > 0 else 1,
                    "spacing_x": float(spacing[0]) if len(spacing) > 0 else 0.0,
                    "spacing_y": float(spacing[1]) if len(spacing) > 1 else 0.0,
                    "spacing_z": float(spacing[2]) if len(spacing) > 2 else 0.0,
                }
            )
            converted += 1
            print(f"[ok] {series_dir} -> {output_nifti}", flush=True)
        except Exception as exc:
            failures.append({"series_dir": str(series_dir), "error": str(exc)})
            print(f"[fail] {series_dir}: {exc}", flush=True)

    fieldnames = [
        "pmbb_id",
        "modality",
        "series_description",
        "study_description",
        "study_date",
        "series_path",
        "nifti_path",
        "source_dicom_dir",
        "study_instance_uid",
        "series_instance_uid",
        "manufacturer",
        "kernel",
        "slice_thickness",
        "n_dicom_files",
        "shape_z",
        "shape_y",
        "shape_x",
        "spacing_x",
        "spacing_y",
        "spacing_z",
    ]
    with catalog_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failures_path = catalog_path.with_suffix(catalog_path.suffix + ".failures.json")
    failures_path.write_text(json.dumps(failures, indent=2))

    print(
        f"[done] converted={converted} skipped={skipped} failed={len(failures)} catalog={catalog_path} failures={failures_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
