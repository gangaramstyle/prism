"""Data package exports."""

from prism_ssl.data.catalog import build_scan_id, load_catalog, sample_scan_candidates
from prism_ssl.data.collate import collate_prism_batch
from prism_ssl.data.preflight import (
    NiftiLoadError,
    NiftiResolutionError,
    ScanGeometry,
    SmallScanError,
    compute_robust_stats,
    infer_scan_geometry,
    load_nifti_scan,
    rotate_volume_about_center,
    rotated_relative_points_to_voxel,
    resolve_nifti_path,
)
from prism_ssl.data.sample_contract import build_dataset_item, compute_pair_targets, tensorize_sample_view
from prism_ssl.data.sharded_dataset import BrokenScanRateExceeded, ShardedScanDataset

__all__ = [
    "build_scan_id",
    "load_catalog",
    "sample_scan_candidates",
    "collate_prism_batch",
    "NiftiLoadError",
    "NiftiResolutionError",
    "ScanGeometry",
    "SmallScanError",
    "compute_robust_stats",
    "infer_scan_geometry",
    "load_nifti_scan",
    "rotate_volume_about_center",
    "rotated_relative_points_to_voxel",
    "resolve_nifti_path",
    "tensorize_sample_view",
    "compute_pair_targets",
    "build_dataset_item",
    "BrokenScanRateExceeded",
    "ShardedScanDataset",
]
