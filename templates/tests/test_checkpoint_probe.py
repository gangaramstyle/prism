from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from prism_ssl.config import load_run_config_from_flat
from prism_ssl.eval.checkpoint_probe import _TOTALSEG_CACHE, dominant_totalseg_label_for_view, masked_l1_per_view
from prism_ssl.model.heads import PrismModelOutput


def test_load_run_config_from_flat_restores_study4_fields() -> None:
    cfg = load_run_config_from_flat(
        {
            "data.sample_unit": "study4",
            "data.n_patches": 384,
            "data.modality_filter": ["CT"],
            "model.d_model": 192,
            "model.proj_dim": 48,
            "loss.w_mim_register_target": 0.7,
            "loss.w_mim_cross_target": 0.4,
        }
    )

    assert cfg.data.sample_unit == "study4"
    assert cfg.data.n_patches == 384
    assert cfg.data.modality_filter == ("CT",)
    assert cfg.model.d_model == 192
    assert cfg.model.proj_dim == 48
    assert cfg.loss.w_mim_register_target == 0.7
    assert cfg.loss.w_mim_cross_target == 0.4


def test_dominant_totalseg_label_for_view_votes_and_falls_back(tmp_path: Path) -> None:
    series_dir = tmp_path / "subjects" / "patient1" / "study1" / "seriesA"
    series_dir.mkdir(parents=True)
    ts_dir = tmp_path / "processing" / "totalsegmentator" / "patient1" / "study1" / "seriesA"
    ts_dir.mkdir(parents=True)

    seg = np.zeros((8, 8, 8), dtype=np.int16)
    seg[1, 1, 1] = 5
    seg[1, 1, 2] = 5
    seg[1, 1, 3] = 9
    seg[2, 2, 2] = 7
    seg[3, 3, 3] = 2
    seg[3, 3, 4] = 2
    seg[3, 4, 3] = 4
    seg[4, 3, 3] = 4
    nib.save(
        nib.Nifti1Image(seg, affine=np.eye(4, dtype=np.float32)),
        ts_dir / "seriesA_e1_ts_total_ct.nii.gz",
    )

    _TOTALSEG_CACHE.clear()

    voted = dominant_totalseg_label_for_view(
        {
            "series_path": str(series_dir),
            "patch_centers_vox": np.array([[1, 1, 1], [1, 1, 2], [1, 1, 3]], dtype=np.int64),
            "prism_center_vox": np.array([2, 2, 2], dtype=np.int64),
        }
    )
    fallback = dominant_totalseg_label_for_view(
        {
            "series_path": str(series_dir),
            "patch_centers_vox": np.array([[0, 0, 0]], dtype=np.int64),
            "prism_center_vox": np.array([2, 2, 2], dtype=np.int64),
        }
    )
    tie_break = dominant_totalseg_label_for_view(
        {
            "series_path": str(series_dir),
            "patch_centers_vox": np.array([[3, 3, 3], [3, 3, 4], [3, 4, 3], [4, 3, 3]], dtype=np.int64),
            "prism_center_vox": np.array([0, 0, 0], dtype=np.int64),
        }
    )

    assert voted == 5
    assert fallback == 7
    assert tie_break == 2


def test_masked_l1_per_view_shapes_and_cross_mask() -> None:
    def _pred_tuple(start: float) -> tuple[torch.Tensor, ...]:
        return tuple(torch.full((2, 2, 1, 1, 1), fill_value=start + offset, dtype=torch.float32) for offset in range(4))

    targets = tuple(torch.zeros((2, 2, 1, 1, 1), dtype=torch.float32) for _ in range(4))
    outputs = PrismModelOutput(
        mim_self_preds=_pred_tuple(1.0),
        mim_self_targets=targets,
        mim_register_preds=_pred_tuple(5.0),
        mim_register_targets=targets,
        mim_cross_preds=_pred_tuple(9.0),
        mim_cross_targets=targets,
    )

    bundle = masked_l1_per_view(outputs, cross_valid=torch.tensor([True, False]))

    assert bundle["self"].shape == (2, 4)
    assert bundle["register"].shape == (2, 4)
    assert bundle["cross"].shape == (2, 4)
    assert torch.allclose(bundle["self"][0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(
        bundle["cross_valid_mask"],
        torch.tensor([[True, True, True, True], [False, False, False, False]]),
    )
