from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import prism_ssl.eval.checkpoint_probe as checkpoint_probe
from prism_ssl.config import load_run_config_from_flat
from prism_ssl.eval.checkpoint_probe import (
    _TOTALSEG_CACHE,
    dominant_totalseg_label_for_view,
    download_wandb_run_checkpoint,
    list_wandb_run_model_artifacts,
    masked_l1_per_view,
    parse_wandb_run_ref,
)
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


def test_parse_wandb_run_ref_accepts_url_and_triplet() -> None:
    assert parse_wandb_run_ref("https://wandb.ai/vineeth-gangaram-penn/nvreason-prism-ssl/runs/v9ng7jz6?nw=x") == (
        "vineeth-gangaram-penn",
        "nvreason-prism-ssl",
        "v9ng7jz6",
    )
    assert parse_wandb_run_ref("vineeth-gangaram-penn/nvreason-prism-ssl/v9ng7jz6") == (
        "vineeth-gangaram-penn",
        "nvreason-prism-ssl",
        "v9ng7jz6",
    )


def test_wandb_artifact_listing_and_download_use_session_tmp(monkeypatch, tmp_path: Path) -> None:
    class _FakeLoggedArtifact:
        def __init__(self, name: str, artifact_type: str, version: str, aliases: list[str]) -> None:
            self.name = name
            self.type = artifact_type
            self.version = version
            self.aliases = aliases

    class _FakeRun:
        def logged_artifacts(self) -> list[_FakeLoggedArtifact]:
            return [
                _FakeLoggedArtifact("prism-ssl-ckpt:v77", "model", "v77", ["step-259921"]),
                _FakeLoggedArtifact("ignore-me:v1", "dataset", "v1", []),
                _FakeLoggedArtifact("prism-ssl-ckpt:v80", "model", "v80", ["step-392994"]),
                _FakeLoggedArtifact("prism-ssl-ckpt:v74", "model", "v74", ["step-128309"]),
            ]

    class _FakeArtifact:
        def __init__(self, artifact_ref: str) -> None:
            self.artifact_ref = artifact_ref

        def download(self, root: str) -> str:
            artifact_dir = Path(root) / "artifact_payload"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "step_392994.ckpt").touch()
            return str(artifact_dir)

    class _FakeApi:
        def __init__(self) -> None:
            self.run_calls: list[str] = []
            self.artifact_calls: list[tuple[str, str]] = []

        def run(self, path: str) -> _FakeRun:
            self.run_calls.append(path)
            return _FakeRun()

        def artifact(self, ref: str, type: str) -> _FakeArtifact:
            self.artifact_calls.append((ref, type))
            return _FakeArtifact(ref)

    class _FakeWandb:
        def __init__(self, api: _FakeApi) -> None:
            self._api = api

        def Api(self, timeout: int) -> _FakeApi:  # noqa: N802
            assert timeout == 60
            return self._api

    api = _FakeApi()
    session_tmp = tmp_path / "session_tmp"
    checkpoint_probe._WANDB_ARTIFACT_LIST_CACHE.clear()
    checkpoint_probe._WANDB_CHECKPOINT_CACHE.clear()
    monkeypatch.setattr(checkpoint_probe, "_SESSION_TMP_DIR", session_tmp)
    monkeypatch.setattr(checkpoint_probe, "_load_wandb_module", lambda: _FakeWandb(api))

    artifacts = list_wandb_run_model_artifacts("vineeth-gangaram-penn/nvreason-prism-ssl/v9ng7jz6")
    cached_artifacts = list_wandb_run_model_artifacts("vineeth-gangaram-penn/nvreason-prism-ssl/v9ng7jz6")
    ckpt_path = download_wandb_run_checkpoint(
        "vineeth-gangaram-penn/nvreason-prism-ssl/v9ng7jz6",
        artifacts[0]["artifact_ref"],
    )
    cached_ckpt_path = download_wandb_run_checkpoint(
        "vineeth-gangaram-penn/nvreason-prism-ssl/v9ng7jz6",
        artifacts[0]["artifact_ref"],
    )

    assert [item["step"] for item in artifacts] == [392994, 259921, 128309]
    assert [item["artifact_ref"] for item in artifacts] == [
        "vineeth-gangaram-penn/nvreason-prism-ssl/prism-ssl-ckpt:v80",
        "vineeth-gangaram-penn/nvreason-prism-ssl/prism-ssl-ckpt:v77",
        "vineeth-gangaram-penn/nvreason-prism-ssl/prism-ssl-ckpt:v74",
    ]
    assert artifacts == cached_artifacts
    assert api.run_calls == ["vineeth-gangaram-penn/nvreason-prism-ssl/v9ng7jz6"]
    assert api.artifact_calls == [("vineeth-gangaram-penn/nvreason-prism-ssl/prism-ssl-ckpt:v80", "model")]
    assert ckpt_path == cached_ckpt_path
    assert ckpt_path.is_file()
    assert str(ckpt_path).startswith(str(session_tmp))


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
