from __future__ import annotations

from pathlib import Path

from prism_ssl.train.checkpoint import select_resume_checkpoint


def test_resume_selection_prefers_local(tmp_path: Path) -> None:
    local_ckpt = tmp_path / "last.ckpt"
    artifact_ckpt = tmp_path / "artifact.ckpt"
    local_ckpt.touch()
    artifact_ckpt.touch()

    chosen = select_resume_checkpoint(local_ckpt, artifact_ckpt)
    assert chosen == local_ckpt
