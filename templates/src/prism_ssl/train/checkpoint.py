"""Checkpoint and artifact helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from prism_ssl.utils import ensure_dir, expand_path


def resolve_local_ckpt_dir(template: str, run_id: str) -> Path:
    expanded = str(expand_path(template.replace("<run_id>", run_id).replace("{run_id}", run_id)))
    return ensure_dir(expanded)


def save_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config_payload: dict[str, Any],
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_payload,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("step", 0))


def prune_local_checkpoints(ckpt_dir: Path, max_keep: int) -> int:
    if max_keep < 1:
        max_keep = 1
    if not ckpt_dir.exists():
        return 0

    checkpoints = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not checkpoints:
        return 0

    keep: set[Path] = set()
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        keep.add(last_ckpt)

    for ckpt in checkpoints:
        if len(keep) >= max_keep:
            break
        keep.add(ckpt)

    removed = 0
    for ckpt in checkpoints:
        if ckpt in keep:
            continue
        try:
            ckpt.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def download_latest_artifact_checkpoint(run: Any, artifact_name: str, tmp_run_dir: Path) -> Path | None:
    try:
        ref = f"{run.entity}/{run.project}/{artifact_name}:latest"
        artifact = run.use_artifact(ref, type="model")
        download_dir = tmp_run_dir / "artifact_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = Path(artifact.download(root=str(download_dir)))
        candidates = list(artifact_dir.rglob("*.ckpt"))
        return candidates[0] if candidates else None
    except Exception:
        return None


def select_resume_checkpoint(local_last_ckpt: Path, artifact_ckpt: Path | None) -> Path | None:
    if local_last_ckpt.exists():
        return local_last_ckpt
    return artifact_ckpt


def upload_artifact_checkpoint(
    run: Any,
    wandb_mod: Any,
    artifact_name: str,
    ckpt_path: Path,
    step: int,
    include_best_alias: bool,
) -> None:
    artifact = wandb_mod.Artifact(name=artifact_name, type="model")
    artifact.add_file(str(ckpt_path), name=f"step_{step}.ckpt")
    aliases = ["latest", f"step-{step}"]
    if include_best_alias:
        aliases.append("best")
    logged = run.log_artifact(artifact, aliases=aliases)
    if hasattr(logged, "wait"):
        logged.wait()


def ensure_tmp_env(tmp_run_dir: Path, wandb_mode: str) -> None:
    os.environ.setdefault("WANDB_DIR", str(tmp_run_dir / "wandb"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(tmp_run_dir / "wandb_cache"))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(tmp_run_dir / "wandb_artifacts"))
    os.environ.setdefault("TMPDIR", str(tmp_run_dir / "tmp"))
    os.environ["WANDB_MODE"] = wandb_mode
    for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_ARTIFACT_DIR", "TMPDIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
