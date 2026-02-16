"""Loss bundle and scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from prism_ssl.config.schema import LossConfig
from prism_ssl.model.heads import PrismModelOutput
from prism_ssl.model.schedules import supcon_weight


@dataclass
class LossBundle:
    total: torch.Tensor
    distance: torch.Tensor
    rotation: torch.Tensor
    window: torch.Tensor
    supcon: torch.Tensor
    supcon_weight: float


def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temp: float) -> torch.Tensor:
    n = emb.shape[0]
    if n < 2:
        return emb.new_tensor(0.0)

    sim = (emb @ emb.T) / max(float(temp), 1e-6)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logits = sim.masked_fill(eye, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return emb.new_tensor(0.0)
    per_anchor = -(pos * logp).sum(dim=1) / pos_count.clamp(min=1)
    return per_anchor[valid].mean()


def _normalize_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_std = torch.std(target.detach())
    scale = torch.clamp(target_std, min=1e-6)
    return pred / scale, target / scale, target_std


def compute_loss_bundle(
    outputs: PrismModelOutput,
    batch: dict,
    loss_cfg: LossConfig,
    step: int,
    distance_loss_fn: nn.Module,
    rotation_loss_fn: nn.Module,
    window_loss_fn: nn.Module,
) -> tuple[LossBundle, dict[str, float]]:
    target_distance = batch["center_distance_mm"].float()
    target_rotation = batch["rotation_delta_deg"].float()
    target_window = batch["window_delta"].float()

    pred_distance = outputs.distance_mm
    pred_rotation = outputs.rotation_delta_deg
    pred_window = outputs.window_delta

    target_stats: dict[str, float] = {}
    pred_stats: dict[str, float] = {}

    if loss_cfg.normalize_targets:
        pred_distance_n, target_distance_n, d_std = _normalize_pair(pred_distance, target_distance)
        pred_rotation_n, target_rotation_n, r_std = _normalize_pair(pred_rotation, target_rotation)
        pred_window_n, target_window_n, w_std = _normalize_pair(pred_window, target_window)

        loss_distance = distance_loss_fn(pred_distance_n, target_distance_n)
        loss_rotation = rotation_loss_fn(pred_rotation_n, target_rotation_n)
        loss_window = window_loss_fn(pred_window_n, target_window_n)

        target_stats["distance_std"] = float(d_std.item())
        target_stats["rotation_std"] = float(r_std.item())
        target_stats["window_std"] = float(w_std.item())
    else:
        loss_distance = distance_loss_fn(pred_distance, target_distance)
        loss_rotation = rotation_loss_fn(pred_rotation, target_rotation)
        loss_window = window_loss_fn(pred_window, target_window)

    supcon_emb = torch.cat([outputs.proj_a, outputs.proj_b], dim=0)
    supcon_labels = torch.cat([batch["series_label"], batch["series_label"]], dim=0)
    loss_supcon = supervised_contrastive_loss(
        supcon_emb,
        supcon_labels,
        temp=loss_cfg.supcon_temperature,
    )

    w_supcon = supcon_weight(
        step=step,
        warmup=loss_cfg.supcon_warmup_steps,
        ramp=loss_cfg.supcon_ramp_steps,
        target=loss_cfg.w_supcon_target,
    )

    total = (
        loss_cfg.w_distance * loss_distance
        + loss_cfg.w_rotation * loss_rotation
        + loss_cfg.w_window * loss_window
        + w_supcon * loss_supcon
    )

    pred_stats["distance_std"] = float(torch.std(pred_distance.detach()).item())
    pred_stats["rotation_std"] = float(torch.std(pred_rotation.detach()).item())
    pred_stats["window_std"] = float(torch.std(pred_window.detach()).item())

    diagnostics = {
        "target_distance_mean": float(torch.mean(target_distance).item()),
        "target_rotation_abs_mean": float(torch.mean(torch.abs(target_rotation)).item()),
        "target_window_abs_mean": float(torch.mean(torch.abs(target_window)).item()),
        "target_distance_std": float(torch.std(target_distance).item()),
        "target_rotation_std": float(torch.std(target_rotation).item()),
        "target_window_std": float(torch.std(target_window).item()),
        "pred_distance_mean": float(torch.mean(pred_distance.detach()).item()),
        "pred_rotation_abs_mean": float(torch.mean(torch.abs(pred_rotation.detach())).item()),
        "pred_window_abs_mean": float(torch.mean(torch.abs(pred_window.detach())).item()),
        "pred_distance_std": pred_stats["distance_std"],
        "pred_rotation_std": pred_stats["rotation_std"],
        "pred_window_std": pred_stats["window_std"],
        "pred_to_target_std_ratio_distance": pred_stats["distance_std"] / max(float(torch.std(target_distance).item()), 1e-6),
        "pred_to_target_std_ratio_rotation": pred_stats["rotation_std"] / max(float(torch.std(target_rotation).item()), 1e-6),
        "pred_to_target_std_ratio_window": pred_stats["window_std"] / max(float(torch.std(target_window).item()), 1e-6),
        "supcon_weight": float(w_supcon),
    }

    return (
        LossBundle(
            total=total,
            distance=loss_distance,
            rotation=loss_rotation,
            window=loss_window,
            supcon=loss_supcon,
            supcon_weight=float(w_supcon),
        ),
        diagnostics,
    )
