"""Loss bundle and scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_ssl.config.schema import LossConfig
from prism_ssl.model.heads import PrismModelOutput
from prism_ssl.model.schedules import supcon_weight


@dataclass
class LossBundle:
    total: torch.Tensor
    distance: torch.Tensor
    supcon: torch.Tensor
    mim: torch.Tensor
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


def compute_loss_bundle(
    outputs: PrismModelOutput,
    batch: dict,
    loss_cfg: LossConfig,
    step: int,
) -> tuple[LossBundle, dict[str, float]]:
    target_center_delta = batch["center_delta_mm"].float()

    # Binary relative-position classification per axis.
    distance_targets = (target_center_delta > 0).float()
    distance_logits = outputs.distance_logits
    distance_valid = (torch.abs(target_center_delta) >= 1.0).float()
    loss_distance_raw = F.binary_cross_entropy_with_logits(distance_logits, distance_targets, reduction="none")
    distance_valid_count = distance_valid.sum().clamp(min=1.0)
    loss_distance = (loss_distance_raw * distance_valid).sum() / distance_valid_count

    mim_losses: list[torch.Tensor] = []
    if outputs.mim_target_a.numel() > 0:
        mim_losses.append(F.l1_loss(outputs.mim_pred_a, outputs.mim_target_a))
    if outputs.mim_target_b.numel() > 0:
        mim_losses.append(F.l1_loss(outputs.mim_pred_b, outputs.mim_target_b))
    if mim_losses:
        loss_mim = torch.stack(mim_losses).mean()
    else:
        loss_mim = loss_distance.new_tensor(0.0)

    with torch.no_grad():
        distance_preds = (distance_logits > 0).float()
        distance_correct = (distance_preds == distance_targets).float()
        distance_acc = (distance_correct * distance_valid).sum() / distance_valid_count
        distance_acc_per_axis = (
            (distance_correct * distance_valid).sum(dim=0) / distance_valid.sum(dim=0).clamp(min=1.0)
        )

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
        + w_supcon * loss_supcon
        + loss_cfg.w_mim * loss_mim
    )

    diagnostics = {
        "distance_acc": float(distance_acc.item()),
        "distance_acc_R": float(distance_acc_per_axis[0].item()),
        "distance_acc_A": float(distance_acc_per_axis[1].item()),
        "distance_acc_S": float(distance_acc_per_axis[2].item()),
        "distance_valid_ratio": float(distance_valid.mean().item()),
        "target_center_delta_mean": float(torch.mean(target_center_delta).item()),
        "target_center_delta_std": float(torch.std(target_center_delta).item()),
        "mim_target_abs_mean": float(
            torch.mean(
                torch.abs(
                    torch.cat(
                        [t.reshape(-1) for t in (outputs.mim_target_a, outputs.mim_target_b) if t.numel() > 0],
                        dim=0,
                    )
                )
            ).item()
        )
        if outputs.mim_target_a.numel() > 0 or outputs.mim_target_b.numel() > 0
        else 0.0,
        "mim_pred_abs_mean": float(
            torch.mean(
                torch.abs(
                    torch.cat(
                        [t.detach().reshape(-1) for t in (outputs.mim_pred_a, outputs.mim_pred_b) if t.numel() > 0],
                        dim=0,
                    )
                )
            ).item()
        )
        if outputs.mim_pred_a.numel() > 0 or outputs.mim_pred_b.numel() > 0
        else 0.0,
        "mim_masked_patch_count": float(outputs.mim_target_a.shape[1] + outputs.mim_target_b.shape[1]),
        "supcon_weight": float(w_supcon),
    }

    return (
        LossBundle(
            total=total,
            distance=loss_distance,
            supcon=loss_supcon,
            mim=loss_mim,
            supcon_weight=float(w_supcon),
        ),
        diagnostics,
    )
