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
    mim_register_weight: float = 0.0
    mim_cross_weight: float = 0.0


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
    if outputs.distance_logits_x is not None:
        return _compute_study4_loss_bundle(outputs, batch, loss_cfg, step)
    return _compute_pair2_loss_bundle(outputs, batch, loss_cfg, step)


def _compute_pair_targets_and_metrics(
    *,
    distance_logits: torch.Tensor,
    target_center_delta: torch.Tensor,
    target_window_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    distance_targets = torch.cat(
        [
            (target_center_delta > 0).float(),
            (target_window_delta > 0).float(),
        ],
        dim=1,
    )
    distance_valid = torch.cat(
        [
            (torch.abs(target_center_delta) >= 1.0).float(),
            torch.ones_like(target_window_delta),
        ],
        dim=1,
    )
    loss_distance_raw = F.binary_cross_entropy_with_logits(distance_logits, distance_targets, reduction="none")
    distance_valid_count = distance_valid.sum().clamp(min=1.0)
    loss_distance = (loss_distance_raw * distance_valid).sum() / distance_valid_count

    distance_preds = (distance_logits > 0).float()
    distance_correct = (distance_preds == distance_targets).float()
    center_valid = distance_valid[:, :3]
    center_correct = distance_correct[:, :3]
    center_valid_count = center_valid.sum().clamp(min=1.0)
    distance_acc = (center_correct * center_valid).sum() / center_valid_count
    distance_acc_per_axis = (
        (center_correct * center_valid).sum(dim=0) / center_valid.sum(dim=0).clamp(min=1.0)
    )
    shared_acc = (distance_correct * distance_valid).sum() / distance_valid_count
    window_correct = distance_correct[:, 3:]
    window_valid = distance_valid[:, 3:]
    window_acc_per_axis = (
        (window_correct * window_valid).sum(dim=0) / window_valid.sum(dim=0).clamp(min=1.0)
    )
    metrics = {
        "distance_acc": distance_acc,
        "distance_acc_per_axis": distance_acc_per_axis,
        "distance_acc_shared": shared_acc,
        "window_acc_per_axis": window_acc_per_axis,
        "distance_valid_ratio": distance_valid.mean(),
    }
    return loss_distance, distance_valid, metrics


def _masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    if valid_mask is not None:
        if valid_mask.ndim != 1:
            raise ValueError(f"valid_mask must be rank-1, got shape={tuple(valid_mask.shape)}")
        if valid_mask.numel() != pred.shape[0]:
            raise ValueError(
                f"valid_mask length {valid_mask.numel()} does not match batch dimension {pred.shape[0]}"
            )
        if not torch.any(valid_mask):
            return pred.new_tensor(0.0)
        pred = pred[valid_mask]
        target = target[valid_mask]
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    return F.l1_loss(pred, target)


def _compute_pair2_loss_bundle(
    outputs: PrismModelOutput,
    batch: dict,
    loss_cfg: LossConfig,
    step: int,
) -> tuple[LossBundle, dict[str, float]]:
    target_center_delta = batch["center_delta_mm"].float()
    target_window_delta = batch["window_delta"].float()
    distance_logits = outputs.distance_logits
    if distance_logits is None or outputs.proj_a is None or outputs.proj_b is None:
        raise ValueError("pair2 loss requires pair2 model outputs")
    loss_distance, distance_valid, pair_metrics = _compute_pair_targets_and_metrics(
        distance_logits=distance_logits,
        target_center_delta=target_center_delta,
        target_window_delta=target_window_delta,
    )

    mim_losses: list[torch.Tensor] = []
    if outputs.mim_target_a is not None and outputs.mim_pred_a is not None and outputs.mim_target_a.numel() > 0:
        mim_losses.append(F.l1_loss(outputs.mim_pred_a, outputs.mim_target_a))
    if outputs.mim_target_b is not None and outputs.mim_pred_b is not None and outputs.mim_target_b.numel() > 0:
        mim_losses.append(F.l1_loss(outputs.mim_pred_b, outputs.mim_target_b))
    if mim_losses:
        loss_mim = torch.stack(mim_losses).mean()
    else:
        loss_mim = loss_distance.new_tensor(0.0)

    with torch.no_grad():
        distance_acc = pair_metrics["distance_acc"]
        distance_acc_per_axis = pair_metrics["distance_acc_per_axis"]
        shared_acc = pair_metrics["distance_acc_shared"]
        window_acc_per_axis = pair_metrics["window_acc_per_axis"]

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
        "distance_acc_shared": float(shared_acc.item()),
        "window_acc_wc": float(window_acc_per_axis[0].item()),
        "window_acc_ww": float(window_acc_per_axis[1].item()),
        "distance_valid_ratio": float(pair_metrics["distance_valid_ratio"].item()),
        "target_center_delta_mean": float(torch.mean(target_center_delta).item()),
        "target_center_delta_std": float(torch.std(target_center_delta).item()),
        "mim_target_abs_mean": float(
            torch.mean(
                torch.abs(
                    torch.cat(
                        [t.reshape(-1) for t in (outputs.mim_target_a, outputs.mim_target_b) if t is not None and t.numel() > 0],
                        dim=0,
                    )
                )
            ).item()
        )
        if (outputs.mim_target_a is not None and outputs.mim_target_a.numel() > 0)
        or (outputs.mim_target_b is not None and outputs.mim_target_b.numel() > 0)
        else 0.0,
        "mim_pred_abs_mean": float(
            torch.mean(
                torch.abs(
                    torch.cat(
                        [t.detach().reshape(-1) for t in (outputs.mim_pred_a, outputs.mim_pred_b) if t is not None and t.numel() > 0],
                        dim=0,
                    )
                )
            ).item()
        )
        if (outputs.mim_pred_a is not None and outputs.mim_pred_a.numel() > 0)
        or (outputs.mim_pred_b is not None and outputs.mim_pred_b.numel() > 0)
        else 0.0,
        "mim_masked_patch_count": float(
            (outputs.mim_target_a.shape[1] if outputs.mim_target_a is not None else 0)
            + (outputs.mim_target_b.shape[1] if outputs.mim_target_b is not None else 0)
        ),
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


def _compute_study4_loss_bundle(
    outputs: PrismModelOutput,
    batch: dict,
    loss_cfg: LossConfig,
    step: int,
) -> tuple[LossBundle, dict[str, float]]:
    if outputs.distance_logits_x is None or outputs.distance_logits_y is None or outputs.proj_views is None:
        raise ValueError("study4 loss requires study4 model outputs")

    loss_distance_x, _, metrics_x = _compute_pair_targets_and_metrics(
        distance_logits=outputs.distance_logits_x,
        target_center_delta=batch["center_delta_mm_x"].float(),
        target_window_delta=batch["window_delta_x"].float(),
    )
    loss_distance_y, _, metrics_y = _compute_pair_targets_and_metrics(
        distance_logits=outputs.distance_logits_y,
        target_center_delta=batch["center_delta_mm_y"].float(),
        target_window_delta=batch["window_delta_y"].float(),
    )
    loss_distance = 0.5 * (loss_distance_x + loss_distance_y)

    proj_views = outputs.proj_views.reshape(-1, outputs.proj_views.shape[-1])
    supcon_labels = batch["series_label_views"].reshape(-1)
    loss_supcon = supervised_contrastive_loss(
        proj_views,
        supcon_labels,
        temp=loss_cfg.supcon_temperature,
    )
    w_supcon = supcon_weight(
        step=step,
        warmup=loss_cfg.supcon_warmup_steps,
        ramp=loss_cfg.supcon_ramp_steps,
        target=loss_cfg.w_supcon_target,
    )
    w_register = supcon_weight(
        step=step,
        warmup=loss_cfg.mim_aux_warmup_steps,
        ramp=loss_cfg.mim_aux_ramp_steps,
        target=loss_cfg.w_mim_register_target,
    )
    w_cross = supcon_weight(
        step=step,
        warmup=loss_cfg.mim_aux_warmup_steps,
        ramp=loss_cfg.mim_aux_ramp_steps,
        target=loss_cfg.w_mim_cross_target,
    )

    self_losses = [
        _masked_l1_loss(pred, target)
        for pred, target in zip(outputs.mim_self_preds, outputs.mim_self_targets)
    ]
    register_losses = [
        _masked_l1_loss(pred, target)
        for pred, target in zip(outputs.mim_register_preds, outputs.mim_register_targets)
    ]
    cross_valid = batch["cross_valid"].bool()
    cross_losses = [
        _masked_l1_loss(pred, target, cross_valid)
        for pred, target in zip(outputs.mim_cross_preds, outputs.mim_cross_targets)
    ]
    loss_mim_self = torch.stack(self_losses).mean() if self_losses else loss_distance.new_tensor(0.0)
    loss_mim_register = torch.stack(register_losses).mean() if register_losses else loss_distance.new_tensor(0.0)
    loss_mim_cross = torch.stack(cross_losses).mean() if cross_losses else loss_distance.new_tensor(0.0)

    total = (
        loss_cfg.w_distance * loss_distance
        + w_supcon * loss_supcon
        + loss_cfg.w_mim * loss_mim_self
        + w_register * loss_mim_register
        + w_cross * loss_mim_cross
    )

    self_loss_det = torch.stack([loss.detach() for loss in self_losses]).mean() if self_losses else total.new_tensor(0.0)
    register_loss_det = (
        torch.stack([loss.detach() for loss in register_losses]).mean() if register_losses else total.new_tensor(0.0)
    )
    cross_loss_det = torch.stack([loss.detach() for loss in cross_losses]).mean() if cross_losses else total.new_tensor(0.0)

    diagnostics = {
        "distance_acc": float(0.5 * (metrics_x["distance_acc"] + metrics_y["distance_acc"]).item()),
        "distance_acc_R": float(
            0.5 * (metrics_x["distance_acc_per_axis"][0] + metrics_y["distance_acc_per_axis"][0]).item()
        ),
        "distance_acc_A": float(
            0.5 * (metrics_x["distance_acc_per_axis"][1] + metrics_y["distance_acc_per_axis"][1]).item()
        ),
        "distance_acc_S": float(
            0.5 * (metrics_x["distance_acc_per_axis"][2] + metrics_y["distance_acc_per_axis"][2]).item()
        ),
        "distance_acc_shared": float(0.5 * (metrics_x["distance_acc_shared"] + metrics_y["distance_acc_shared"]).item()),
        "window_acc_wc": float(
            0.5 * (metrics_x["window_acc_per_axis"][0] + metrics_y["window_acc_per_axis"][0]).item()
        ),
        "window_acc_ww": float(
            0.5 * (metrics_x["window_acc_per_axis"][1] + metrics_y["window_acc_per_axis"][1]).item()
        ),
        "distance_valid_ratio": float(0.5 * (metrics_x["distance_valid_ratio"] + metrics_y["distance_valid_ratio"]).item()),
        "distance_x": float(loss_distance_x.item()),
        "distance_y": float(loss_distance_y.item()),
        "distance_acc_x": float(metrics_x["distance_acc"].item()),
        "distance_acc_y": float(metrics_y["distance_acc"].item()),
        "distance_acc_shared_x": float(metrics_x["distance_acc_shared"].item()),
        "distance_acc_shared_y": float(metrics_y["distance_acc_shared"].item()),
        "window_acc_wc_x": float(metrics_x["window_acc_per_axis"][0].item()),
        "window_acc_wc_y": float(metrics_y["window_acc_per_axis"][0].item()),
        "window_acc_ww_x": float(metrics_x["window_acc_per_axis"][1].item()),
        "window_acc_ww_y": float(metrics_y["window_acc_per_axis"][1].item()),
        "target_center_delta_mean": float(
            torch.mean(torch.cat([batch["center_delta_mm_x"].float(), batch["center_delta_mm_y"].float()], dim=0)).item()
        ),
        "target_center_delta_std": float(
            torch.std(torch.cat([batch["center_delta_mm_x"].float(), batch["center_delta_mm_y"].float()], dim=0)).item()
        ),
        "mim_target_abs_mean": float(
            torch.mean(
                torch.abs(torch.cat([target.reshape(-1) for target in outputs.mim_self_targets if target.numel() > 0], dim=0))
            ).item()
        )
        if outputs.mim_self_targets
        else 0.0,
        "mim_pred_abs_mean": float(
            torch.mean(
                torch.abs(torch.cat([pred.detach().reshape(-1) for pred in outputs.mim_self_preds if pred.numel() > 0], dim=0))
            ).item()
        )
        if outputs.mim_self_preds
        else 0.0,
        "mim_masked_patch_count": float(sum(int(target.shape[1]) for target in outputs.mim_self_targets)),
        "supcon_weight": float(w_supcon),
        "loss_mim_self": float(self_loss_det.item()),
        "loss_mim_register": float(register_loss_det.item()),
        "loss_mim_cross": float(cross_loss_det.item()),
        "mim_register_gain": float((self_loss_det - register_loss_det).item()),
        "mim_cross_gain_vs_self": float((self_loss_det - cross_loss_det).item()),
        "mim_cross_gain_vs_register": float((register_loss_det - cross_loss_det).item()),
        "cross_valid_ratio": float(cross_valid.float().mean().item()),
        "mim_register_weight": float(w_register),
        "mim_cross_weight": float(w_cross),
    }

    return (
        LossBundle(
            total=total,
            distance=loss_distance,
            supcon=loss_supcon,
            mim=loss_mim_self,
            supcon_weight=float(w_supcon),
            mim_register_weight=float(w_register),
            mim_cross_weight=float(w_cross),
        ),
        diagnostics,
    )
