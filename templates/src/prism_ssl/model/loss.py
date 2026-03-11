"""Loss bundle and scheduling."""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from prism_ssl.config.schema import LossConfig
from prism_ssl.model.heads import PrismModelOutput
from prism_ssl.model.schedules import supcon_weight


@dataclass
class LossBundle:
    total: torch.Tensor
    pair_relation: torch.Tensor
    supcon_instance: torch.Tensor
    supcon_protocol: torch.Tensor
    patch_size: torch.Tensor
    mim: torch.Tensor
    w_supcon_instance: float
    w_supcon_protocol: float


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


def positives_per_anchor_mean(labels: torch.Tensor) -> float:
    same = labels[:, None] == labels[None, :]
    positives = torch.clamp(same.sum(dim=1) - 1, min=0)
    return float(positives.float().mean().item())


def _compute_pair_targets_and_metrics(
    *,
    pair_relation_logits: torch.Tensor,
    target_center_delta: torch.Tensor,
    target_window_delta: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    relation_targets = torch.cat(
        [
            (target_center_delta > 0).float(),
            (target_window_delta > 0).float(),
        ],
        dim=1,
    )
    relation_valid = torch.cat(
        [
            (torch.abs(target_center_delta) >= 1.0).float(),
            torch.ones_like(target_window_delta),
        ],
        dim=1,
    )
    relation_loss_raw = F.binary_cross_entropy_with_logits(pair_relation_logits, relation_targets, reduction="none")
    relation_valid_count = relation_valid.sum().clamp(min=1.0)
    loss_pair_relation = (relation_loss_raw * relation_valid).sum() / relation_valid_count

    relation_preds = (pair_relation_logits > 0).float()
    relation_correct = (relation_preds == relation_targets).float()
    center_valid = relation_valid[:, :3]
    center_correct = relation_correct[:, :3]
    center_valid_count = center_valid.sum().clamp(min=1.0)
    relation_acc = (center_correct * center_valid).sum() / center_valid_count
    relation_acc_per_axis = (
        (center_correct * center_valid).sum(dim=0) / center_valid.sum(dim=0).clamp(min=1.0)
    )
    relation_acc_shared = (relation_correct * relation_valid).sum() / relation_valid_count
    window_correct = relation_correct[:, 3:]
    window_valid = relation_valid[:, 3:]
    window_acc_per_axis = (
        (window_correct * window_valid).sum(dim=0) / window_valid.sum(dim=0).clamp(min=1.0)
    )
    metrics = {
        "pair_relation_acc": relation_acc,
        "pair_relation_acc_per_axis": relation_acc_per_axis,
        "pair_relation_acc_shared": relation_acc_shared,
        "window_acc_per_axis": window_acc_per_axis,
        "pair_relation_valid_ratio": relation_valid.mean(),
    }
    return loss_pair_relation, metrics


def _patch_size_targets(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    target_a = torch.log2(batch["source_patch_mm_a"].float().clamp(min=1e-6))
    target_b = torch.log2(batch["source_patch_mm_b"].float().clamp(min=1e-6))
    return target_a, target_b


def compute_loss_bundle(
    outputs: PrismModelOutput,
    batch: dict,
    loss_cfg: LossConfig,
    step: int,
) -> tuple[LossBundle, dict[str, float]]:
    if (
        outputs.pair_relation_logits is None
        or outputs.proj_instance_a is None
        or outputs.proj_instance_b is None
        or outputs.proj_protocol_a is None
        or outputs.proj_protocol_b is None
        or outputs.patch_size_pred_a is None
        or outputs.patch_size_pred_b is None
    ):
        raise ValueError("pair2 loss requires pair-relation, dual SupCon, and patch-size outputs")

    target_center_delta = batch["center_delta_mm"].float()
    target_window_delta = batch["window_delta"].float()
    loss_pair_relation, pair_metrics = _compute_pair_targets_and_metrics(
        pair_relation_logits=outputs.pair_relation_logits,
        target_center_delta=target_center_delta,
        target_window_delta=target_window_delta,
    )

    mim_losses: list[torch.Tensor] = []
    if outputs.mim_target_a is not None and outputs.mim_pred_a is not None and outputs.mim_target_a.numel() > 0:
        mim_losses.append(F.l1_loss(outputs.mim_pred_a, outputs.mim_target_a))
    if outputs.mim_target_b is not None and outputs.mim_pred_b is not None and outputs.mim_target_b.numel() > 0:
        mim_losses.append(F.l1_loss(outputs.mim_pred_b, outputs.mim_target_b))
    loss_mim_self = torch.stack(mim_losses).mean() if mim_losses else loss_pair_relation.new_tensor(0.0)

    instance_emb = torch.cat([outputs.proj_instance_a, outputs.proj_instance_b], dim=0)
    instance_labels = torch.cat([batch["series_instance_label"], batch["series_instance_label"]], dim=0)
    loss_supcon_instance = supervised_contrastive_loss(
        instance_emb,
        instance_labels,
        temp=loss_cfg.supcon_temperature,
    )

    protocol_emb = torch.cat([outputs.proj_protocol_a, outputs.proj_protocol_b], dim=0)
    protocol_labels = torch.cat([batch["series_protocol_label"], batch["series_protocol_label"]], dim=0)
    loss_supcon_protocol = supervised_contrastive_loss(
        protocol_emb,
        protocol_labels,
        temp=loss_cfg.supcon_temperature,
    )

    w_supcon_instance = supcon_weight(
        step=step,
        warmup=loss_cfg.supcon_warmup_steps,
        ramp=loss_cfg.supcon_ramp_steps,
        target=loss_cfg.w_supcon_instance_target,
    )
    w_supcon_protocol = supcon_weight(
        step=step,
        warmup=loss_cfg.supcon_warmup_steps,
        ramp=loss_cfg.supcon_ramp_steps,
        target=loss_cfg.w_supcon_protocol_target,
    )

    patch_size_target_a, patch_size_target_b = _patch_size_targets(batch)
    patch_size_losses = [
        F.smooth_l1_loss(outputs.patch_size_pred_a.float(), patch_size_target_a),
        F.smooth_l1_loss(outputs.patch_size_pred_b.float(), patch_size_target_b),
    ]
    loss_patch_size = torch.stack(patch_size_losses).mean()

    total = (
        loss_cfg.w_distance * loss_pair_relation
        + loss_cfg.w_mim * loss_mim_self
        + w_supcon_instance * loss_supcon_instance
        + w_supcon_protocol * loss_supcon_protocol
        + loss_cfg.w_patch_size * loss_patch_size
    )

    with torch.no_grad():
        relation_acc = pair_metrics["pair_relation_acc"]
        relation_acc_per_axis = pair_metrics["pair_relation_acc_per_axis"]
        relation_acc_shared = pair_metrics["pair_relation_acc_shared"]
        window_acc_per_axis = pair_metrics["window_acc_per_axis"]
        patch_size_pred_mm = torch.pow(2.0, torch.cat([outputs.patch_size_pred_a.float(), outputs.patch_size_pred_b.float()], dim=0))
        patch_size_target_mm = torch.cat([batch["source_patch_mm_a"].float(), batch["source_patch_mm_b"].float()], dim=0)

    diagnostics = {
        "pair_relation_acc": float(relation_acc.item()),
        "pair_relation_acc_x": float(relation_acc_per_axis[0].item()),
        "pair_relation_acc_y": float(relation_acc_per_axis[1].item()),
        "pair_relation_acc_z": float(relation_acc_per_axis[2].item()),
        "pair_relation_acc_shared": float(relation_acc_shared.item()),
        "window_acc_wc": float(window_acc_per_axis[0].item()),
        "window_acc_ww": float(window_acc_per_axis[1].item()),
        "pair_relation_valid_ratio": float(pair_metrics["pair_relation_valid_ratio"].item()),
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
        "w_supcon_instance": float(w_supcon_instance),
        "w_supcon_protocol": float(w_supcon_protocol),
        "patch_size_mae_mm": float(torch.mean(torch.abs(patch_size_pred_mm - patch_size_target_mm)).item()),
        "source_patch_mm_mean": float(patch_size_target_mm.mean().item()),
        "source_patch_mm_min": float(patch_size_target_mm.min().item()),
        "source_patch_mm_max": float(patch_size_target_mm.max().item()),
        "supcon_instance_positives_per_anchor_mean": positives_per_anchor_mean(instance_labels),
        "supcon_protocol_positives_per_anchor_mean": positives_per_anchor_mean(protocol_labels),
    }

    return (
        LossBundle(
            total=total,
            pair_relation=loss_pair_relation,
            supcon_instance=loss_supcon_instance,
            supcon_protocol=loss_supcon_protocol,
            patch_size=loss_patch_size,
            mim=loss_mim_self,
            w_supcon_instance=float(w_supcon_instance),
            w_supcon_protocol=float(w_supcon_protocol),
        ),
        diagnostics,
    )
