"""Prism SSL prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_ssl.model.backbone import PatchPositionEncoder


@dataclass
class PrismModelOutput:
    distance_mm: torch.Tensor
    rotation_delta_deg: torch.Tensor
    window_delta: torch.Tensor
    proj_a: torch.Tensor
    proj_b: torch.Tensor


class PrismSSLModel(nn.Module):
    def __init__(self, patch_dim: int = 256, d_model: int = 256, proj_dim: int = 128):
        super().__init__()
        self.encoder = PatchPositionEncoder(patch_dim=patch_dim, d_model=d_model)

        self.distance_head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 1))
        self.rotation_head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 3))
        self.window_head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 2))

        self.proj_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(
        self,
        patches_a: torch.Tensor,
        positions_a: torch.Tensor,
        patches_b: torch.Tensor,
        positions_b: torch.Tensor,
    ) -> PrismModelOutput:
        emb_a = self.encoder(patches_a, positions_a)
        emb_b = self.encoder(patches_b, positions_b)
        fused = torch.cat([emb_a, emb_b], dim=1)

        distance_mm = self.distance_head(fused).squeeze(-1)
        rotation_delta_deg = self.rotation_head(fused)
        window_delta = self.window_head(fused)

        proj_a = F.normalize(self.proj_head(emb_a), dim=1)
        proj_b = F.normalize(self.proj_head(emb_b), dim=1)

        return PrismModelOutput(
            distance_mm=distance_mm,
            rotation_delta_deg=rotation_delta_deg,
            window_delta=window_delta,
            proj_a=proj_a,
            proj_b=proj_b,
        )
