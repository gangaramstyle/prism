"""Model backbone for prism SSL."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchPositionEncoder(nn.Module):
    """Encode patch pixels + relative positions into pooled embeddings."""

    def __init__(self, patch_dim: int = 256, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_proj = nn.Linear(3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, patches: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, n_patches = patches.shape[:2]
        x = patches.reshape(bsz, n_patches, -1)
        x = self.patch_proj(x) + self.pos_proj(positions)
        x = self.norm(x + self.ffn(x))
        return x.mean(dim=1)
