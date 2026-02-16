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


class TransformerPatchPositionEncoder(nn.Module):
    """ViT-style token encoder over patch/position tokens."""

    def __init__(
        self,
        patch_dim: int = 256,
        d_model: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_proj = nn.Linear(3, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        ff_dim = int(float(d_model) * float(mlp_ratio))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, patches: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, n_patches = patches.shape[:2]
        x = patches.reshape(bsz, n_patches, -1)
        x = self.patch_proj(x) + self.pos_proj(positions)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x)
        x = self.encoder(x)
        return self.norm(x[:, 0, :])
