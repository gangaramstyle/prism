"""Prism SSL prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_ssl.model.backbone import AbsoluteSinCosPositionEmbedding3D, TransformerPatchPositionEncoder


@dataclass
class PrismModelOutput:
    distance_logits: torch.Tensor  # [B, 3] — binary logits per axis (R, A, S)
    proj_a: torch.Tensor
    proj_b: torch.Tensor
    mim_pred_a: torch.Tensor
    mim_pred_b: torch.Tensor
    mim_target_a: torch.Tensor
    mim_target_b: torch.Tensor


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.mlp(self.mlp_norm(query))
        return query


class MaskedPatchDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        patch_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pos_embed = AbsoluteSinCosPositionEmbedding3D(dim)
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.to_patch = nn.Linear(dim, patch_dim)

    def forward(self, masked_positions: torch.Tensor, context: torch.Tensor, patch_shape: tuple[int, ...]) -> torch.Tensor:
        if masked_positions.shape[1] == 0:
            return context.new_empty((context.shape[0], 0, *patch_shape))
        query = self.pos_embed(masked_positions).to(dtype=context.dtype)
        for layer in self.layers:
            query = layer(query, context)
        patches = self.to_patch(self.norm(query))
        return patches.view(masked_positions.shape[0], masked_positions.shape[1], *patch_shape)


class PrismSSLModel(nn.Module):
    def __init__(
        self,
        patch_dim: int = 256,
        model_name: str = "vit_l",
        d_model: int = 256,
        proj_dim: int = 128,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        mim_mask_ratio: float = 0.25,
        mim_decoder_layers: int = 2,
    ):
        super().__init__()
        key = model_name.strip().lower()
        if key in {"vit_l", "vit_large", "vit-large"}:
            self.encoder = TransformerPatchPositionEncoder(
                patch_dim=patch_dim,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model.name='{model_name}'. Supported: vit_l")

        self.patch_shape = (16, 16, 1)
        self.patch_dim = int(patch_dim)
        self.mim_mask_ratio = float(max(0.0, min(0.95, mim_mask_ratio)))

        self.distance_head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 3))

        self.proj_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.mim_decoder = MaskedPatchDecoder(
            dim=d_model,
            patch_dim=patch_dim,
            num_heads=num_heads,
            num_layers=mim_decoder_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    @staticmethod
    def _gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        view = [x.shape[0], indices.shape[1]] + [1] * (x.ndim - 2)
        expand = list(x.shape)
        expand[1] = indices.shape[1]
        gather_idx = indices.view(*view).expand(*expand)
        return torch.gather(x, 1, gather_idx)

    def _split_visible_masked(
        self,
        patches: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_patches = patches.shape[:2]
        if n_patches <= 1 or self.mim_mask_ratio <= 0.0:
            empty_patches = patches.new_empty((bsz, 0, *patches.shape[2:]))
            empty_positions = positions.new_empty((bsz, 0, positions.shape[-1]))
            return patches, positions, empty_patches, empty_positions

        n_mask = int(round(n_patches * self.mim_mask_ratio))
        n_mask = max(1, min(n_patches - 1, n_mask))
        perm = torch.argsort(torch.rand(bsz, n_patches, device=patches.device), dim=1)
        masked_idx = perm[:, :n_mask]
        visible_idx = perm[:, n_mask:]
        visible_patches = self._gather_tokens(patches, visible_idx)
        visible_positions = self._gather_tokens(positions, visible_idx)
        masked_patches = self._gather_tokens(patches, masked_idx)
        masked_positions = self._gather_tokens(positions, masked_idx)
        return visible_patches, visible_positions, masked_patches, masked_positions

    def forward(
        self,
        patches_a: torch.Tensor,
        positions_a: torch.Tensor,
        patches_b: torch.Tensor,
        positions_b: torch.Tensor,
    ) -> PrismModelOutput:
        vis_patches_a, vis_pos_a, masked_patches_a, masked_pos_a = self._split_visible_masked(patches_a, positions_a)
        vis_patches_b, vis_pos_b, masked_patches_b, masked_pos_b = self._split_visible_masked(patches_b, positions_b)

        # Batch both views together to reduce encoder launch overhead.
        both_patches = torch.cat([vis_patches_a, vis_patches_b], dim=0)
        both_positions = torch.cat([vis_pos_a, vis_pos_b], dim=0)
        both_tokens = self.encoder(both_patches, both_positions)
        tokens_a, tokens_b = both_tokens.chunk(2, dim=0)

        distance_cls_a = tokens_a[:, 0, :]
        supcon_cls_a = tokens_a[:, 1, :]
        patch_tokens_a = tokens_a[:, 2:, :]
        distance_cls_b = tokens_b[:, 0, :]
        supcon_cls_b = tokens_b[:, 1, :]
        patch_tokens_b = tokens_b[:, 2:, :]

        fused = torch.cat([distance_cls_a, distance_cls_b], dim=1)
        distance_logits = self.distance_head(fused)

        proj_a = F.normalize(self.proj_head(supcon_cls_a), dim=1)
        proj_b = F.normalize(self.proj_head(supcon_cls_b), dim=1)

        mim_pred_a = self.mim_decoder(masked_pos_a, patch_tokens_a, self.patch_shape)
        mim_pred_b = self.mim_decoder(masked_pos_b, patch_tokens_b, self.patch_shape)

        return PrismModelOutput(
            distance_logits=distance_logits,
            proj_a=proj_a,
            proj_b=proj_b,
            mim_pred_a=mim_pred_a,
            mim_pred_b=mim_pred_b,
            mim_target_a=masked_patches_a,
            mim_target_b=masked_patches_b,
        )
