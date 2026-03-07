"""Prism SSL prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_ssl.model.backbone import AbsoluteSinCosPositionEmbedding3D, TransformerPatchPositionEncoder


@dataclass
class PrismModelOutput:
    distance_logits: torch.Tensor | None = None  # [B, 5] — binary logits for (R, A, S, wc, ww)
    proj_a: torch.Tensor | None = None
    proj_b: torch.Tensor | None = None
    mim_pred_a: torch.Tensor | None = None
    mim_pred_b: torch.Tensor | None = None
    mim_target_a: torch.Tensor | None = None
    mim_target_b: torch.Tensor | None = None
    distance_logits_x: torch.Tensor | None = None
    distance_logits_y: torch.Tensor | None = None
    proj_views: torch.Tensor | None = None
    mim_self_preds: tuple[torch.Tensor, ...] = ()
    mim_self_targets: tuple[torch.Tensor, ...] = ()
    mim_register_preds: tuple[torch.Tensor, ...] = ()
    mim_register_targets: tuple[torch.Tensor, ...] = ()
    mim_cross_preds: tuple[torch.Tensor, ...] = ()
    mim_cross_targets: tuple[torch.Tensor, ...] = ()


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
        pos_min_wavelength_mm: float,
        pos_max_wavelength_mm: float,
    ) -> None:
        super().__init__()
        self.pos_embed = AbsoluteSinCosPositionEmbedding3D(
            dim,
            min_wavelength_mm=pos_min_wavelength_mm,
            max_wavelength_mm=pos_max_wavelength_mm,
        )
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


class CLSConditionedMemoryBuilder(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        supcon_cls: torch.Tensor,
        direction_cls: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens.shape[1] == 0:
            return patch_tokens
        n_tokens = patch_tokens.shape[1]
        sup = supcon_cls.unsqueeze(1).expand(-1, n_tokens, -1)
        direction = direction_cls.unsqueeze(1).expand(-1, n_tokens, -1)
        fused = torch.cat([sup, direction, patch_tokens], dim=-1)
        return self.fuse(fused)


class CLSConditionedPatchDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        patch_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
        pos_min_wavelength_mm: float,
        pos_max_wavelength_mm: float,
    ) -> None:
        super().__init__()
        self.pos_embed = AbsoluteSinCosPositionEmbedding3D(
            dim,
            min_wavelength_mm=pos_min_wavelength_mm,
            max_wavelength_mm=pos_max_wavelength_mm,
        )
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
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

    def forward(
        self,
        masked_positions: torch.Tensor,
        target_supcon_cls: torch.Tensor,
        target_direction_cls: torch.Tensor,
        context: torch.Tensor,
        patch_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if masked_positions.shape[1] == 0:
            return context.new_empty((context.shape[0], 0, *patch_shape))
        pos_embed = self.pos_embed(masked_positions).to(dtype=context.dtype)
        sup = target_supcon_cls.unsqueeze(1).expand(-1, masked_positions.shape[1], -1)
        direction = target_direction_cls.unsqueeze(1).expand(-1, masked_positions.shape[1], -1)
        query = self.query_mlp(torch.cat([sup, direction, pos_embed], dim=-1))
        for layer in self.layers:
            query = layer(query, context)
        patches = self.to_patch(self.norm(query))
        return patches.view(masked_positions.shape[0], masked_positions.shape[1], *patch_shape)


class PrismSSLModel(nn.Module):
    def __init__(
        self,
        patch_dim: int = 256,
        n_patches: int = 1024,
        model_name: str = "vit_l",
        d_model: int = 256,
        proj_dim: int = 128,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pos_min_wavelength_mm: float = 4.0,
        pos_max_wavelength_mm: float = 64.0,
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
                pos_min_wavelength_mm=pos_min_wavelength_mm,
                pos_max_wavelength_mm=pos_max_wavelength_mm,
            )
        else:
            raise ValueError(f"Unknown model.name='{model_name}'. Supported: vit_l")

        self.patch_shape = (16, 16, 1)
        self.patch_dim = int(patch_dim)
        self.n_patches = int(n_patches)
        self.mim_mask_ratio = float(max(0.0, min(0.95, mim_mask_ratio)))

        self.distance_head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 5))

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
            pos_min_wavelength_mm=pos_min_wavelength_mm,
            pos_max_wavelength_mm=pos_max_wavelength_mm,
        )
        self.study4_memory_builder = CLSConditionedMemoryBuilder(d_model, dropout=dropout)
        self.study4_decoder = CLSConditionedPatchDecoder(
            dim=d_model,
            patch_dim=patch_dim,
            num_heads=num_heads,
            num_layers=mim_decoder_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_min_wavelength_mm=pos_min_wavelength_mm,
            pos_max_wavelength_mm=pos_max_wavelength_mm,
        )
        self.register_tokens = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

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

    def _register_context(
        self,
        *,
        batch_size: int,
        n_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if n_tokens <= 0:
            return torch.empty((batch_size, 0, self.register_tokens.shape[-1]), device=device, dtype=dtype)
        n_tokens = min(int(n_tokens), self.register_tokens.shape[1])
        return self.register_tokens[:, :n_tokens, :].expand(batch_size, -1, -1).to(device=device, dtype=dtype)

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

    def forward_study4(
        self,
        patches_views: torch.Tensor,
        positions_views: torch.Tensor,
        cross_valid: torch.Tensor | None = None,
    ) -> PrismModelOutput:
        bsz, n_views = patches_views.shape[:2]
        if n_views != 4:
            raise ValueError(f"study4 expects 4 views, got {n_views}")
        if cross_valid is None:
            cross_valid = torch.ones(bsz, dtype=torch.bool, device=patches_views.device)
        else:
            cross_valid = cross_valid.to(device=patches_views.device, dtype=torch.bool).reshape(bsz)

        flat_patches = patches_views.reshape(bsz * n_views, *patches_views.shape[2:])
        flat_positions = positions_views.reshape(bsz * n_views, *positions_views.shape[2:])
        vis_patches, vis_positions, masked_patches, masked_positions = self._split_visible_masked(flat_patches, flat_positions)

        encoded = self.encoder(vis_patches, vis_positions)
        direction_cls = encoded[:, 0, :]
        supcon_cls = encoded[:, 1, :]
        patch_tokens = encoded[:, 2:, :]

        direction_cls_views = direction_cls.reshape(bsz, 4, -1)
        supcon_cls_views = supcon_cls.reshape(bsz, 4, -1)
        patch_tokens_views = patch_tokens.reshape(bsz, 4, patch_tokens.shape[1], patch_tokens.shape[2])
        masked_positions_views = masked_positions.reshape(bsz, 4, masked_positions.shape[1], masked_positions.shape[2])
        masked_patches_views = masked_patches.reshape(bsz, 4, *masked_patches.shape[1:])

        fused_x = torch.cat([direction_cls_views[:, 0, :], direction_cls_views[:, 2, :]], dim=1)
        fused_y = torch.cat([direction_cls_views[:, 1, :], direction_cls_views[:, 3, :]], dim=1)
        distance_logits_x = self.distance_head(fused_x)
        distance_logits_y = self.distance_head(fused_y)

        proj_views = F.normalize(self.proj_head(supcon_cls), dim=1).reshape(bsz, 4, -1)
        memory_views = self.study4_memory_builder(
            supcon_cls=supcon_cls,
            direction_cls=direction_cls,
            patch_tokens=patch_tokens,
        ).reshape(bsz, 4, patch_tokens.shape[1], patch_tokens.shape[2])

        self_preds: list[torch.Tensor] = []
        self_targets: list[torch.Tensor] = []
        register_preds: list[torch.Tensor] = []
        register_targets: list[torch.Tensor] = []
        cross_preds: list[torch.Tensor] = []
        cross_targets: list[torch.Tensor] = []

        pairings = ((0, 1), (1, 0), (2, 3), (3, 2))
        for target_idx, other_idx in pairings:
            target_supcon = supcon_cls_views[:, target_idx, :]
            target_direction = direction_cls_views[:, target_idx, :]
            target_masked_positions = masked_positions_views[:, target_idx, :, :]
            target_masked_patches = masked_patches_views[:, target_idx, ...]
            target_memory = memory_views[:, target_idx, :, :]
            other_memory = memory_views[:, other_idx, :, :]

            pred_self = self.study4_decoder(
                target_masked_positions,
                target_supcon,
                target_direction,
                target_memory,
                self.patch_shape,
            )
            register_context = self._register_context(
                batch_size=bsz,
                n_tokens=other_memory.shape[1],
                device=target_memory.device,
                dtype=target_memory.dtype,
            )
            pred_register = self.study4_decoder(
                target_masked_positions,
                target_supcon,
                target_direction,
                torch.cat([target_memory, register_context], dim=1),
                self.patch_shape,
            )
            other_context = torch.where(
                cross_valid[:, None, None],
                other_memory,
                register_context,
            )
            pred_cross = self.study4_decoder(
                target_masked_positions,
                target_supcon,
                target_direction,
                torch.cat([target_memory, other_context], dim=1),
                self.patch_shape,
            )

            self_preds.append(pred_self)
            self_targets.append(target_masked_patches)
            register_preds.append(pred_register)
            register_targets.append(target_masked_patches)
            cross_preds.append(pred_cross)
            cross_targets.append(target_masked_patches)

        return PrismModelOutput(
            distance_logits_x=distance_logits_x,
            distance_logits_y=distance_logits_y,
            proj_views=proj_views,
            mim_self_preds=tuple(self_preds),
            mim_self_targets=tuple(self_targets),
            mim_register_preds=tuple(register_preds),
            mim_register_targets=tuple(register_targets),
            mim_cross_preds=tuple(cross_preds),
            mim_cross_targets=tuple(cross_targets),
        )
