"""Model backbone for prism SSL."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from timm.layers import DropPath, LayerScale, Mlp
from timm.layers.attention import AttentionRope


def _build_inv_freq(n_pairs: int, min_wavelength_mm: float, max_wavelength_mm: float) -> torch.Tensor:
    if n_pairs <= 0:
        return torch.zeros(0, dtype=torch.float32)
    wl_min = max(float(min_wavelength_mm), 1e-3)
    wl_max = max(float(max_wavelength_mm), wl_min + 1e-3)
    if n_pairs == 1:
        wavelengths = torch.tensor([math.sqrt(wl_min * wl_max)], dtype=torch.float32)
    else:
        wavelengths = torch.exp(torch.linspace(math.log(wl_min), math.log(wl_max), n_pairs, dtype=torch.float32))
    return (2.0 * math.pi) / wavelengths


def _axis_slices(dim: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    base = dim // 3
    return ((0, base), (base, 2 * base), (2 * base, dim))


class AbsoluteSinCosPositionEmbedding3D(nn.Module):
    """3D absolute sinusoidal embedding for RAS-mm positions."""

    def __init__(
        self,
        dim: int,
        *,
        min_wavelength_mm: float = 4.0,
        max_wavelength_mm: float = 512.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.axis_specs: list[tuple[int, int, int]] = []
        slices = _axis_slices(self.dim)
        for axis, (start, end) in enumerate(slices):
            width = max(int(end - start), 0)
            usable = (width // 2) * 2
            n_pairs = usable // 2
            inv_freq = _build_inv_freq(n_pairs, min_wavelength_mm, max_wavelength_mm)
            self.register_buffer(f"abs_inv_freq_{axis}", inv_freq, persistent=False)
            self.axis_specs.append((int(start), int(width), int(usable)))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, N, 3] in RAS mm
        bsz, n_tokens = positions.shape[:2]
        out = positions.new_zeros((bsz, n_tokens, self.dim))
        pos = positions.float()
        for axis, (start, width, usable) in enumerate(self.axis_specs):
            if width <= 0 or usable <= 0:
                continue
            n_pairs = usable // 2
            inv_freq = getattr(self, f"abs_inv_freq_{axis}")
            theta = pos[..., axis].unsqueeze(-1) * inv_freq.view(1, 1, n_pairs)
            sin = torch.sin(theta).to(dtype=out.dtype)
            cos = torch.cos(theta).to(dtype=out.dtype)
            out[..., start : start + usable] = torch.stack([sin, cos], dim=-1).flatten(-2)
        return out


class MultiAxisRoPEEmbeddingCat(nn.Module):
    """Build cat-format rotary embeddings for timm AttentionRope."""

    def __init__(
        self,
        head_dim: int,
        *,
        min_wavelength_mm: float = 4.0,
        max_wavelength_mm: float = 512.0,
    ) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.axis_specs: list[tuple[int, int, int]] = []
        slices = _axis_slices(self.head_dim)
        for axis, (start, end) in enumerate(slices):
            width = max(int(end - start), 0)
            usable = (width // 2) * 2
            n_pairs = usable // 2
            inv_freq = _build_inv_freq(n_pairs, min_wavelength_mm, max_wavelength_mm)
            self.register_buffer(f"rope_inv_freq_{axis}", inv_freq, persistent=False)
            self.axis_specs.append((int(start), int(width), int(usable)))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, N, 3] -> rope emb: [B, 1, N, 2*head_dim]
        bsz, n_tokens = positions.shape[:2]
        sin = positions.new_zeros((bsz, n_tokens, self.head_dim))
        cos = positions.new_ones((bsz, n_tokens, self.head_dim))
        pos = positions.float()
        for axis, (start, width, usable) in enumerate(self.axis_specs):
            if width <= 0 or usable <= 0:
                continue
            n_pairs = usable // 2
            inv_freq = getattr(self, f"rope_inv_freq_{axis}")
            theta = pos[..., axis].unsqueeze(-1) * inv_freq.view(1, 1, n_pairs)
            sin_axis = torch.repeat_interleave(torch.sin(theta), repeats=2, dim=-1).to(dtype=sin.dtype)
            cos_axis = torch.repeat_interleave(torch.cos(theta), repeats=2, dim=-1).to(dtype=cos.dtype)
            sin[..., start : start + usable] = sin_axis
            cos[..., start : start + usable] = cos_axis
        emb = torch.cat([sin, cos], dim=-1)
        return emb.unsqueeze(1)


class TimmRoPEBlock(nn.Module):
    """Transformer block composed from timm AttentionRope + MLP primitives."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        dropout: float,
        drop_path: float = 0.0,
        init_values: float | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionRope(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            num_prefix_tokens=1,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(float(dim) * float(mlp_ratio)),
            act_layer=nn.GELU,
            drop=dropout,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope=rope)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TransformerPatchPositionEncoder(nn.Module):
    """ViT-style token encoder with absolute 3D sin/cos + RoPE attention via timm."""

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
        self.pos_abs = AbsoluteSinCosPositionEmbedding3D(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self.rope_embed = MultiAxisRoPEEmbeddingCat(d_model // num_heads)
        self.blocks = nn.ModuleList(
            [
                TimmRoPEBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=0.0,
                    init_values=None,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, patches: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, n_patches = patches.shape[:2]
        x = patches.reshape(bsz, n_patches, -1)
        x = self.patch_proj(x)
        x = x + self.pos_abs(positions).to(dtype=x.dtype)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x)

        rope = self.rope_embed(positions).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x[:, 0, :])
