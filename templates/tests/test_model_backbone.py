from __future__ import annotations

import pytest
import torch

from prism_ssl.model import PrismSSLModel


def _inputs(batch: int = 2, patches: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patches_a = torch.randn(batch, patches, 16, 16, 1)
    positions_a = torch.randn(batch, patches, 3)
    patches_b = torch.randn(batch, patches, 16, 16, 1)
    positions_b = torch.randn(batch, patches, 3)
    return patches_a, positions_a, patches_b, positions_b


def test_patch_mlp_backbone_forward_shapes():
    model = PrismSSLModel(
        patch_dim=256,
        model_name="patch_mlp",
        d_model=64,
        proj_dim=32,
        dropout=0.0,
    )
    out = model(*_inputs())
    assert out.distance_mm.shape == (2,)
    assert out.rotation_delta_deg.shape == (2, 3)
    assert out.window_delta.shape == (2, 2)
    assert out.proj_a.shape == (2, 32)
    assert out.proj_b.shape == (2, 32)


def test_vit_l_backbone_forward_shapes():
    model = PrismSSLModel(
        patch_dim=256,
        model_name="vit_l",
        d_model=128,
        proj_dim=64,
        num_layers=2,
        num_heads=8,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    out = model(*_inputs())
    assert out.distance_mm.shape == (2,)
    assert out.rotation_delta_deg.shape == (2, 3)
    assert out.window_delta.shape == (2, 2)
    assert out.proj_a.shape == (2, 64)
    assert out.proj_b.shape == (2, 64)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        PrismSSLModel(model_name="unknown-model")
