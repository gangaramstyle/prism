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


def test_vit_l_backbone_forward_shapes():
    model = PrismSSLModel(
        patch_dim=256,
        model_name="vit_l",
        d_model=144,
        proj_dim=32,
        num_layers=2,
        num_heads=8,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    out = model(*_inputs())
    assert out.pair_relation_logits.shape == (2, 5)
    assert out.proj_instance_a.shape == (2, 32)
    assert out.proj_instance_b.shape == (2, 32)
    assert out.proj_protocol_a.shape == (2, 32)
    assert out.proj_protocol_b.shape == (2, 32)
    assert out.patch_size_pred_a.shape == (2,)
    assert out.patch_size_pred_b.shape == (2,)
    assert out.mim_pred_a.shape[0] == 2
    assert out.mim_pred_a.shape[-3:] == (16, 16, 1)
    assert out.mim_pred_a.shape == out.mim_target_a.shape
    assert out.mim_pred_b.shape == out.mim_target_b.shape


def test_vit_l_backbone_second_config_forward_shapes():
    model = PrismSSLModel(
        patch_dim=256,
        model_name="vit_l",
        d_model=144,
        proj_dim=64,
        num_layers=2,
        num_heads=6,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    out = model(*_inputs())
    assert out.pair_relation_logits.shape == (2, 5)
    assert out.proj_instance_a.shape == (2, 64)
    assert out.proj_instance_b.shape == (2, 64)
    assert out.proj_protocol_a.shape == (2, 64)
    assert out.proj_protocol_b.shape == (2, 64)
    assert out.patch_size_pred_a.shape == (2,)
    assert out.patch_size_pred_b.shape == (2,)
    assert out.mim_pred_a.shape == out.mim_target_a.shape
    assert out.mim_pred_b.shape == out.mim_target_b.shape


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        PrismSSLModel(model_name="unknown-model")
