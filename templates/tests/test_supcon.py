import torch

from prism_ssl.model.loss import supervised_contrastive_loss


def test_supcon_zero_when_no_positive_pairs():
    emb = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss = supervised_contrastive_loss(emb, labels, temp=0.1)
    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0


def test_supcon_positive_case_is_finite():
    emb = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    loss = supervised_contrastive_loss(emb, labels, temp=0.1)
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0
