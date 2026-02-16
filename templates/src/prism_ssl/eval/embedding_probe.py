"""Simple embedding diagnostics."""

from __future__ import annotations

import torch


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(embeddings, dim=-1)
    return x @ x.T
