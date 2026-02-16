"""Evaluation exports."""

from prism_ssl.eval.embedding_probe import cosine_similarity_matrix
from prism_ssl.eval.proxy_metrics import add_proxy_quality_column, compute_proxy_quality_score

__all__ = ["cosine_similarity_matrix", "add_proxy_quality_column", "compute_proxy_quality_score"]
