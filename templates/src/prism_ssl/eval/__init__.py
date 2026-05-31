"""Evaluation exports."""

from prism_ssl.eval.embedding_probe import cosine_similarity_matrix
from prism_ssl.eval.proxy_metrics import add_proxy_quality_column, compute_proxy_quality_score
from prism_ssl.eval.representation_probe import (
    collect_representations,
    download_wandb_artifact_checkpoint,
    knn_probe_table,
    label_count_table,
    load_probe_model,
    nearest_neighbor_table,
    projection_table,
    resolve_checkpoint_path,
)

__all__ = [
    "cosine_similarity_matrix",
    "add_proxy_quality_column",
    "compute_proxy_quality_score",
    "collect_representations",
    "download_wandb_artifact_checkpoint",
    "knn_probe_table",
    "label_count_table",
    "load_probe_model",
    "nearest_neighbor_table",
    "projection_table",
    "resolve_checkpoint_path",
]
