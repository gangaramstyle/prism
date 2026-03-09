"""Evaluation exports."""

from prism_ssl.eval.embedding_probe import cosine_similarity_matrix
from prism_ssl.eval.checkpoint_probe import (
    build_eval_batch,
    build_model_from_checkpoint,
    download_wandb_run_checkpoint,
    dominant_totalseg_label_for_view,
    list_wandb_run_model_artifacts,
    load_checkpoint_payload,
    masked_l1_per_view,
    nearest_neighbor_purity,
    pca_project,
    parse_wandb_run_ref,
    sample_study4_examples,
    within_between_cosine_gap,
)
from prism_ssl.eval.proxy_metrics import add_proxy_quality_column, compute_proxy_quality_score

__all__ = [
    "cosine_similarity_matrix",
    "load_checkpoint_payload",
    "build_model_from_checkpoint",
    "parse_wandb_run_ref",
    "list_wandb_run_model_artifacts",
    "download_wandb_run_checkpoint",
    "sample_study4_examples",
    "build_eval_batch",
    "dominant_totalseg_label_for_view",
    "pca_project",
    "nearest_neighbor_purity",
    "within_between_cosine_gap",
    "masked_l1_per_view",
    "add_proxy_quality_column",
    "compute_proxy_quality_score",
]
