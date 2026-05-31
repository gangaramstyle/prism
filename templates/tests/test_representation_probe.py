from __future__ import annotations

import numpy as np
import polars as pl

from prism_ssl.eval.representation_probe import (
    embedding_similarity_pair_table,
    embedding_similarity_summary_table,
    knn_probe_table,
    nearest_neighbor_table,
    pair_prediction_long_table,
    pair_prediction_metric_table,
    pca_2d,
    projection_table,
)


def test_pca_2d_returns_coordinates_and_explained_variance():
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.9, -0.1, 0.0],
        ],
        dtype=np.float32,
    )

    coords, explained = pca_2d(embeddings)

    assert coords.shape == (4, 2)
    assert explained.shape == (2,)
    assert float(explained[0]) > 0.9


def test_knn_probe_and_neighbor_table_use_metadata_labels():
    metadata = pl.DataFrame(
        {
            "view_id": ["a0", "a1", "b0", "b1"],
            "body_part": ["chest", "chest", "head_neck", "head_neck"],
            "series_family": ["thorax", "thorax", "brain", "brain"],
        }
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [-0.95, -0.05],
        ],
        dtype=np.float32,
    )

    probe = knn_probe_table(metadata, embeddings, label_columns=["body_part", "series_family"], k=1)
    neighbors = nearest_neighbor_table(metadata, embeddings, anchor_index=0, k=2)
    projected, explained = projection_table(metadata, embeddings)

    assert probe["accuracy"].to_list() == [1.0, 1.0]
    assert neighbors["view_id"].to_list()[0] == "a0"
    assert neighbors["view_id"].to_list()[1] == "a1"
    assert {"pc1", "pc2"}.issubset(set(projected.columns))
    assert explained.shape == (2,)


def test_similarity_pair_table_labels_relationships():
    metadata = pl.DataFrame(
        {
            "view_id": ["a0", "a1", "b0"],
            "scan_id": ["scan_a", "scan_a", "scan_b"],
            "series_id": ["series_a", "series_a", "series_b"],
            "series_family": ["thorax", "thorax", "brain"],
            "body_part": ["chest", "chest", "head_neck"],
            "modality": ["CT", "CT", "MR"],
        }
    )
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32)

    pairs = embedding_similarity_pair_table(metadata, embeddings, max_pairs=10, seed=1)
    summary = embedding_similarity_summary_table(pairs)

    assert "same_scan" in set(pairs["pair_type"].to_list())
    assert summary.height >= 1
    assert float(pairs.filter(pl.col("pair_type") == "same_scan")["cosine_similarity"][0]) > 0.9


def test_pair_prediction_tables_compute_axis_metrics():
    predictions = pl.DataFrame(
        {
            "pair_id": ["p0", "p1"],
            "scan_id": ["s0", "s1"],
            "series_family": ["thorax", "brain"],
            "body_part": ["chest", "head_neck"],
            "target_center_delta_mm_x": [1.0, -1.0],
            "pred_center_delta_mm_x": [0.5, -0.5],
            "abs_error_center_delta_mm_x": [0.5, 0.5],
            "target_center_delta_mm_y": [2.0, -2.0],
            "pred_center_delta_mm_y": [1.0, -1.0],
            "abs_error_center_delta_mm_y": [1.0, 1.0],
            "target_center_delta_mm_z": [3.0, -3.0],
            "pred_center_delta_mm_z": [2.0, -2.0],
            "abs_error_center_delta_mm_z": [1.0, 1.0],
            "target_rotation_delta_deg_x": [4.0, -4.0],
            "pred_rotation_delta_deg_x": [3.0, -3.0],
            "abs_error_rotation_delta_deg_x": [1.0, 1.0],
            "target_rotation_delta_deg_y": [5.0, -5.0],
            "pred_rotation_delta_deg_y": [3.0, -3.0],
            "abs_error_rotation_delta_deg_y": [2.0, 2.0],
            "target_rotation_delta_deg_z": [6.0, -6.0],
            "pred_rotation_delta_deg_z": [3.0, -3.0],
            "abs_error_rotation_delta_deg_z": [3.0, 3.0],
            "target_window_delta_wc": [7.0, -7.0],
            "pred_window_delta_wc": [3.0, -3.0],
            "abs_error_window_delta_wc": [4.0, 4.0],
            "target_window_delta_ww": [8.0, -8.0],
            "pred_window_delta_ww": [3.0, -3.0],
            "abs_error_window_delta_ww": [5.0, 5.0],
        }
    )

    long = pair_prediction_long_table(predictions)
    metrics = pair_prediction_metric_table(predictions)

    assert long.height == 16
    center_x = metrics.filter((pl.col("field") == "center_delta_mm") & (pl.col("axis") == "x"))
    assert center_x.height == 1
    assert float(center_x["mae"][0]) == 0.5
    assert float(center_x["sign_accuracy"][0]) == 1.0
