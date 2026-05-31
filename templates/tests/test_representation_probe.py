from __future__ import annotations

import numpy as np
import polars as pl

from prism_ssl.eval.representation_probe import (
    knn_probe_table,
    nearest_neighbor_table,
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
