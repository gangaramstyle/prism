# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair",
#     "marimo",
#     "nibabel",
#     "numpy",
#     "polars",
#     "pyyaml",
#     "scipy",
#     "timm",
#     "torch",
#     "wandb",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


with app.setup:
    import os
    import sys
    from pathlib import Path
    from textwrap import dedent

    import altair as alt
    import marimo as mo
    import polars as pl

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from prism_ssl.eval.representation_probe import (
        collect_representations,
        knn_probe_table,
        label_count_table,
        load_probe_model,
        nearest_neighbor_table,
        projection_table,
        resolve_checkpoint_path,
    )

    alt.data_transformers.disable_max_rows()


@app.cell
def _():
    mo.md(
        dedent(
            """
# Prism SSL Representation Probe

Load a trained checkpoint, sample real PRISM views through the shared data path,
and inspect whether embeddings organize by series type, anatomy, contrast, and
nearest-neighbor retrieval behavior.
"""
        )
    )
    return


@app.cell
def _():
    default_catalog = os.environ.get(
        "CATALOG_PATH",
        str(Path.home() / "prism-ssl" / "templates" / "results" / "manifests" / "pmbb_catalog_near_iso.csv"),
    )
    default_artifact = os.environ.get(
        "PRISM_WANDB_ARTIFACT_REF",
        "vineeth-gangaram-penn/nvreason-prism-ssl/prism-ssl-ckpt:best",
    )
    default_checkpoint = os.environ.get("PRISM_NOTEBOOK_CHECKPOINT", "")

    checkpoint_path = mo.ui.text(label="Checkpoint path", value=default_checkpoint, full_width=True)
    artifact_ref = mo.ui.text(label="W&B artifact ref", value=default_artifact, full_width=True)
    catalog_path = mo.ui.text(label="Catalog / manifest path", value=default_catalog, full_width=True)
    device_key = mo.ui.dropdown(options=["auto", "cuda", "cpu"], value="auto", label="Device")
    n_scans = mo.ui.slider(start=4, stop=256, value=48, step=4, label="Scans")
    views_per_scan = mo.ui.slider(start=1, stop=4, value=1, step=1, label="Views per scan")
    n_patches = mo.ui.dropdown(options=[64, 128, 256, 512, 1024], value=256, label="Patches per view")
    eval_batch_size = mo.ui.slider(start=1, stop=32, value=4, step=1, label="Eval batch")
    seed = mo.ui.number(label="Seed", value=42, step=1)
    modality_csv = mo.ui.text(label="Modalities", value="CT,MR")
    run_probe = mo.ui.run_button(label="Build embeddings")

    mo.vstack(
        [
            mo.md("## Controls"),
            mo.hstack([checkpoint_path, artifact_ref]),
            catalog_path,
            mo.hstack([device_key, n_scans, views_per_scan, n_patches, eval_batch_size, seed, modality_csv]),
            run_probe,
        ]
    )
    return (
        artifact_ref,
        catalog_path,
        checkpoint_path,
        device_key,
        eval_batch_size,
        modality_csv,
        n_patches,
        n_scans,
        run_probe,
        seed,
        views_per_scan,
    )


@app.cell
def _(
    Path,
    artifact_ref,
    catalog_path,
    checkpoint_path,
    collect_representations,
    device_key,
    eval_batch_size,
    load_probe_model,
    modality_csv,
    mo,
    n_patches,
    n_scans,
    os,
    resolve_checkpoint_path,
    run_probe,
    seed,
    views_per_scan,
):
    mo.stop(not bool(run_probe.value), mo.callout("Set controls, then run `Build embeddings`.", kind="info"))

    tmp_root = Path(os.environ.get("PRISM_NOTEBOOK_TMP", f"/tmp/{os.environ.get('USER', 'user')}/prism_repr_probe"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    modalities = tuple(part.strip().upper() for part in str(modality_csv.value).split(",") if part.strip())
    ckpt_path = resolve_checkpoint_path(
        checkpoint_path=str(checkpoint_path.value),
        artifact_ref=str(artifact_ref.value),
        download_root=tmp_root,
    )
    loaded_model = load_probe_model(ckpt_path, device_key=str(device_key.value))
    repr_batch = collect_representations(
        loaded=loaded_model,
        catalog_path=str(catalog_path.value),
        n_scans=int(n_scans.value),
        views_per_scan=int(views_per_scan.value),
        n_patches=int(n_patches.value),
        seed=int(seed.value),
        batch_size=int(eval_batch_size.value),
        modality_filter=modalities,
    )
    return ckpt_path, loaded_model, repr_batch


@app.cell
def _(ckpt_path, loaded_model, mo, pl, repr_batch):
    metadata = repr_batch.metadata
    broken = repr_batch.broken
    model_summary = pl.DataFrame(
        [
            {
                "checkpoint": str(ckpt_path),
                "step": int(loaded_model.step),
                "device": str(loaded_model.device),
                "views": int(metadata.height),
                "broken_scans": int(broken.height),
                "d_model": int(loaded_model.config.model.d_model),
                "proj_dim": int(loaded_model.config.model.proj_dim),
                "position_frame": loaded_model.config.data.position_frame_for_model,
            }
        ]
    )
    mo.vstack([mo.md("## Loaded Model"), model_summary])
    return broken, metadata


@app.cell
def _(broken, mo):
    broken_view = mo.md("No broken scans while building this probe batch.") if broken.height == 0 else broken
    mo.vstack([mo.md("## Broken Scan Log"), broken_view])
    return


@app.cell
def _(label_count_table, metadata, mo):
    label_columns = ["modality", "series_family", "body_part", "contrast_bucket", "native_acquisition_plane", "manufacturer"]
    counts = label_count_table(metadata, label_columns, top_k=16)
    mo.vstack([mo.md("## Cohort Counts"), counts])
    return counts, label_columns


@app.cell
def _(label_columns, mo):
    embedding_space = mo.ui.dropdown(options=["projection", "cls"], value="projection", label="Embedding")
    color_by = mo.ui.dropdown(options=label_columns, value="body_part", label="Color by")
    neighbor_label = mo.ui.dropdown(options=label_columns, value="series_family", label="Probe label")
    mo.hstack([embedding_space, color_by, neighbor_label])
    return color_by, embedding_space, neighbor_label


@app.cell
def _(embedding_space, repr_batch):
    embeddings = repr_batch.projection_embeddings if embedding_space.value == "projection" else repr_batch.cls_embeddings
    return (embeddings,)


@app.cell
def _(alt, color_by, embeddings, metadata, mo, projection_table):
    projected, explained = projection_table(metadata, embeddings)
    chart = (
        alt.Chart(projected.to_dicts())
        .mark_circle(size=58, opacity=0.78)
        .encode(
            x=alt.X("pc1:Q", title=f"PC1 ({float(explained[0]) * 100:.1f}%)"),
            y=alt.Y("pc2:Q", title=f"PC2 ({float(explained[1]) * 100:.1f}%)"),
            color=alt.Color(f"{color_by.value}:N", title=str(color_by.value)),
            tooltip=[
                "view_id:N",
                "modality:N",
                "series_family:N",
                "body_part:N",
                "contrast_bucket:N",
                "series_description:N",
                "report_snippet:N",
            ],
        )
        .properties(height=520)
        .interactive()
    )
    mo.vstack([mo.md("## PCA Embedding Map"), chart])
    return (projected,)


@app.cell
def _(embeddings, knn_probe_table, label_columns, metadata, mo):
    probe_metrics = knn_probe_table(metadata, embeddings, label_columns=label_columns, k=5, min_count=2)
    mo.vstack([mo.md("## kNN Separability Checks"), probe_metrics])
    return (probe_metrics,)


@app.cell
def _(metadata, mo):
    max_anchor = max(int(metadata.height) - 1, 0)
    anchor_index = mo.ui.slider(start=0, stop=max_anchor, value=0, step=1, label="Anchor view")
    neighbor_k = mo.ui.slider(start=4, stop=32, value=12, step=1, label="Neighbors")
    mo.hstack([anchor_index, neighbor_k])
    return anchor_index, neighbor_k


@app.cell
def _(anchor_index, embeddings, metadata, mo, nearest_neighbor_table, neighbor_k):
    neighbors = nearest_neighbor_table(
        metadata,
        embeddings,
        anchor_index=int(anchor_index.value),
        k=int(neighbor_k.value),
    )
    cols = [
        "rank",
        "anchor",
        "cosine_similarity",
        "modality",
        "series_family",
        "body_part",
        "contrast_bucket",
        "native_acquisition_plane",
        "series_description",
        "report_snippet",
    ]
    view = neighbors.select([col for col in cols if col in neighbors.columns])
    mo.vstack([mo.md("## Nearest-Neighbor Retrieval"), view])
    return (neighbors,)


@app.cell
def _(alt, counts, mo, neighbor_label, pl, probe_metrics):
    count_chart = (
        alt.Chart(counts.filter(pl.col("label") == str(neighbor_label.value)).to_dicts())
        .mark_bar()
        .encode(
            x=alt.X("count:Q"),
            y=alt.Y("value:N", sort="-x", title=str(neighbor_label.value)),
            tooltip=["label:N", "value:N", "count:Q"],
        )
        .properties(height=360)
    )
    metric_row = probe_metrics.filter(pl.col("label") == str(neighbor_label.value))
    mo.vstack([mo.md("## Selected Label Detail"), metric_row, count_chart])
    return


if __name__ == "__main__":
    app.run()
