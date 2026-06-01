# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair",
#     "ipyniivue",
#     "marimo",
#     "nibabel",
#     "numpy",
#     "pillow",
#     "polars",
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
    from PIL import Image

    try:
        from ipyniivue import NiiVue, ShowRender

        IPYNIIVUE_AVAILABLE = True
    except ModuleNotFoundError:
        NiiVue = None
        ShowRender = None
        IPYNIIVUE_AVAILABLE = False

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from prism_ssl.eval.report_reference_probe import (
        count_table,
        filter_report_refs,
        load_report_ref_manifest,
        load_report_ref_slice_preview,
        resolve_report_ref_manifest_path,
        row_at,
    )

    alt.data_transformers.disable_max_rows()


@app.cell
def _():
    mo.md(
        dedent(
            """
# PRISM Report-Reference Probe

Explore weak report references of the form `series N image M`, mapped to
NIfTI slices via DICOM series metadata.

The coordinate supervision here is intentionally slice-level. When a row has a
`high_exact_instance` mapping, the report image number matched a DICOM
`InstanceNumber`; the row does **not** contain a lesion point unless a later
annotation source provides one.
"""
        )
    )
    return


@app.cell
def _():
    default_manifest = os.environ.get("REPORT_REF_PATH", "")
    manifest_path = mo.ui.text(
        label="Report-reference manifest path (blank = full parquet if present, then smoke fallback)",
        value=default_manifest,
        full_width=True,
    )
    reload_manifest = mo.ui.run_button(label="Reload manifest")
    mo.vstack([manifest_path, reload_manifest])
    return manifest_path, reload_manifest


@app.cell
def _(REPO_ROOT, load_report_ref_manifest, manifest_path, reload_manifest, resolve_report_ref_manifest_path):
    reload_manifest
    resolved_manifest_path = resolve_report_ref_manifest_path(str(manifest_path.value), REPO_ROOT)
    report_refs_all = load_report_ref_manifest(resolved_manifest_path)
    return report_refs_all, resolved_manifest_path


@app.cell
def _(mo, report_refs_all, resolved_manifest_path):
    mo.vstack(
        [
            mo.md("## Manifest"),
            mo.md(f"`{resolved_manifest_path}`"),
            mo.md(f"Rows: **{report_refs_all.height:,}**"),
        ]
    )
    return


@app.cell
def _(pl, report_refs_all):
    organ_options = ["ALL"] + sorted(
        [
            str(v)
            for v in report_refs_all.select(pl.col("organ_hint").drop_nulls().unique()).to_series().to_list()
            if str(v)
        ]
    )
    confidence_options = ["high_exact_instance", "mapped", "all"] + sorted(
        [
            str(v)
            for v in report_refs_all.select(pl.col("slice_mapping_confidence").drop_nulls().unique()).to_series().to_list()
            if str(v) not in {"high_exact_instance"}
        ]
    )
    modality_options = ["ALL"] + sorted(
        [
            str(v)
            for v in report_refs_all.select(pl.col("modality").drop_nulls().unique()).to_series().to_list()
            if str(v)
        ]
    )
    return confidence_options, modality_options, organ_options


@app.cell
def _(confidence_options, modality_options, mo, organ_options):
    confidence_filter = mo.ui.dropdown(
        label="Slice mapping confidence",
        options=confidence_options,
        value="high_exact_instance" if "high_exact_instance" in confidence_options else confidence_options[0],
    )
    modality_filter = mo.ui.dropdown(label="Modality", options=modality_options, value="ALL")
    organ_filter = mo.ui.dropdown(label="Organ hint", options=organ_options, value="ALL")
    section_contains = mo.ui.text(label="Report section contains", value="")
    mo.hstack([confidence_filter, modality_filter, organ_filter, section_contains])
    return confidence_filter, modality_filter, organ_filter, section_contains


@app.cell
def _(confidence_filter, filter_report_refs, modality_filter, organ_filter, report_refs_all, section_contains):
    report_refs = filter_report_refs(
        report_refs_all,
        confidence_filter=str(confidence_filter.value),
        modality_filter=str(modality_filter.value),
        organ_filter=str(organ_filter.value),
        section_contains=str(section_contains.value),
    )
    return (report_refs,)


@app.cell
def _(alt, count_table, mo, report_refs):
    confidence_counts = count_table(report_refs, ["slice_mapping_confidence"], top_k=30)
    organ_counts = count_table(report_refs, ["organ_hint"], top_k=30)
    modality_organ_counts = count_table(report_refs, ["modality", "organ_hint"], top_k=200)
    section_counts = count_table(report_refs, ["report_section"], top_k=30)

    confidence_chart = (
        alt.Chart({"values": confidence_counts.to_dicts()})
        .mark_bar()
        .encode(
            x=alt.X("count:Q"),
            y=alt.Y("slice_mapping_confidence:N", sort="-x"),
            tooltip=["slice_mapping_confidence:N", "count:Q"],
        )
        .properties(height=180)
    )
    organ_chart = (
        alt.Chart({"values": organ_counts.to_dicts()})
        .mark_bar(color="#2b7a78")
        .encode(
            x=alt.X("count:Q"),
            y=alt.Y("organ_hint:N", sort="-x"),
            tooltip=["organ_hint:N", "count:Q"],
        )
        .properties(height=360)
    )
    nesting_heatmap = (
        alt.Chart({"values": modality_organ_counts.to_dicts()})
        .mark_rect()
        .encode(
            x=alt.X("modality:N"),
            y=alt.Y("organ_hint:N"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="tealblues")),
            tooltip=["modality:N", "organ_hint:N", "count:Q"],
        )
        .properties(height=360)
    )
    section_chart = (
        alt.Chart({"values": section_counts.to_dicts()})
        .mark_bar(color="#cc8b65")
        .encode(
            x=alt.X("count:Q"),
            y=alt.Y("report_section:N", sort="-x"),
            tooltip=["report_section:N", "count:Q"],
        )
        .properties(height=360)
    )
    mo.vstack(
        [
            mo.md(f"## Filtered Cohort: {report_refs.height:,} rows"),
            mo.hstack([confidence_chart, organ_chart]),
            mo.md("### Nesting View"),
            mo.md("Heatmap-style nesting is more stable than a treemap for now; it keeps modality and organ counts readable."),
            nesting_heatmap,
            mo.md("### Top Report Sections"),
            section_chart,
        ]
    )
    return confidence_counts, modality_organ_counts, organ_counts, section_counts


@app.cell
def _(mo, report_refs):
    mo.stop(report_refs.height == 0, mo.callout("No rows match the current filters.", kind="warn"))
    row_index = mo.ui.slider(start=0, stop=max(report_refs.height - 1, 0), value=0, step=1, label="Selected row")
    max_display_px = mo.ui.slider(start=256, stop=1024, value=640, step=64, label="Slice preview max px")
    show_slice_anchor = mo.ui.checkbox(label="Show DICOM slice anchor marker", value=False)
    mo.hstack([row_index, max_display_px, show_slice_anchor])
    return max_display_px, row_index, show_slice_anchor


@app.cell
def _(report_refs, row_at, row_index):
    selected_row = row_at(report_refs, int(row_index.value))
    return (selected_row,)


@app.cell
def _(mo, pl, selected_row):
    detail_columns = [
        "modality",
        "organ_hint",
        "report_section",
        "slice_mapping_confidence",
        "series_match_confidence",
        "series_number_reported",
        "image_number_reported",
        "dicom_instance_number",
        "slice_axis_name",
        "canonical_slice_index",
        "series_description",
        "nifti_path",
    ]
    details = pl.DataFrame([{col: selected_row.get(col) for col in detail_columns}])
    sentence = selected_row.get("sentence") or ""
    mo.vstack(
        [
            mo.md("## Selected Reference"),
            details,
            mo.md("### Report Sentence"),
            mo.callout(str(sentence), kind="neutral"),
        ]
    )
    return details, sentence


@app.cell
def _(Image, load_report_ref_slice_preview, max_display_px, mo, selected_row, show_slice_anchor):
    preview = load_report_ref_slice_preview(
        selected_row,
        max_display_px=int(max_display_px.value),
        show_slice_anchor=bool(show_slice_anchor.value),
    )
    image = Image.fromarray(preview.image_rgb)
    mo.vstack(
        [
            mo.md(
                f"## Referenced Slice Preview\n\n"
                f"Axis `{preview.axis_name}`, slice `{preview.slice_index}`, "
                f"shape `{preview.shape}`, window `{preview.window_center:.1f}/{preview.window_width:.1f}`."
            ),
            image,
            mo.callout(preview.note, kind="warn"),
        ]
    )
    return image, preview


@app.cell
def _(IPYNIIVUE_AVAILABLE, NiiVue, ShowRender, mo, selected_row):
    if not IPYNIIVUE_AVAILABLE:
        viewer = mo.callout(
            "`ipyniivue` is not installed in this environment. Launch with the report-reference marimo job script "
            "or install it into the active uv environment to enable the 3D viewer.",
            kind="warn",
        )
    else:
        viewer = NiiVue(
            back_color=(1, 1, 1, 1),
            show_3d_crosshair=True,
            multiplanar_show_render=ShowRender.ALWAYS,
            yoke_3d_to_2d_zoom=True,
        )
        viewer.load_volumes([{"path": str(selected_row.get("nifti_path")), "name": str(selected_row.get("series_description"))}])
        if hasattr(viewer, "set_clip_plane"):
            viewer.set_clip_plane(-0.2, 0, 120)
    mo.vstack(
        [
            mo.md("## Optional NiiVue Viewer"),
            mo.md(
                "NiiVue gives a real 3D/multiplanar view of the selected volume. "
                "Use the 2D slice preview above for the exact report-referenced slice index."
            ),
            viewer,
        ]
    )
    return (viewer,)


@app.cell
def _(alt, count_table, mo, report_refs):
    series_counts = count_table(report_refs, ["modality", "organ_hint", "series_description"], top_k=80)
    chart = (
        alt.Chart({"values": series_counts.to_dicts()})
        .mark_bar()
        .encode(
            x=alt.X("count:Q"),
            y=alt.Y("series_description:N", sort="-x"),
            color=alt.Color("organ_hint:N"),
            tooltip=["modality:N", "organ_hint:N", "series_description:N", "count:Q"],
        )
        .properties(height=520)
    )
    mo.vstack(
        [
            mo.md("## Series-Type Nesting"),
            mo.md("This is a quick way to see whether report-derived organ labels are concentrated in a few protocols."),
            chart,
            series_counts,
        ]
    )
    return (series_counts,)


if __name__ == "__main__":
    app.run()
