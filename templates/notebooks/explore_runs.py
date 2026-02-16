import marimo

__generated_with = "0.11.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import time
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    try:
        import wandb
    except Exception:
        wandb = None

    return Path, alt, mo, os, pl, time, wandb


@app.cell
def _(Path, os, time):
    CACHE_DIR = Path(os.environ.get("PRISM_NOTEBOOK_CACHE", f"/tmp/{os.environ.get('USER', 'user')}/prism_ssl_notebook_cache"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_TTL_SECONDS = int(os.environ.get("PRISM_NOTEBOOK_CACHE_TTL", "900"))

    def _cache_path(name: str) -> Path:
        return CACHE_DIR / f"{name}.parquet"

    def _is_cache_valid(path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age <= CACHE_TTL_SECONDS

    return CACHE_DIR, CACHE_TTL_SECONDS, _cache_path, _is_cache_valid


@app.cell
def _(mo):
    view_intro = mo.md(
        """
# Prism SSL Run Explorer

W&B-first exploration notebook with optional `/tmp` caching.

Required utilities provided:
- `fetch_runs(entity, project, filters)`
- `fetch_history(run_ids, keys)`
- `compute_proxy_score(df_runs, df_hist)`
- `summarize_best_configs(df)`
"""
    )
    view_intro
    return


@app.cell
def _(mo):
    entity = mo.ui.text(label="W&B entity", value="")
    project = mo.ui.text(label="W&B project", value="nvreason-prism-ssl")
    tag_filter = mo.ui.text(label="Tag contains", value="")
    state_filter = mo.ui.text(label="State equals", value="")
    refresh = mo.ui.checkbox(label="Force refresh", value=False)
    return entity, project, refresh, state_filter, tag_filter


@app.cell
def _(Path, _cache_path, _is_cache_valid, pl, wandb):
    _CURRENT_CONTEXT = {"entity": "", "project": ""}

    def fetch_runs(entity: str, project: str, filters: dict | None = None, force_refresh: bool = False) -> pl.DataFrame:
        _CURRENT_CONTEXT["entity"] = entity
        _CURRENT_CONTEXT["project"] = project

        cache_name = f"runs_{entity}_{project}".replace("/", "_")
        path = _cache_path(cache_name)
        if (not force_refresh) and _is_cache_valid(path):
            return pl.read_parquet(path)

        if wandb is None:
            raise RuntimeError("wandb package is required for online fetch")

        api = wandb.Api(timeout=60)
        rows = []
        for run in api.runs(f"{entity}/{project}"):
            cfg = dict(run.config) if run.config else {}
            summary = dict(run.summary) if run.summary else {}
            tags = list(run.tags) if run.tags else []

            rows.append(
                {
                    "run_id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "tags": tags,
                    "created_at": str(run.created_at),
                    "loss": summary.get("train/loss", summary.get("loss")),
                    "throughput_effective_patches_per_sec": summary.get("throughput_effective_patches_per_sec", summary.get("train/throughput_effective_patches_per_sec")),
                    "stall_steps_ge_2000ms": summary.get("stall_steps_ge_2000ms"),
                    "stall_steps_ge_10000ms": summary.get("stall_steps_ge_10000ms"),
                    "replacement_completed_count": summary.get("replacement_completed_count"),
                    "replacement_failed_count": summary.get("replacement_failed_count"),
                    "broken_ratio": summary.get("broken_ratio"),
                    "cfg_batch_size": cfg.get("train.batch_size", cfg.get("batch_size")),
                    "cfg_warm_pool_size": cfg.get("data.warm_pool_size", cfg.get("warm_pool_size")),
                    "cfg_w_supcon_target": cfg.get("loss.w_supcon_target", cfg.get("loss_weight_supcon")),
                    "cfg_supcon_ramp_steps": cfg.get("loss.supcon_ramp_steps", cfg.get("supcon_ramp_steps")),
                }
            )

        df = pl.DataFrame(rows)
        if filters and len(df) > 0:
            if filters.get("state"):
                df = df.filter(pl.col("state") == str(filters["state"]))
            if filters.get("tag"):
                tag = str(filters["tag"]).lower()
                df = df.filter(pl.col("tags").map_elements(lambda t: any(tag in str(x).lower() for x in (t or [])), return_dtype=pl.Boolean))

        if len(df) > 0:
            df.write_parquet(path)
        return df

    def fetch_history(run_ids: list[str], keys: list[str], force_refresh: bool = False) -> pl.DataFrame:
        entity = _CURRENT_CONTEXT["entity"]
        project = _CURRENT_CONTEXT["project"]
        if not entity or not project:
            raise RuntimeError("Call fetch_runs(entity, project, ...) before fetch_history")

        if wandb is None:
            raise RuntimeError("wandb package is required for online fetch")

        api = wandb.Api(timeout=60)
        rows = []
        for run_id in run_ids:
            cache_name = f"hist_{entity}_{project}_{run_id}".replace("/", "_")
            path = _cache_path(cache_name)
            if (not force_refresh) and _is_cache_valid(path):
                cached = pl.read_parquet(path)
                if len(cached) > 0:
                    rows.extend(cached.to_dicts())
                continue

            run = api.run(f"{entity}/{project}/{run_id}")
            history_rows = []
            for row in run.scan_history(keys=keys):
                item = {k: row.get(k) for k in keys}
                item["run_id"] = run_id
                history_rows.append(item)
            hist_df = pl.DataFrame(history_rows) if history_rows else pl.DataFrame([])
            if len(hist_df) > 0:
                hist_df.write_parquet(path)
                rows.extend(hist_df.to_dicts())

        return pl.DataFrame(rows) if rows else pl.DataFrame([])

    def compute_proxy_score(df_runs: pl.DataFrame, df_hist: pl.DataFrame) -> pl.DataFrame:
        if len(df_runs) == 0:
            return df_runs

        runs = df_runs.with_columns(
            (
                (-10.0 * pl.col("loss").cast(pl.Float64).fill_null(0.0))
                + (0.01 * pl.col("throughput_effective_patches_per_sec").cast(pl.Float64).fill_null(0.0))
                - (0.5 * pl.col("stall_steps_ge_2000ms").cast(pl.Float64).fill_null(0.0))
                - (2.0 * pl.col("stall_steps_ge_10000ms").cast(pl.Float64).fill_null(0.0))
            ).alias("proxy_quality_score")
        )

        if len(df_hist) > 0 and "train/throughput_effective_patches_per_sec" in df_hist.columns:
            hist_agg = (
                df_hist.group_by("run_id")
                .agg(
                    [
                        pl.col("train/throughput_effective_patches_per_sec").mean().alias("hist_tput_mean"),
                        pl.col("train/step_time_ms").quantile(0.99).alias("hist_step_ms_p99"),
                    ]
                )
            )
            runs = runs.join(hist_agg, on="run_id", how="left")

        return runs

    def summarize_best_configs(df: pl.DataFrame) -> pl.DataFrame:
        if len(df) == 0 or "proxy_quality_score" not in df.columns:
            return df
        return (
            df.sort("proxy_quality_score", descending=True)
            .select(
                [
                    "run_id",
                    "name",
                    "proxy_quality_score",
                    "cfg_batch_size",
                    "cfg_warm_pool_size",
                    "cfg_w_supcon_target",
                    "cfg_supcon_ramp_steps",
                    "throughput_effective_patches_per_sec",
                    "replacement_failed_count",
                    "broken_ratio",
                ]
            )
            .head(20)
        )

    return compute_proxy_score, fetch_history, fetch_runs, summarize_best_configs


@app.cell
def _(compute_proxy_score, entity, fetch_history, fetch_runs, project, refresh, state_filter, tag_filter):
    if not entity.value or not project.value:
        runs_df = None
        hist_df = None
        scored_df = None
    else:
        filters = {
            "state": state_filter.value.strip(),
            "tag": tag_filter.value.strip(),
        }
        runs_df = fetch_runs(entity.value, project.value, filters=filters, force_refresh=refresh.value)
        run_ids = runs_df["run_id"].to_list()[:50] if runs_df is not None and len(runs_df) > 0 else []
        keys = [
            "_step",
            "train/loss",
            "train/loss_distance_mm",
            "train/loss_rotation_deg",
            "train/loss_window",
            "train/loss_supcon",
            "train/throughput_effective_patches_per_sec",
            "train/step_time_ms",
            "data/replacement_failed_count",
            "data/broken_ratio",
        ]
        hist_df = fetch_history(run_ids, keys=keys, force_refresh=refresh.value) if run_ids else None
        scored_df = compute_proxy_score(runs_df, hist_df if hist_df is not None else __import__("polars").DataFrame([]))
    return hist_df, runs_df, scored_df


@app.cell
def _(mo, runs_df, scored_df, summarize_best_configs):
    if runs_df is None:
        view_top = mo.md("Enter entity and project to load runs.")
    elif len(runs_df) == 0:
        view_top = mo.md("No runs found.")
    else:
        view_top = mo.vstack([mo.md("## Top Configs"), summarize_best_configs(scored_df)])
    view_top
    return


@app.cell
def _(alt, hist_df, mo, pl):
    if hist_df is None or len(hist_df) == 0:
        view_charts = mo.md("No history loaded for charts.")
    else:
        hist = hist_df
        if "_step" not in hist.columns:
            hist = hist.with_columns(pl.arange(0, len(hist)).alias("_step"))

        base = alt.Chart(hist.to_pandas())

        loss_chart = (
            base.mark_line()
            .encode(
                x=alt.X("_step:Q", title="Step"),
                y=alt.Y("train/loss:Q", title="Loss"),
                color="run_id:N",
            )
            .properties(title="Loss Decomposition")
        )

        throughput_chart = (
            base.mark_line()
            .encode(
                x=alt.X("_step:Q", title="Step"),
                y=alt.Y("train/throughput_effective_patches_per_sec:Q", title="Effective Throughput"),
                color="run_id:N",
            )
            .properties(title="Throughput")
        )

        step_tail_chart = (
            base.mark_line()
            .encode(
                x=alt.X("_step:Q", title="Step"),
                y=alt.Y("train/step_time_ms:Q", title="Step Time (ms)"),
                color="run_id:N",
            )
            .properties(title="Tail Latency Diagnostics")
        )

        replacement_chart = (
            base.mark_line()
            .encode(
                x=alt.X("_step:Q", title="Step"),
                y=alt.Y("data/replacement_failed_count:Q", title="Replacement Failures"),
                color="run_id:N",
            )
            .properties(title="Replacement/Failure Counters")
        )

        view_charts = mo.vstack([loss_chart, throughput_chart, step_tail_chart, replacement_chart])
    view_charts
    return


if __name__ == "__main__":
    app.run()
