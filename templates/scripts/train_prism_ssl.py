#!/usr/bin/env python3
"""CLI entrypoint for Prism SSL training."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism_ssl.config import apply_overrides, load_run_config
from prism_ssl.train.train_loop import run_training, write_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Prism SSL v1")
    p.add_argument("--config", type=str, required=True)

    p.add_argument("--n-scans", type=int, default=None)
    p.add_argument("--catalog-path", type=str, default=None)
    p.add_argument("--n-patches", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--warm-pool-size", type=int, default=None)
    p.add_argument("--visits-per-scan", type=int, default=None)
    p.add_argument("--max-prefetch-replacements", type=int, default=None)

    p.add_argument("--model-name", type=str, default=None, choices=["patch_mlp", "vit_l"])
    p.add_argument("--model-d-model", type=int, default=None)
    p.add_argument("--model-num-layers", type=int, default=None)
    p.add_argument("--model-num-heads", type=int, default=None)
    p.add_argument("--model-mlp-ratio", type=float, default=None)
    p.add_argument("--model-dropout", type=float, default=None)
    p.add_argument("--model-proj-dim", type=int, default=None)

    p.add_argument("--loss-weight-supcon", type=float, default=None)
    p.add_argument("--supcon-warmup-steps", type=int, default=None)
    p.add_argument("--supcon-ramp-steps", type=int, default=None)

    p.add_argument("--broken-abort-ratio", type=float, default=None)
    p.add_argument("--broken-abort-min-attempts", type=int, default=None)
    p.add_argument("--max-broken-series-log", type=int, default=None)

    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])

    p.add_argument("--tmp-run-dir", type=str, default=None)
    p.add_argument("--summary-output", type=str, default=None)
    p.add_argument("--local-ckpt-dir", type=str, default=None)

    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--allow-failures", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_run_config(args.config)

    overrides = {
        "data.n_scans": args.n_scans,
        "data.catalog_path": args.catalog_path,
        "data.n_patches": args.n_patches,
        "train.batch_size": args.batch_size,
        "data.workers": args.workers,
        "data.warm_pool_size": args.warm_pool_size,
        "data.visits_per_scan": args.visits_per_scan,
        "data.max_prefetch_replacements": args.max_prefetch_replacements,
        "model.name": args.model_name,
        "model.d_model": args.model_d_model,
        "model.num_layers": args.model_num_layers,
        "model.num_heads": args.model_num_heads,
        "model.mlp_ratio": args.model_mlp_ratio,
        "model.dropout": args.model_dropout,
        "model.proj_dim": args.model_proj_dim,
        "loss.w_supcon_target": args.loss_weight_supcon,
        "loss.supcon_warmup_steps": args.supcon_warmup_steps,
        "loss.supcon_ramp_steps": args.supcon_ramp_steps,
        "data.broken_abort_ratio": args.broken_abort_ratio,
        "data.broken_abort_min_attempts": args.broken_abort_min_attempts,
        "data.max_broken_series_log": args.max_broken_series_log,
        "wandb.run_name": args.wandb_run_name,
        "wandb.project": args.wandb_project,
        "wandb.entity": args.wandb_entity,
        "wandb.mode": args.wandb_mode,
        "runtime.tmp_run_dir": args.tmp_run_dir,
        "runtime.summary_output": args.summary_output,
        "checkpoint.local_ckpt_dir": args.local_ckpt_dir,
        "checkpoint.no_resume": args.no_resume,
        "train.allow_failures": args.allow_failures,
    }
    config = apply_overrides(config, overrides)

    result = run_training(config)
    print(f"[{result['status']}] Training finished", flush=True)

    if config.runtime.summary_output:
        write_summary(config.runtime.summary_output, result)
        print(f"[summary] Wrote {config.runtime.summary_output}", flush=True)

    if result["status"] not in {"ok", "stopped_quota"} and not config.train.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
