"""Main training loop."""

from __future__ import annotations

import atexit
import json
import os
import platform
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prism_ssl.config import RunConfig, flatten_config
from prism_ssl.data import BrokenScanRateExceeded, ShardedScanDataset, collate_prism_batch, load_catalog, sample_scan_candidates
from prism_ssl.model import PrismSSLModel, compute_loss_bundle
from prism_ssl.train.checkpoint import (
    download_latest_artifact_checkpoint,
    ensure_tmp_env,
    load_checkpoint,
    prune_local_checkpoints,
    resolve_local_ckpt_dir,
    save_checkpoint,
    upload_artifact_checkpoint,
)
from prism_ssl.train.logging import format_step_log
from prism_ssl.train.metrics import RunAccumulator, build_tail_metrics
from prism_ssl.train.quota_guard import compute_dir_size_gb, compute_home_usage_gb, quota_state
from prism_ssl.utils import atomic_write_json, ensure_dir, expand_path, set_global_seed
from prism_ssl.utils.time import StepTimeTracker


def resolve_default_tmp_run_dir() -> str:
    user = os.environ.get("USER", "unknown")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return f"/tmp/{user}/prism_ssl/{job_id}"


def _register_tmp_cleanup(tmp_run_dir: Path) -> None:
    def _cleanup() -> None:
        shutil.rmtree(tmp_run_dir, ignore_errors=True)

    atexit.register(_cleanup)

    def _signal_handler(signum, _frame) -> None:
        _cleanup()
        raise SystemExit(int(signum))

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _parse_broken_exception(exc: Exception) -> tuple[int | None, int | None, float | None]:
    message = str(exc)
    match = re.search(r"attempted=(\d+) broken=(\d+) ratio=([0-9.]+)", message)
    if match is None:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


def _count_unique_broken_series(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    seen: set[str] = set()
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            series = str(payload.get("series", ""))
            if series:
                seen.add(series)
    return len(seen)


def run_training(config: RunConfig) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok"}
    result.update({f"cfg_{k}": v for k, v in flatten_config(config).items()})

    set_global_seed(config.train.seed)

    tmp_root = config.runtime.tmp_run_dir or resolve_default_tmp_run_dir()
    tmp_run_dir = ensure_dir(expand_path(tmp_root))
    ensure_tmp_env(tmp_run_dir, config.wandb.mode)
    _register_tmp_cleanup(tmp_run_dir)

    broken_log_path = tmp_run_dir / "broken_series.jsonl"

    if config.train.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.train.device)
    result["cfg_resolved_device"] = str(device)

    run = None
    wandb_mod = None
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if config.wandb.mode != "disabled":
        try:
            import wandb as wandb_mod  # type: ignore

            run = wandb_mod.init(
                project=config.wandb.project,
                entity=config.wandb.entity or None,
                name=config.wandb.run_name or None,
                mode=config.wandb.mode,
                id=config.wandb.resume_id or None,
                resume="allow",
                config=flatten_config(config),
                tags=config.wandb.tags,
            )
            run_id = run.id
            result["wandb_run_id"] = run.id
            result["wandb_url"] = run.url
        except Exception as exc:
            if config.wandb.mode == "online":
                raise RuntimeError(f"W&B initialization failed: {exc}") from exc
            run = None

    local_ckpt_dir = resolve_local_ckpt_dir(config.checkpoint.local_ckpt_dir, run_id)
    local_last_ckpt = local_ckpt_dir / "last.ckpt"
    artifact_name = "prism-ssl-ckpt"

    df = load_catalog(config.data.catalog_path)
    records = sample_scan_candidates(
        df,
        n_scans=config.data.n_scans,
        seed=config.train.seed,
        modality_filter=config.data.modality_filter,
    )
    if not records:
        raise RuntimeError("No candidate scan records found after modality/path filtering")
    result["cfg_resolved_n_scans"] = len(records)

    dataset = ShardedScanDataset(
        scan_records=records,
        n_patches=config.data.n_patches,
        base_patch_mm=config.data.patch_mm,
        method=config.data.method,
        warm_pool_size=config.data.warm_pool_size,
        visits_per_scan=config.data.visits_per_scan,
        seed=config.train.seed,
        max_prefetch_replacements=config.data.max_prefetch_replacements,
        strict_background_errors=config.data.strict_background_errors,
        broken_abort_ratio=config.data.broken_abort_ratio,
        broken_abort_min_attempts=config.data.broken_abort_min_attempts,
        max_broken_series_log=config.data.max_broken_series_log,
        broken_series_log_path=str(broken_log_path),
        pair_views=True,
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": config.train.batch_size,
        "num_workers": config.data.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_prism_batch,
    }
    if config.data.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(dataset, **loader_kwargs)
    data_iter = iter(loader)

    patch_dim = 16 * 16 * 1
    model = PrismSSLModel(
        patch_dim=patch_dim,
        model_name=config.model.name,
        d_model=config.model.d_model,
        proj_dim=config.model.proj_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    result["model_param_count"] = int(sum(p.numel() for p in model.parameters()))

    distance_loss_fn = nn.SmoothL1Loss()
    rotation_loss_fn = nn.SmoothL1Loss()
    window_loss_fn = nn.SmoothL1Loss()

    use_amp = config.train.precision == "bf16" and device.type == "cuda"

    start_step = 0
    if not config.checkpoint.no_resume:
        if local_last_ckpt.exists():
            try:
                start_step = load_checkpoint(local_last_ckpt, model, optimizer, device)
                print(f"[resume] Resumed from local checkpoint: {local_last_ckpt}", flush=True)
            except Exception as exc:
                print(f"[resume] Local checkpoint unusable; starting from step 0: {exc}", flush=True)
                start_step = 0
        elif run is not None and wandb_mod is not None and bool(config.wandb.resume_id):
            artifact_ckpt = download_latest_artifact_checkpoint(run, artifact_name, tmp_run_dir)
            if artifact_ckpt is not None and artifact_ckpt.exists():
                try:
                    start_step = load_checkpoint(artifact_ckpt, model, optimizer, device)
                    print(f"[resume] Resumed from W&B artifact checkpoint: {artifact_ckpt}", flush=True)
                except Exception as exc:
                    print(f"[resume] Artifact checkpoint unusable; starting from step 0: {exc}", flush=True)
                    start_step = 0

    best_loss: float | None = None
    final_step = start_step
    data_health_abort = False
    training_stopped_for_quota = False
    near_zero_target_std_windows = 0

    step_times = StepTimeTracker()
    accum = RunAccumulator()

    try:
        for step in range(start_step, config.train.max_steps):
            _sync(device)
            t0 = time.perf_counter()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            except Exception as exc:
                attempted, broken, ratio = _parse_broken_exception(exc)
                if isinstance(exc, BrokenScanRateExceeded) or "BrokenScanRateExceeded" in str(exc):
                    data_health_abort = True
                    result["status"] = "aborted_broken_ratio"
                    if attempted is not None:
                        accum.attempted_series_count = max(accum.attempted_series_count, attempted)
                    if broken is not None:
                        accum.broken_series_count = max(accum.broken_series_count, broken)
                    if ratio is not None:
                        result["broken_ratio_from_exception"] = ratio
                    break
                raise

            patches_a = batch["patches_a"].to(device, non_blocking=True)
            positions_a = batch["positions_a"].to(device, non_blocking=True)
            patches_b = batch["patches_b"].to(device, non_blocking=True)
            positions_b = batch["positions_b"].to(device, non_blocking=True)

            batch["center_distance_mm"] = batch["center_distance_mm"].to(device, non_blocking=True)
            batch["rotation_delta_deg"] = batch["rotation_delta_deg"].to(device, non_blocking=True)
            batch["window_delta"] = batch["window_delta"].to(device, non_blocking=True)
            batch["series_label"] = batch["series_label"].to(device, non_blocking=True)

            autocast_enabled = use_amp
            dtype = torch.bfloat16 if use_amp else torch.float32
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                outputs = model(patches_a, positions_a, patches_b, positions_b)
                loss_bundle, diagnostics = compute_loss_bundle(
                    outputs,
                    batch,
                    config.loss,
                    step=step,
                    distance_loss_fn=distance_loss_fn,
                    rotation_loss_fn=rotation_loss_fn,
                    window_loss_fn=window_loss_fn,
                )

            optimizer.zero_grad(set_to_none=True)
            loss_bundle.total.backward()
            optimizer.step()
            _sync(device)

            step_time_s = time.perf_counter() - t0
            step_time_ms = step_time_s * 1000.0
            step_times.add(step_time_ms)

            final_step = step + 1
            patches_this_step = int(patches_a.shape[0] * patches_a.shape[1])

            throughput_effective = accum.update_step(
                step_time_s=step_time_s,
                patches_this_step=patches_this_step,
                replacement_completed_delta=batch["replacement_completed_count_delta"],
                replacement_failed_delta=batch["replacement_failed_count_delta"],
                replacement_wait_ms_delta=batch["replacement_wait_time_ms_delta"],
                attempted_series_delta=batch["attempted_series_delta"],
                broken_series_delta=batch["broken_series_delta"],
            )

            labels = batch["series_label"]
            same = labels[:, None] == labels[None, :]
            positives = same.sum(dim=1) - 1
            positives = torch.clamp(positives, min=0)
            positives_mean = float(positives.float().mean().item())

            target_std_flags = [
                diagnostics["target_distance_std"] < 1e-6,
                diagnostics["target_rotation_std"] < 1e-6,
                diagnostics["target_window_std"] < 1e-6,
            ]
            if any(target_std_flags):
                near_zero_target_std_windows += 1
            else:
                near_zero_target_std_windows = 0

            if best_loss is None or float(loss_bundle.total.item()) < best_loss:
                best_loss = float(loss_bundle.total.item())

            if final_step % config.train.log_every == 0:
                print(
                    format_step_log(
                        step=final_step,
                        total_loss=float(loss_bundle.total.item()),
                        loss_distance=float(loss_bundle.distance.item()),
                        loss_rotation=float(loss_bundle.rotation.item()),
                        loss_window=float(loss_bundle.window.item()),
                        loss_supcon=float(loss_bundle.supcon.item()),
                        supcon_weight=loss_bundle.supcon_weight,
                        step_time_ms=step_time_ms,
                        throughput_effective=throughput_effective,
                        broken_ratio=accum.broken_ratio,
                    ),
                    flush=True,
                )

                if near_zero_target_std_windows >= 5:
                    print(
                        "[warn] Target std has been near-zero for >=5 windows; verify sampling diversity.",
                        flush=True,
                    )

                save_checkpoint(local_last_ckpt, model, optimizer, final_step, flatten_config(config))
                prune_local_checkpoints(local_ckpt_dir, max_keep=config.checkpoint.max_local_checkpoints)

                gpu_mem_mb = (
                    float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                    if device.type == "cuda"
                    else 0.0
                )

                metrics = {
                    "train/loss": float(loss_bundle.total.item()),
                    "train/loss_distance_mm": float(loss_bundle.distance.item()),
                    "train/loss_rotation_deg": float(loss_bundle.rotation.item()),
                    "train/loss_window": float(loss_bundle.window.item()),
                    "train/loss_supcon": float(loss_bundle.supcon.item()),
                    "train/w_supcon": float(loss_bundle.supcon_weight),
                    "train/target_distance_mm_mean": diagnostics["target_distance_mean"],
                    "train/target_rotation_abs_mean": diagnostics["target_rotation_abs_mean"],
                    "train/target_window_abs_mean": diagnostics["target_window_abs_mean"],
                    "train/target_distance_std": diagnostics["target_distance_std"],
                    "train/target_rotation_std": diagnostics["target_rotation_std"],
                    "train/target_window_std": diagnostics["target_window_std"],
                    "train/pred_distance_std": diagnostics["pred_distance_std"],
                    "train/pred_rotation_std": diagnostics["pred_rotation_std"],
                    "train/pred_window_std": diagnostics["pred_window_std"],
                    "train/pred_to_target_std_ratio_distance": diagnostics["pred_to_target_std_ratio_distance"],
                    "train/pred_to_target_std_ratio_rotation": diagnostics["pred_to_target_std_ratio_rotation"],
                    "train/pred_to_target_std_ratio_window": diagnostics["pred_to_target_std_ratio_window"],
                    "train/supcon_positives_per_anchor_mean": positives_mean,
                    "train/step_time_ms": step_time_ms,
                    "train/patches_per_sec_step": patches_this_step / max(step_time_s, 1e-6),
                    "train/throughput_effective_patches_per_sec": throughput_effective,
                    "train/gpu_mem_peak_mb": gpu_mem_mb,
                    "data/attempted_series": accum.attempted_series_count,
                    "data/broken_series": accum.broken_series_count,
                    "data/broken_ratio": accum.broken_ratio,
                    "data/replacement_completed_count": accum.replacement_completed_count,
                    "data/replacement_failed_count": accum.replacement_failed_count,
                    "data/replacement_wait_time_ms_total": accum.replacement_wait_time_ms_total,
                    "train/global_step": final_step,
                }
                if run is not None:
                    run.log(metrics, step=final_step)

                home_usage_gb = compute_home_usage_gb()
                quota = quota_state(home_usage_gb, config.quota.home_soft_limit_gb, config.quota.home_hard_limit_gb)
                if quota == "soft":
                    print(f"[quota] soft-limit reached: home_usage_gb={home_usage_gb:.2f}", flush=True)
                elif quota == "hard":
                    print(f"[quota] hard-limit reached: home_usage_gb={home_usage_gb:.2f}; stopping run", flush=True)
                    training_stopped_for_quota = True
                    result["status"] = "stopped_quota"
                    break

            if (
                run is not None
                and wandb_mod is not None
                and config.checkpoint.artifact_every_steps > 0
                and final_step % config.checkpoint.artifact_every_steps == 0
            ):
                temp_ckpt = tmp_run_dir / "artifact_ckpts" / f"step_{final_step}.ckpt"
                save_checkpoint(temp_ckpt, model, optimizer, final_step, flatten_config(config))
                try:
                    upload_artifact_checkpoint(
                        run=run,
                        wandb_mod=wandb_mod,
                        artifact_name=artifact_name,
                        ckpt_path=temp_ckpt,
                        step=final_step,
                        include_best_alias=best_loss is not None and float(loss_bundle.total.item()) <= best_loss,
                    )
                finally:
                    temp_ckpt.unlink(missing_ok=True)

        if not data_health_abort and not training_stopped_for_quota:
            result["status"] = "ok"

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result["status"] = "oom"
        else:
            result["status"] = "error"
            result["error_message"] = str(exc)
        if not config.train.allow_failures:
            raise
    except Exception as exc:
        result["status"] = "error"
        result["error_message"] = str(exc)
        if not config.train.allow_failures:
            raise
    finally:
        try:
            save_checkpoint(local_last_ckpt, model, optimizer, final_step, flatten_config(config))
            prune_local_checkpoints(local_ckpt_dir, max_keep=config.checkpoint.max_local_checkpoints)
        except Exception:
            pass

        # If worker aborted before all deltas surfaced, reconcile from log file.
        broken_unique = _count_unique_broken_series(broken_log_path)
        if broken_unique > accum.broken_series_count:
            accum.broken_series_count = broken_unique

        result["final_step"] = int(final_step)
        result["throughput_effective_patches_per_sec"] = float(accum.throughput_effective_patches_per_sec)
        result["attempted_series"] = int(accum.attempted_series_count)
        result["broken_series"] = int(accum.broken_series_count)
        result["broken_ratio"] = float(accum.broken_ratio)
        result["broken_series_log_path"] = str(broken_log_path)
        result["replacement_completed_count"] = int(accum.replacement_completed_count)
        result["replacement_failed_count"] = int(accum.replacement_failed_count)
        result["replacement_wait_time_ms_total"] = float(accum.replacement_wait_time_ms_total)
        result.update(build_tail_metrics(step_times))

        result["local_ckpt_path"] = str(local_last_ckpt)
        result["local_ckpt_dir_size_gb"] = float(compute_dir_size_gb(local_ckpt_dir))
        result["home_usage_gb"] = compute_home_usage_gb()
        result["hostname"] = platform.node()
        result["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
        result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        if run is not None:
            try:
                run.summary["final_status"] = result["status"]
                run.summary["final_step"] = result["final_step"]
                run.summary["broken_ratio"] = result["broken_ratio"]
                run.summary["throughput_effective_patches_per_sec"] = result[
                    "throughput_effective_patches_per_sec"
                ]
            finally:
                run.finish()

    return result


def write_summary(path: str, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)
