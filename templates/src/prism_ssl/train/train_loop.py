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
from prism_ssl.train.quota_guard import compute_dir_size_gb
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


def _hours_to_seconds(hours: float) -> float:
    return max(float(hours), 0.0) * 3600.0


def _checkpoint_due(*, now_s: float, last_s: float, every_hours: float) -> bool:
    interval_s = _hours_to_seconds(every_hours)
    return interval_s > 0.0 and (now_s - last_s) >= interval_s


def run_training(config: RunConfig) -> dict[str, Any]:
    flat_config = flatten_config(config)
    result: dict[str, Any] = {"status": "ok"}
    result.update({f"cfg_{k}": v for k, v in flat_config.items()})

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
                config=flat_config,
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

    worker_scratch_dir = str(tmp_run_dir / "scan_scratch") if config.data.use_local_scratch else None
    dataset = ShardedScanDataset(
        scan_records=records,
        n_patches=config.data.n_patches,
        source_patch_mm_min=config.data.source_patch_mm_min,
        source_patch_mm_max=config.data.source_patch_mm_max,
        source_patch_mm_distribution=config.data.source_patch_mm_distribution,
        warm_pool_size=config.data.warm_pool_size,
        visits_per_scan=config.data.visits_per_scan,
        seed=config.train.seed,
        max_prefetch_replacements=config.data.max_prefetch_replacements,
        use_totalseg_body_centers=config.data.use_totalseg_body_centers,
        pair_local_curriculum_steps=config.data.pair_local_curriculum_steps,
        pair_local_final_prob=config.data.pair_local_final_prob,
        pair_local_start_radius_mm=config.data.pair_local_start_radius_mm,
        pair_local_end_radius_mm=config.data.pair_local_end_radius_mm,
        strict_background_errors=config.data.strict_background_errors,
        broken_abort_ratio=config.data.broken_abort_ratio,
        broken_abort_min_attempts=config.data.broken_abort_min_attempts,
        max_broken_series_log=config.data.max_broken_series_log,
        broken_series_log_path=str(broken_log_path),
        scratch_dir=worker_scratch_dir,
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": config.train.batch_size,
        "num_workers": config.data.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_prism_batch,
    }
    if config.data.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)
    data_iter = iter(loader)

    patch_dim = 16 * 16 * 1
    model = PrismSSLModel(
        patch_dim=patch_dim,
        n_patches=config.data.n_patches,
        model_name=config.model.name,
        d_model=config.model.d_model,
        proj_dim=config.model.proj_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        pos_min_wavelength_mm=config.model.pos_min_wavelength_mm,
        pos_max_wavelength_mm=config.model.pos_max_wavelength_mm,
        mim_mask_ratio=config.model.mim_mask_ratio,
        mim_decoder_layers=config.model.mim_decoder_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    result["model_param_count"] = int(sum(p.numel() for p in model.parameters()))
    print(
        "[model] "
        f"name={config.model.name} d_model={config.model.d_model} "
        f"layers={config.model.num_layers} heads={config.model.num_heads} "
        f"proj_dim={config.model.proj_dim} "
        f"pos_wl_mm=({config.model.pos_min_wavelength_mm:.1f},{config.model.pos_max_wavelength_mm:.1f}) "
        f"mim_mask_ratio={config.model.mim_mask_ratio:.2f} "
        f"mim_decoder_layers={config.model.mim_decoder_layers} params={result['model_param_count']}",
        flush=True,
    )
    print(
        "[train] "
        f"batch_size={config.train.batch_size} n_patches={config.data.n_patches} "
        f"n_scans={len(records)} warm_pool_size={config.data.warm_pool_size} visits_per_scan={config.data.visits_per_scan} "
        f"patch_views_per_step={config.train.batch_size * config.data.n_patches * 2} "
        f"workers={config.data.workers} log_every={config.train.log_every} "
        f"ts_body_centers={str(config.data.use_totalseg_body_centers).lower()} "
        f"pair_local_prob_final={config.data.pair_local_final_prob:.2f} "
        f"pair_local_radius_mm=({config.data.pair_local_start_radius_mm:.0f},{config.data.pair_local_end_radius_mm:.0f}) "
        f"pair_local_steps={config.data.pair_local_curriculum_steps} "
        f"source_patch_mm=({config.data.source_patch_mm_min:.1f},{config.data.source_patch_mm_max:.1f}) "
        f"source_patch_dist={config.data.source_patch_mm_distribution} "
        f"local_ckpt_every_h={config.checkpoint.local_ckpt_every_hours:.2f} "
        f"artifact_every_h={config.checkpoint.artifact_every_hours:.2f} "
        f"artifact_every_steps={config.checkpoint.artifact_every_steps}",
        flush=True,
    )

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
    near_zero_target_std_windows = 0

    step_times = StepTimeTracker()
    accum = RunAccumulator()
    run_wall_t0 = time.perf_counter()
    last_local_ckpt_s = run_wall_t0
    last_artifact_ckpt_s = run_wall_t0

    try:
        for step in range(start_step, config.train.max_steps):
            step_t0 = time.perf_counter()
            _sync(device)
            data_wait_t0 = time.perf_counter()

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

            data_wait_ms = (time.perf_counter() - data_wait_t0) * 1000.0
            compute_t0 = time.perf_counter()

            patches_a = batch["patches_a"].to(device, non_blocking=True)
            positions_a = batch["positions_a"].to(device, non_blocking=True)
            patches_b = batch["patches_b"].to(device, non_blocking=True)
            positions_b = batch["positions_b"].to(device, non_blocking=True)

            batch["center_delta_mm"] = batch["center_delta_mm"].to(device, non_blocking=True)
            batch["center_distance_mm"] = batch["center_distance_mm"].to(device, non_blocking=True)
            batch["window_delta"] = batch["window_delta"].to(device, non_blocking=True)
            batch["series_instance_label"] = batch["series_instance_label"].to(device, non_blocking=True)
            batch["series_protocol_label"] = batch["series_protocol_label"].to(device, non_blocking=True)
            batch["source_patch_mm_a"] = batch["source_patch_mm_a"].to(device, non_blocking=True)
            batch["source_patch_mm_b"] = batch["source_patch_mm_b"].to(device, non_blocking=True)

            autocast_enabled = use_amp
            dtype = torch.bfloat16 if use_amp else torch.float32
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                outputs = model(patches_a, positions_a, patches_b, positions_b)
                loss_bundle, diagnostics = compute_loss_bundle(
                    outputs,
                    batch,
                    config.loss,
                    step=step,
                )

            optimizer.zero_grad(set_to_none=True)
            loss_bundle.total.backward()
            optimizer.step()
            _sync(device)
            gpu_time_s = time.perf_counter() - compute_t0

            final_step = step + 1
            pair_patches_this_step = int(patches_a.shape[0] * patches_a.shape[1])
            patch_views_this_step = pair_patches_this_step * 2
            sample_views_this_step = int(patches_a.shape[0] * 2)

            target_std_flags = [
                diagnostics["target_center_delta_std"] < 1e-6,
            ]
            if any(target_std_flags):
                near_zero_target_std_windows += 1
            else:
                near_zero_target_std_windows = 0

            if best_loss is None or float(loss_bundle.total.item()) < best_loss:
                best_loss = float(loss_bundle.total.item())

            now_s = time.perf_counter()
            if _checkpoint_due(
                now_s=now_s,
                last_s=last_local_ckpt_s,
                every_hours=config.checkpoint.local_ckpt_every_hours,
            ):
                save_checkpoint(local_last_ckpt, model, optimizer, final_step, flat_config)
                prune_local_checkpoints(local_ckpt_dir, max_keep=config.checkpoint.max_local_checkpoints)
                last_local_ckpt_s = time.perf_counter()
                print(f"[ckpt] local step={final_step} path={local_last_ckpt}", flush=True)

            if (
                run is not None
                and wandb_mod is not None
                and (
                    (
                        config.checkpoint.artifact_every_steps > 0
                        and final_step % config.checkpoint.artifact_every_steps == 0
                    )
                    or _checkpoint_due(
                        now_s=time.perf_counter(),
                        last_s=last_artifact_ckpt_s,
                        every_hours=config.checkpoint.artifact_every_hours,
                    )
                )
            ):
                temp_ckpt = tmp_run_dir / "artifact_ckpts" / f"step_{final_step}.ckpt"
                save_checkpoint(temp_ckpt, model, optimizer, final_step, flat_config)
                try:
                    upload_artifact_checkpoint(
                        run=run,
                        wandb_mod=wandb_mod,
                        artifact_name=artifact_name,
                        ckpt_path=temp_ckpt,
                        step=final_step,
                        include_best_alias=best_loss is not None and float(loss_bundle.total.item()) <= best_loss,
                    )
                    last_artifact_ckpt_s = time.perf_counter()
                    print(f"[ckpt] artifact step={final_step} name={artifact_name}", flush=True)
                finally:
                    temp_ckpt.unlink(missing_ok=True)

            step_time_s = time.perf_counter() - step_t0
            step_time_ms = step_time_s * 1000.0
            gpu_time_ms = gpu_time_s * 1000.0
            post_step_ms = max(step_time_ms - data_wait_ms - gpu_time_ms, 0.0)
            step_times.add(step_time_ms)
            step_throughput = patch_views_this_step / max(step_time_s, 1e-6)
            compute_throughput = patch_views_this_step / max(gpu_time_s, 1e-6)

            throughput_effective = accum.update_step(
                elapsed_wall_s=time.perf_counter() - run_wall_t0,
                patch_views_this_step=patch_views_this_step,
                sample_views_this_step=sample_views_this_step,
                replacement_completed_delta=batch["replacement_completed_count_delta"],
                replacement_failed_delta=batch["replacement_failed_count_delta"],
                replacement_wait_ms_delta=batch["replacement_wait_time_ms_delta"],
                attempted_series_delta=batch["attempted_series_delta"],
                broken_series_delta=batch["broken_series_delta"],
                loaded_series_delta=batch["loaded_series_delta"],
                loaded_with_body_delta=batch["loaded_with_body_delta"],
                sampled_body_center_views_delta=batch["sampled_body_center_views_delta"],
            )

            if final_step % config.train.log_every == 0:
                gpu_mem_mb = (
                    float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                    if device.type == "cuda"
                    else 0.0
                )
                gpu_reserved_mb = (
                    float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0))
                    if device.type == "cuda"
                    else 0.0
                )
                print(
                    format_step_log(
                        step=final_step,
                        total_loss=float(loss_bundle.total.item()),
                        loss_pair_relation=float(loss_bundle.pair_relation.item()),
                        pair_relation_acc=diagnostics["pair_relation_acc"],
                        pair_relation_acc_shared=diagnostics["pair_relation_acc_shared"],
                        window_acc_wc=diagnostics["window_acc_wc"],
                        window_acc_ww=diagnostics["window_acc_ww"],
                        loss_supcon_instance=float(loss_bundle.supcon_instance.item()),
                        loss_supcon_protocol=float(loss_bundle.supcon_protocol.item()),
                        loss_patch_size=float(loss_bundle.patch_size.item()),
                        loss_mim=float(loss_bundle.mim.item()),
                        w_supcon_instance=loss_bundle.w_supcon_instance,
                        w_supcon_protocol=loss_bundle.w_supcon_protocol,
                        patch_size_mae_mm=diagnostics["patch_size_mae_mm"],
                        source_patch_mm_mean=diagnostics["source_patch_mm_mean"],
                        center_distance_mm_mean=diagnostics["center_distance_mm_mean"],
                        low_variation_sample_count=diagnostics["low_variation_sample_count"],
                        step_time_ms=step_time_ms,
                        data_wait_ms=data_wait_ms,
                        gpu_time_ms=gpu_time_ms,
                        gpu_mem_peak_mb=gpu_mem_mb,
                        gpu_mem_reserved_mb=gpu_reserved_mb,
                        post_step_ms=post_step_ms,
                        step_throughput=step_throughput,
                        throughput_effective=throughput_effective,
                        broken_ratio=accum.broken_ratio,
                        ts_loaded_ratio=accum.loaded_with_body_ratio,
                        ts_view_ratio=accum.sampled_body_center_view_ratio,
                    ),
                    flush=True,
                )

                if near_zero_target_std_windows >= 5:
                    print(
                        "[warn] Target std has been near-zero for >=5 windows; verify sampling diversity.",
                        flush=True,
                    )

                metrics = {
                    "train/loss": float(loss_bundle.total.item()),
                    "train/loss_pair_relation": float(loss_bundle.pair_relation.item()),
                    "train/pair_relation_acc": diagnostics["pair_relation_acc"],
                    "train/pair_relation_acc_x": diagnostics["pair_relation_acc_x"],
                    "train/pair_relation_acc_y": diagnostics["pair_relation_acc_y"],
                    "train/pair_relation_acc_z": diagnostics["pair_relation_acc_z"],
                    "train/pair_relation_acc_shared": diagnostics["pair_relation_acc_shared"],
                    "train/window_acc_wc": diagnostics["window_acc_wc"],
                    "train/window_acc_ww": diagnostics["window_acc_ww"],
                    "train/pair_relation_valid_ratio": diagnostics["pair_relation_valid_ratio"],
                    "train/loss_supcon_instance": float(loss_bundle.supcon_instance.item()),
                    "train/loss_supcon_protocol": float(loss_bundle.supcon_protocol.item()),
                    "train/loss_patch_size": float(loss_bundle.patch_size.item()),
                    "train/loss_mim": float(loss_bundle.mim.item()),
                    "train/w_supcon_instance": float(loss_bundle.w_supcon_instance),
                    "train/w_supcon_protocol": float(loss_bundle.w_supcon_protocol),
                    "train/target_center_delta_mm_mean": diagnostics["target_center_delta_mean"],
                    "train/target_center_delta_std": diagnostics["target_center_delta_std"],
                    "train/center_distance_mm_mean": diagnostics["center_distance_mm_mean"],
                    "train/center_distance_mm_std": diagnostics["center_distance_mm_std"],
                    "train/center_distance_mm_min": diagnostics["center_distance_mm_min"],
                    "train/center_distance_mm_max": diagnostics["center_distance_mm_max"],
                    "train/mim_target_abs_mean": diagnostics["mim_target_abs_mean"],
                    "train/mim_pred_abs_mean": diagnostics["mim_pred_abs_mean"],
                    "train/mim_masked_patch_count": diagnostics["mim_masked_patch_count"],
                    "train/patch_size_mae_mm": diagnostics["patch_size_mae_mm"],
                    "train/source_patch_mm_mean": diagnostics["source_patch_mm_mean"],
                    "train/source_patch_mm_min": diagnostics["source_patch_mm_min"],
                    "train/source_patch_mm_max": diagnostics["source_patch_mm_max"],
                    "train/low_variation_sample_count": diagnostics["low_variation_sample_count"],
                    "train/low_variation_sample_ratio": diagnostics["low_variation_sample_ratio"],
                    "train/low_variation_view_count": diagnostics["low_variation_view_count"],
                    "train/low_variation_both_views_count": diagnostics["low_variation_both_views_count"],
                    "train/low_variation_view_std_threshold": diagnostics["low_variation_view_std_threshold"],
                    "train/patch_pixel_std_a_mean": diagnostics["patch_pixel_std_a_mean"],
                    "train/patch_pixel_std_b_mean": diagnostics["patch_pixel_std_b_mean"],
                    "train/supcon_instance_positives_per_anchor_mean": diagnostics["supcon_instance_positives_per_anchor_mean"],
                    "train/supcon_protocol_positives_per_anchor_mean": diagnostics["supcon_protocol_positives_per_anchor_mean"],
                    "train/step_time_ms": step_time_ms,
                    "train/data_wait_ms": data_wait_ms,
                    "train/gpu_time_ms": gpu_time_ms,
                    "train/post_step_ms": post_step_ms,
                    "train/patch_views_per_step": patch_views_this_step,
                    "train/patches_per_sec_step": step_throughput,
                    "train/patches_per_sec_compute_step": compute_throughput,
                    "train/throughput_effective_patches_per_sec": throughput_effective,
                    "train/gpu_mem_peak_mb": gpu_mem_mb,
                    "train/gpu_mem_reserved_mb": gpu_reserved_mb,
                    "data/attempted_series": accum.attempted_series_count,
                    "data/broken_series": accum.broken_series_count,
                    "data/broken_ratio": accum.broken_ratio,
                    "data/replacement_completed_count": accum.replacement_completed_count,
                    "data/replacement_failed_count": accum.replacement_failed_count,
                    "data/replacement_wait_time_ms_total": accum.replacement_wait_time_ms_total,
                    "data/loaded_series": accum.loaded_series_count,
                    "data/loaded_series_with_ts_body": accum.loaded_with_body_series_count,
                    "data/loaded_series_with_ts_body_ratio": accum.loaded_with_body_ratio,
                    "data/sampled_body_center_views": accum.sampled_body_center_view_count,
                    "data/sampled_body_center_view_ratio": accum.sampled_body_center_view_ratio,
                    "train/global_step": final_step,
                }
                if run is not None:
                    run.log(metrics, step=final_step)

        if not data_health_abort:
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
            save_checkpoint(local_last_ckpt, model, optimizer, final_step, flat_config)
            prune_local_checkpoints(local_ckpt_dir, max_keep=config.checkpoint.max_local_checkpoints)
        except Exception:
            pass

        broken_unique = _count_unique_broken_series(broken_log_path)
        if broken_unique > accum.broken_series_count:
            accum.broken_series_count = broken_unique

        result["final_step"] = int(final_step)
        result["throughput_effective_patches_per_sec"] = float(accum.throughput_effective_patches_per_sec)
        result["attempted_series"] = int(accum.attempted_series_count)
        result["broken_series"] = int(accum.broken_series_count)
        result["broken_ratio"] = float(accum.broken_ratio)
        result["loaded_series"] = int(accum.loaded_series_count)
        result["loaded_series_with_ts_body"] = int(accum.loaded_with_body_series_count)
        result["loaded_series_with_ts_body_ratio"] = float(accum.loaded_with_body_ratio)
        result["sampled_body_center_views"] = int(accum.sampled_body_center_view_count)
        result["sampled_body_center_view_ratio"] = float(accum.sampled_body_center_view_ratio)
        result["broken_series_log_path"] = str(broken_log_path)
        result["replacement_completed_count"] = int(accum.replacement_completed_count)
        result["replacement_failed_count"] = int(accum.replacement_failed_count)
        result["replacement_wait_time_ms_total"] = float(accum.replacement_wait_time_ms_total)
        result.update(build_tail_metrics(step_times))

        result["local_ckpt_path"] = str(local_last_ckpt)
        result["local_ckpt_dir_size_gb"] = float(compute_dir_size_gb(local_ckpt_dir))
        result["home_usage_gb"] = None
        result["patch_views_per_step"] = int(config.train.batch_size * config.data.n_patches * 2)
        result["hostname"] = platform.node()
        result["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
        result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        if run is not None:
            try:
                run.summary["final_status"] = result["status"]
                run.summary["final_step"] = result["final_step"]
                run.summary["broken_ratio"] = result["broken_ratio"]
                run.summary["loaded_series_with_ts_body_ratio"] = result["loaded_series_with_ts_body_ratio"]
                run.summary["sampled_body_center_view_ratio"] = result["sampled_body_center_view_ratio"]
                run.summary["throughput_effective_patches_per_sec"] = result[
                    "throughput_effective_patches_per_sec"
                ]
            finally:
                run.finish()

    return result


def write_summary(path: str, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)
