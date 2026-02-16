"""Sharded iterable dataset with warm-pool replacement and broken-scan health guard."""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

from prism_ssl.config.schema import ScanRecord
from prism_ssl.data.preflight import SmallScanError, load_nifti_scan
from prism_ssl.utils import append_jsonl, shard_worker


class BackgroundReplacementError(RuntimeError):
    """Raised when strict replacement mode is enabled and replacement fails."""


class BrokenScanRateExceeded(RuntimeError):
    """Raised when broken-scan ratio exceeds configured threshold."""

    def __init__(
        self,
        attempted_series: int,
        broken_series: int,
        ratio: float,
        broken_series_names: list[str],
    ) -> None:
        self.attempted_series = attempted_series
        self.broken_series = broken_series
        self.ratio = ratio
        self.broken_series_names = broken_series_names
        super().__init__(
            "BrokenScanRateExceeded: "
            f"attempted={attempted_series} broken={broken_series} ratio={ratio:.4f}"
        )


@dataclass
class _ReplacementEvents:
    completed: int = 0
    failed: int = 0
    wait_ms: float = 0.0
    attempted_delta: int = 0
    broken_delta: int = 0


class WarmPool:
    """Fixed-size scan pool with asynchronous non-blocking slot replacement."""

    def __init__(
        self,
        *,
        capacity: int,
        visits_per_scan: int,
        base_patch_mm: float,
        worker_id: int,
        strict_background_errors: bool,
        max_prefetch_replacements: int,
        broken_abort_ratio: float,
        broken_abort_min_attempts: int,
        max_broken_series_log: int,
        broken_series_log_path: str,
    ) -> None:
        self._capacity = max(1, int(capacity))
        self._visits_per_scan = max(1, int(visits_per_scan))
        self._base_patch_mm = float(base_patch_mm)
        self._strict_background_errors = bool(strict_background_errors)
        self._max_prefetch_replacements = max(1, int(max_prefetch_replacements))
        self._broken_abort_ratio = float(broken_abort_ratio)
        self._broken_abort_min_attempts = int(broken_abort_min_attempts)
        self._max_broken_series_log = int(max_broken_series_log)
        self._broken_series_log_path = broken_series_log_path
        self._worker_id = int(worker_id)

        self._slots: list[dict[str, Any]] = []
        self._rr_index = 0
        self._inflight_replacements = 0
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_prefetch_replacements,
            thread_name_prefix=f"prism-warmpool-w{worker_id}",
        )

        self.replacement_completed_count = 0
        self.replacement_failed_count = 0
        self.replacement_wait_time_ms_total = 0.0

        self.attempted_series_count = 0
        self.broken_series_count = 0
        self._attempted_delta_pending = 0
        self._broken_delta_pending = 0

        self._broken_series_names: list[str] = []
        self._broken_series_seen: set[str] = set()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._slots)

    def _note_attempt(self) -> None:
        self.attempted_series_count += 1
        self._attempted_delta_pending += 1

    def _note_broken(self, series_name: str, error: Exception | str, stage: str) -> None:
        self.broken_series_count += 1
        self._broken_delta_pending += 1

        if series_name and series_name not in self._broken_series_seen:
            if len(self._broken_series_names) < self._max_broken_series_log:
                self._broken_series_names.append(series_name)
            self._broken_series_seen.add(series_name)

        append_jsonl(
            self._broken_series_log_path,
            {
                "ts": time.time(),
                "worker_id": self._worker_id,
                "series": series_name,
                "stage": stage,
                "error": str(error),
                "attempted_series": self.attempted_series_count,
                "broken_series": self.broken_series_count,
            },
        )
        self._check_abort_threshold()

    def _check_abort_threshold(self) -> None:
        attempted = self.attempted_series_count
        if attempted < self._broken_abort_min_attempts:
            return
        ratio = self.broken_series_count / float(max(attempted, 1))
        if ratio > self._broken_abort_ratio:
            raise BrokenScanRateExceeded(
                attempted_series=attempted,
                broken_series=self.broken_series_count,
                ratio=ratio,
                broken_series_names=list(self._broken_series_names),
            )

    def _drain_health_deltas(self) -> tuple[int, int]:
        attempted_delta = self._attempted_delta_pending
        broken_delta = self._broken_delta_pending
        self._attempted_delta_pending = 0
        self._broken_delta_pending = 0
        return attempted_delta, broken_delta

    def _load_one(self, record: ScanRecord) -> tuple[Any, str]:
        return load_nifti_scan(record, base_patch_mm=self._base_patch_mm)

    def try_add_initial_slot(self, record: ScanRecord) -> bool:
        self._note_attempt()
        try:
            scan, path = self._load_one(record)
        except Exception as exc:
            self._note_broken(record.series_path, exc, stage="initial_load")
            return False

        self._slots.append(
            {
                "scan_id": record.scan_id,
                "series_id": record.series_id,
                "series_path": record.series_path,
                "scan": scan,
                "visits": 0,
                "resolved_nifti_path": path,
                "replacing": False,
                "replacement_record": None,
                "replacement_scan_id": None,
                "future": None,
                "future_start": 0.0,
            }
        )
        return True

    def _start_replacement_if_possible(self, slot_idx: int) -> bool:
        if self._inflight_replacements >= self._max_prefetch_replacements:
            return False

        slot = self._slots[slot_idx]
        if not slot["replacing"] or slot["future"] is not None:
            return False

        replacement_record = slot["replacement_record"]
        if replacement_record is None:
            return False

        self._note_attempt()
        slot["future"] = self._executor.submit(self._load_one, replacement_record)
        slot["future_start"] = time.perf_counter()
        self._inflight_replacements += 1
        return True

    def request_replacement(self, slot_idx: int, record: ScanRecord) -> bool:
        slot = self._slots[slot_idx]
        if slot["replacing"]:
            return False
        slot["replacing"] = True
        slot["replacement_record"] = record
        slot["replacement_scan_id"] = record.scan_id
        return self._start_replacement_if_possible(slot_idx)

    def mark_series_broken(self, slot_idx: int, exc: Exception) -> None:
        slot = self._slots[slot_idx]
        self._note_broken(str(slot.get("series_path", "")), exc, stage="sample")

    def poll_replacements(self) -> _ReplacementEvents:
        events = _ReplacementEvents()

        for slot in self._slots:
            future: Future | None = slot["future"]
            if future is None or not future.done():
                continue

            slot["future"] = None
            self._inflight_replacements = max(0, self._inflight_replacements - 1)
            wait_ms = max(0.0, (time.perf_counter() - float(slot.get("future_start", 0.0))) * 1000.0)
            self.replacement_wait_time_ms_total += wait_ms
            events.wait_ms += wait_ms

            try:
                scan, path = future.result()
            except Exception as exc:
                events.failed += 1
                self.replacement_failed_count += 1
                failed_series = str((slot.get("replacement_record") or {}).series_path if slot.get("replacement_record") else slot.get("series_path", ""))
                self._note_broken(failed_series, exc, stage="replacement")

                slot["replacing"] = False
                slot["replacement_record"] = None
                slot["replacement_scan_id"] = None

                if self._strict_background_errors:
                    raise BackgroundReplacementError(
                        f"Replacement failed for scan_id={slot.get('scan_id', 'unknown')}"
                    ) from exc
                continue

            replacement_record: ScanRecord | None = slot["replacement_record"]
            slot["scan"] = scan
            slot["scan_id"] = slot["replacement_scan_id"]
            slot["series_id"] = replacement_record.series_id if replacement_record else slot["scan_id"]
            slot["series_path"] = replacement_record.series_path if replacement_record else slot.get("series_path", "")
            slot["resolved_nifti_path"] = path
            slot["visits"] = 0
            slot["replacing"] = False
            slot["replacement_record"] = None
            slot["replacement_scan_id"] = None

            events.completed += 1
            self.replacement_completed_count += 1

        for idx in range(len(self._slots)):
            if self._inflight_replacements >= self._max_prefetch_replacements:
                break
            self._start_replacement_if_possible(idx)

        attempted_delta, broken_delta = self._drain_health_deltas()
        events.attempted_delta = attempted_delta
        events.broken_delta = broken_delta
        return events

    def sample(self, rng: random.Random) -> tuple[int, dict[str, Any], bool] | None:
        if not self._slots:
            raise RuntimeError("WarmPool has no valid slots; cannot sample")
        available = [idx for idx, slot in enumerate(self._slots) if not slot["replacing"]]
        if not available:
            return None

        slot_idx = available[self._rr_index % len(available)]
        self._rr_index += 1
        slot = self._slots[slot_idx]
        slot["visits"] += 1
        needs_replacement = slot["visits"] >= self._visits_per_scan and not slot["replacing"]
        return slot_idx, slot, needs_replacement

    def cleanup(self) -> None:
        with self._lock:
            self._executor.shutdown(wait=True, cancel_futures=False)


class ShardedScanDataset(IterableDataset):
    """Stream samples from hash-sharded scans with warm-pool replacement."""

    def __init__(
        self,
        *,
        scan_records: list[ScanRecord],
        n_patches: int,
        base_patch_mm: float,
        method: str,
        warm_pool_size: int,
        visits_per_scan: int,
        seed: int,
        max_prefetch_replacements: int,
        strict_background_errors: bool,
        broken_abort_ratio: float,
        broken_abort_min_attempts: int,
        max_broken_series_log: int,
        broken_series_log_path: str,
        pair_views: bool = True,
    ) -> None:
        self.scan_records = list(scan_records)
        self.n_patches = int(n_patches)
        self.base_patch_mm = float(base_patch_mm)
        self.method = method
        self.warm_pool_size = int(warm_pool_size)
        self.visits_per_scan = int(visits_per_scan)
        self.seed = int(seed)
        self.max_prefetch_replacements = int(max_prefetch_replacements)
        self.strict_background_errors = bool(strict_background_errors)
        self.broken_abort_ratio = float(broken_abort_ratio)
        self.broken_abort_min_attempts = int(broken_abort_min_attempts)
        self.max_broken_series_log = int(max_broken_series_log)
        self.broken_series_log_path = broken_series_log_path
        self.pair_views = bool(pair_views)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id = 0
            n_workers = 1
        else:
            worker_id = worker.id
            n_workers = worker.num_workers

        shard_records = [r for r in self.scan_records if shard_worker(r.scan_id, n_workers) == worker_id]
        if not shard_records:
            return

        rng = random.Random(self.seed + worker_id)
        rng.shuffle(shard_records)

        effective_pool = min(max(1, self.warm_pool_size), len(shard_records))

        pool = WarmPool(
            capacity=effective_pool,
            visits_per_scan=self.visits_per_scan,
            base_patch_mm=self.base_patch_mm,
            worker_id=worker_id,
            strict_background_errors=self.strict_background_errors,
            max_prefetch_replacements=self.max_prefetch_replacements,
            broken_abort_ratio=self.broken_abort_ratio,
            broken_abort_min_attempts=self.broken_abort_min_attempts,
            max_broken_series_log=self.max_broken_series_log,
            broken_series_log_path=self.broken_series_log_path,
        )

        next_idx = 0
        inspected_for_bootstrap = 0
        while len(pool) < effective_pool and inspected_for_bootstrap < len(shard_records):
            rec = shard_records[next_idx % len(shard_records)]
            next_idx += 1
            inspected_for_bootstrap += 1
            pool.try_add_initial_slot(rec)

        if len(pool) == 0:
            attempted_delta, broken_delta = pool._drain_health_deltas()  # noqa: SLF001
            raise RuntimeError(
                "Warm pool bootstrap failed: no valid scans. "
                f"attempted={attempted_delta} broken={broken_delta} worker={worker_id}"
            )

        sample_count = 0

        def _next_record() -> ScanRecord:
            nonlocal next_idx
            rec = shard_records[next_idx % len(shard_records)]
            next_idx += 1
            return rec

        try:
            while True:
                events = pool.poll_replacements()
                sampled = pool.sample(rng)
                if sampled is None:
                    time.sleep(0.005)
                    continue
                slot_idx, slot, needs_replacement = sampled

                sample_seed = self.seed + worker_id * 1_000_000 + sample_count
                np_rng_state = np.random.get_state()
                py_rng_state = random.getstate()
                random.seed(sample_seed)
                np.random.seed(sample_seed)

                try:
                    scan = slot["scan"]
                    if self.pair_views:
                        result_a = scan.train_sample(self.n_patches, seed=sample_seed * 2, method=self.method)
                        result_b = scan.train_sample(self.n_patches, seed=sample_seed * 2 + 1, method=self.method)
                    else:
                        result_a = scan.train_sample(self.n_patches, seed=sample_seed, method=self.method)
                        result_b = result_a
                except SmallScanError as exc:
                    pool.mark_series_broken(slot_idx, exc)
                    pool.request_replacement(slot_idx, _next_record())
                    sample_count += 1
                    continue
                finally:
                    random.setstate(py_rng_state)
                    np.random.set_state(np_rng_state)

                replacement_requested = False
                if needs_replacement:
                    replacement_requested = pool.request_replacement(slot_idx, _next_record())

                sample_count += 1

                def _tensorize(result: dict[str, Any]) -> dict[str, torch.Tensor]:
                    patches = result["normalized_patches"]
                    if patches.ndim == 2:
                        patches = patches[np.newaxis, ...]
                    patches = torch.from_numpy(patches[..., np.newaxis].astype(np.float32, copy=False))
                    positions = torch.from_numpy(np.atleast_2d(result["relative_patch_centers_pt"]).astype(np.float32, copy=False))
                    rotation = torch.from_numpy(np.asarray(result["rotation_matrix_ras"], dtype=np.float32))
                    prism_center_pt = torch.from_numpy(np.asarray(result["prism_center_pt"], dtype=np.float32).reshape(3))
                    rotation_degrees = torch.from_numpy(np.asarray(result["rotation_degrees"], dtype=np.float32).reshape(3))
                    window_params = torch.from_numpy(np.asarray([result["wc"], result["ww"]], dtype=np.float32))
                    return {
                        "patches": patches,
                        "positions": positions,
                        "rotation": rotation,
                        "prism_center_pt": prism_center_pt,
                        "rotation_degrees": rotation_degrees,
                        "window_params": window_params,
                    }

                view_a = _tensorize(result_a)
                view_b = _tensorize(result_b)
                center_delta_mm = view_b["prism_center_pt"] - view_a["prism_center_pt"]
                center_distance_mm = torch.linalg.norm(center_delta_mm, dim=0)
                rotation_delta_deg = view_b["rotation_degrees"] - view_a["rotation_degrees"]
                window_delta = view_b["window_params"] - view_a["window_params"]

                yield {
                    "patches": view_a["patches"],
                    "positions": view_a["positions"],
                    "rotation": view_a["rotation"],
                    "prism_center_pt": view_a["prism_center_pt"],
                    "rotation_degrees": view_a["rotation_degrees"],
                    "window_params": view_a["window_params"],
                    "patches_a": view_a["patches"],
                    "positions_a": view_a["positions"],
                    "rotation_a": view_a["rotation"],
                    "prism_center_pt_a": view_a["prism_center_pt"],
                    "rotation_degrees_a": view_a["rotation_degrees"],
                    "window_params_a": view_a["window_params"],
                    "patches_b": view_b["patches"],
                    "positions_b": view_b["positions"],
                    "rotation_b": view_b["rotation"],
                    "prism_center_pt_b": view_b["prism_center_pt"],
                    "rotation_degrees_b": view_b["rotation_degrees"],
                    "window_params_b": view_b["window_params"],
                    "center_delta_mm": center_delta_mm,
                    "center_distance_mm": center_distance_mm,
                    "rotation_delta_deg": rotation_delta_deg,
                    "window_delta": window_delta,
                    "scan_id": slot["scan_id"],
                    "series_id": slot["series_id"],
                    "replacement_completed_count_delta": events.completed,
                    "replacement_failed_count_delta": events.failed,
                    "replacement_wait_time_ms_delta": events.wait_ms,
                    "attempted_series_delta": events.attempted_delta,
                    "broken_series_delta": events.broken_delta,
                    "replacement_requested": replacement_requested,
                }
        finally:
            pool.cleanup()
