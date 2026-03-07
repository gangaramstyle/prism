"""Sharded iterable dataset with warm-pool replacement and broken-scan health guard."""

from __future__ import annotations

import os
import random
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

from prism_ssl.config.schema import ScanRecord
from prism_ssl.data.preflight import SmallScanError, load_nifti_scan, resolve_nifti_path, voxel_points_to_world, world_points_to_voxel
from prism_ssl.data.sample_contract import build_dataset_item, build_study4_dataset_item
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
    loaded_delta: int = 0
    loaded_with_body_delta: int = 0


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
        use_totalseg_body_centers: bool,
        broken_abort_ratio: float,
        broken_abort_min_attempts: int,
        max_broken_series_log: int,
        broken_series_log_path: str,
        scratch_dir: str | None = None,
    ) -> None:
        self._capacity = max(1, int(capacity))
        self._visits_per_scan = max(1, int(visits_per_scan))
        self._base_patch_mm = float(base_patch_mm)
        self._strict_background_errors = bool(strict_background_errors)
        self._max_prefetch_replacements = max(1, int(max_prefetch_replacements))
        self._use_totalseg_body_centers = bool(use_totalseg_body_centers)
        self._broken_abort_ratio = float(broken_abort_ratio)
        self._broken_abort_min_attempts = int(broken_abort_min_attempts)
        self._max_broken_series_log = int(max_broken_series_log)
        self._broken_series_log_path = broken_series_log_path
        self._worker_id = int(worker_id)
        self._scratch_dir = scratch_dir

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
        self.loaded_series_count = 0
        self.loaded_with_body_series_count = 0
        self._attempted_delta_pending = 0
        self._broken_delta_pending = 0
        self._loaded_delta_pending = 0
        self._loaded_with_body_delta_pending = 0

        self._broken_series_names: list[str] = []
        self._broken_series_seen: set[str] = set()
        self._lock = threading.Lock()

        if self._scratch_dir:
            self._worker_scratch = Path(self._scratch_dir) / f"worker_{worker_id}"
            self._worker_scratch.mkdir(parents=True, exist_ok=True)
        else:
            self._worker_scratch = None

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

    def _drain_load_deltas(self) -> tuple[int, int]:
        loaded_delta = self._loaded_delta_pending
        loaded_with_body_delta = self._loaded_with_body_delta_pending
        self._loaded_delta_pending = 0
        self._loaded_with_body_delta_pending = 0
        return loaded_delta, loaded_with_body_delta

    def _note_loaded(self, scan: Any) -> None:
        self.loaded_series_count += 1
        self._loaded_delta_pending += 1
        if str(getattr(scan, "body_sampling_source", "")):
            self.loaded_with_body_series_count += 1
            self._loaded_with_body_delta_pending += 1

    def _slot_scratch_dir(self, slot_idx: int) -> Path | None:
        if self._worker_scratch is None:
            return None
        slot_dir = self._worker_scratch / f"slot_{slot_idx:03d}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        return slot_dir

    @staticmethod
    def _path_suffix(path: str) -> str:
        p = Path(path)
        if p.name.endswith(".nii.gz"):
            return ".nii.gz"
        if p.suffix:
            return p.suffix
        return ".nii"

    def _clear_slot_scratch(self, slot_idx: int) -> None:
        slot_dir = self._slot_scratch_dir(slot_idx)
        if slot_dir is None:
            return
        for entry in slot_dir.glob("*"):
            if entry.is_file():
                entry.unlink(missing_ok=True)

    def _stage_to_scratch(self, slot_idx: int, path: str) -> str:
        slot_dir = self._slot_scratch_dir(slot_idx)
        if slot_dir is None:
            return path
        self._clear_slot_scratch(slot_idx)

        suffix = self._path_suffix(path)
        dst = slot_dir / f"scan{suffix}"
        tmp = slot_dir / f"scan{suffix}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            shutil.copy2(path, tmp)
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return str(dst)

    def _load_one(self, record: ScanRecord, slot_idx: int) -> tuple[Any, str]:
        if self._worker_scratch is None:
            return load_nifti_scan(
                record,
                base_patch_mm=self._base_patch_mm,
                use_totalseg_body_centers=self._use_totalseg_body_centers,
            )

        source_path = record.nifti_path or resolve_nifti_path(record.series_path)
        staged_path = self._stage_to_scratch(slot_idx, source_path)
        staged_record = ScanRecord(
            scan_id=record.scan_id,
            series_id=record.series_id,
            modality=record.modality,
            series_path=record.series_path,
            nifti_path=staged_path,
        )
        try:
            return load_nifti_scan(
                staged_record,
                base_patch_mm=self._base_patch_mm,
                use_totalseg_body_centers=self._use_totalseg_body_centers,
            )
        except Exception:
            self._clear_slot_scratch(slot_idx)
            raise

    def try_add_initial_slot(self, record: ScanRecord) -> bool:
        slot_idx = len(self._slots)
        self._note_attempt()
        try:
            scan, path = self._load_one(record, slot_idx)
        except Exception as exc:
            self._note_broken(record.series_path, exc, stage="initial_load")
            return False
        self._note_loaded(scan)

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
                "slot_idx": slot_idx,
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
        slot["future"] = self._executor.submit(self._load_one, replacement_record, slot_idx)
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
            self._note_loaded(scan)

            events.completed += 1
            self.replacement_completed_count += 1

        for idx in range(len(self._slots)):
            if self._inflight_replacements >= self._max_prefetch_replacements:
                break
            self._start_replacement_if_possible(idx)

        attempted_delta, broken_delta = self._drain_health_deltas()
        events.attempted_delta = attempted_delta
        events.broken_delta = broken_delta
        loaded_delta, loaded_with_body_delta = self._drain_load_deltas()
        events.loaded_delta = loaded_delta
        events.loaded_with_body_delta = loaded_with_body_delta
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
            if self._worker_scratch is not None:
                shutil.rmtree(self._worker_scratch, ignore_errors=True)


class ShardedScanDataset(IterableDataset):
    """Stream samples from hash-sharded scans with warm-pool replacement."""

    def __init__(
        self,
        *,
        scan_records: list[ScanRecord],
        n_patches: int,
        base_patch_mm: float,
        warm_pool_size: int,
        visits_per_scan: int,
        seed: int,
        max_prefetch_replacements: int,
        use_totalseg_body_centers: bool,
        pair_local_curriculum_steps: int,
        pair_local_final_prob: float,
        pair_local_start_radius_mm: float,
        pair_local_end_radius_mm: float,
        strict_background_errors: bool,
        broken_abort_ratio: float,
        broken_abort_min_attempts: int,
        max_broken_series_log: int,
        broken_series_log_path: str,
        scratch_dir: str | None = None,
        pair_views: bool = True,
    ) -> None:
        self.scan_records = list(scan_records)
        self.n_patches = int(n_patches)
        self.base_patch_mm = float(base_patch_mm)
        self.warm_pool_size = int(warm_pool_size)
        self.visits_per_scan = int(visits_per_scan)
        self.seed = int(seed)
        self.max_prefetch_replacements = int(max_prefetch_replacements)
        self.use_totalseg_body_centers = bool(use_totalseg_body_centers)
        self.pair_local_curriculum_steps = max(0, int(pair_local_curriculum_steps))
        self.pair_local_final_prob = float(np.clip(pair_local_final_prob, 0.0, 1.0))
        self.pair_local_start_radius_mm = max(float(pair_local_start_radius_mm), 0.0)
        self.pair_local_end_radius_mm = max(float(pair_local_end_radius_mm), 0.0)
        self.strict_background_errors = bool(strict_background_errors)
        self.broken_abort_ratio = float(broken_abort_ratio)
        self.broken_abort_min_attempts = int(broken_abort_min_attempts)
        self.max_broken_series_log = int(max_broken_series_log)
        self.broken_series_log_path = broken_series_log_path
        self.scratch_dir = scratch_dir
        self.pair_views = bool(pair_views)

    def _pair_curriculum(self, sample_index: int) -> tuple[float, float]:
        if self.pair_local_curriculum_steps <= 0 or self.pair_local_final_prob <= 0.0:
            return 0.0, self.pair_local_start_radius_mm
        progress = min(max(int(sample_index), 0) / float(self.pair_local_curriculum_steps), 1.0)
        local_prob = self.pair_local_final_prob * progress
        radius_mm = self.pair_local_start_radius_mm + (
            self.pair_local_end_radius_mm - self.pair_local_start_radius_mm
        ) * progress
        return float(local_prob), float(max(radius_mm, 0.0))

    def _sample_pair_center_vox(
        self,
        scan: Any,
        center_a_vox: np.ndarray,
        seed: int,
        sample_index: int,
    ) -> tuple[np.ndarray, bool]:
        """Sample paired-view centers with a slow global-to-local curriculum."""
        rng = np.random.default_rng(int(seed))
        center_a = np.asarray(center_a_vox, dtype=np.int64).reshape(3)
        local_prob, local_radius_mm = self._pair_curriculum(sample_index)

        if local_prob > 0.0 and float(rng.random()) < local_prob:
            local_center, local_from_body = scan.sample_center_near_with_source(
                rng,
                center_a,
                radius_mm=local_radius_mm,
                patch_vox=scan.patch_shape_vox,
            )
            if np.any(local_center != center_a):
                return local_center, bool(local_from_body)

        for _ in range(32):
            center_b, center_b_from_body = scan.sample_prism_center_with_source(rng, patch_vox=scan.patch_shape_vox)
            if np.any(center_b != center_a):
                return center_b, bool(center_b_from_body)

        return scan.sample_center_near_with_source(
            rng,
            center_a,
            radius_mm=max(local_radius_mm, self.base_patch_mm * 2.0),
            patch_vox=scan.patch_shape_vox,
        )

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
            use_totalseg_body_centers=self.use_totalseg_body_centers,
            broken_abort_ratio=self.broken_abort_ratio,
            broken_abort_min_attempts=self.broken_abort_min_attempts,
            max_broken_series_log=self.max_broken_series_log,
            broken_series_log_path=self.broken_series_log_path,
            scratch_dir=self.scratch_dir,
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
                        result_a = scan.train_sample(self.n_patches, seed=sample_seed * 2)
                        pair_center_b_vox, pair_center_b_from_body = self._sample_pair_center_vox(
                            scan,
                            np.asarray(result_a["prism_center_vox"], dtype=np.int64),
                            seed=sample_seed * 2 + 1,
                            sample_index=sample_count,
                        )
                        result_b = scan.train_sample(
                            self.n_patches,
                            seed=sample_seed * 2 + 3,
                            subset_center_vox=pair_center_b_vox,
                            sampled_body_center=pair_center_b_from_body,
                        )
                    else:
                        result_a = scan.train_sample(self.n_patches, seed=sample_seed)
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

                yield build_dataset_item(
                    result_a=result_a,
                    result_b=result_b,
                    scan_id=str(slot["scan_id"]),
                    series_id=str(slot["series_id"]),
                    replacement_completed_count_delta=events.completed,
                    replacement_failed_count_delta=events.failed,
                    replacement_wait_time_ms_delta=events.wait_ms,
                    attempted_series_delta=events.attempted_delta,
                    broken_series_delta=events.broken_delta,
                    loaded_series_delta=events.loaded_delta,
                    loaded_with_body_delta=events.loaded_with_body_delta,
                    replacement_requested=replacement_requested,
                )
        finally:
            pool.cleanup()


@dataclass(frozen=True)
class _StudyGroup:
    study_id: str
    records: tuple[ScanRecord, ...]


class StudyShardedScanDataset(IterableDataset):
    """Stream 4-view same-study samples with optional paired-world cross reconstruction."""

    def __init__(
        self,
        *,
        scan_records: list[ScanRecord],
        n_patches: int,
        base_patch_mm: float,
        warm_pool_size: int,
        visits_per_scan: int,
        seed: int,
        use_totalseg_body_centers: bool,
        pair_local_curriculum_steps: int,
        pair_local_final_prob: float,
        pair_local_start_radius_mm: float,
        pair_local_end_radius_mm: float,
        broken_abort_ratio: float,
        broken_abort_min_attempts: int,
        max_broken_series_log: int,
        broken_series_log_path: str,
        scratch_dir: str | None = None,
    ) -> None:
        self.n_patches = int(n_patches)
        self.base_patch_mm = float(base_patch_mm)
        self.warm_pool_size = int(warm_pool_size)
        self.visits_per_scan = int(visits_per_scan)
        self.seed = int(seed)
        self.use_totalseg_body_centers = bool(use_totalseg_body_centers)
        self.pair_local_curriculum_steps = max(0, int(pair_local_curriculum_steps))
        self.pair_local_final_prob = float(np.clip(pair_local_final_prob, 0.0, 1.0))
        self.pair_local_start_radius_mm = max(float(pair_local_start_radius_mm), 0.0)
        self.pair_local_end_radius_mm = max(float(pair_local_end_radius_mm), 0.0)
        self.broken_abort_ratio = float(broken_abort_ratio)
        self.broken_abort_min_attempts = int(broken_abort_min_attempts)
        self.max_broken_series_log = int(max_broken_series_log)
        self.broken_series_log_path = broken_series_log_path
        self.scratch_dir = scratch_dir

        groups: dict[str, list[ScanRecord]] = {}
        for record in scan_records:
            groups.setdefault(record.study_id, []).append(record)
        self.study_groups = [
            _StudyGroup(study_id=study_id, records=tuple(records))
            for study_id, records in groups.items()
            if records
        ]

    def _check_abort_threshold(self, attempted_series: int, broken_series: int, broken_names: list[str]) -> None:
        if attempted_series < self.broken_abort_min_attempts:
            return
        ratio = broken_series / float(max(attempted_series, 1))
        if ratio > self.broken_abort_ratio:
            raise BrokenScanRateExceeded(
                attempted_series=attempted_series,
                broken_series=broken_series,
                ratio=ratio,
                broken_series_names=list(broken_names),
            )

    def _note_broken(self, broken_seen: set[str], broken_names: list[str], series_name: str, error: Exception | str, stage: str, worker_id: int, attempted_series: int, broken_series: int) -> None:
        if series_name and series_name not in broken_seen:
            if len(broken_names) < self.max_broken_series_log:
                broken_names.append(series_name)
            broken_seen.add(series_name)
        append_jsonl(
            self.broken_series_log_path,
            {
                "ts": time.time(),
                "worker_id": worker_id,
                "series": series_name,
                "stage": stage,
                "error": str(error),
                "attempted_series": attempted_series,
                "broken_series": broken_series,
            },
        )

    def _pair_curriculum(self, sample_index: int) -> tuple[float, float]:
        if self.pair_local_curriculum_steps <= 0 or self.pair_local_final_prob <= 0.0:
            return 0.0, self.pair_local_start_radius_mm
        progress = min(max(int(sample_index), 0) / float(self.pair_local_curriculum_steps), 1.0)
        local_prob = self.pair_local_final_prob * progress
        radius_mm = self.pair_local_start_radius_mm + (
            self.pair_local_end_radius_mm - self.pair_local_start_radius_mm
        ) * progress
        return float(local_prob), float(max(radius_mm, 0.0))

    def _sample_pair_center_vox(
        self,
        scan: Any,
        center_a_vox: np.ndarray,
        seed: int,
        sample_index: int,
    ) -> tuple[np.ndarray, bool]:
        rng = np.random.default_rng(int(seed))
        center_a = np.asarray(center_a_vox, dtype=np.int64).reshape(3)
        local_prob, local_radius_mm = self._pair_curriculum(sample_index)
        if local_prob > 0.0 and float(rng.random()) < local_prob:
            local_center, local_from_body = scan.sample_center_near_with_source(
                rng,
                center_a,
                radius_mm=local_radius_mm,
                patch_vox=scan.patch_shape_vox,
            )
            if np.any(local_center != center_a):
                return local_center, bool(local_from_body)
        for _ in range(32):
            center_b, center_b_from_body = scan.sample_prism_center_with_source(rng, patch_vox=scan.patch_shape_vox)
            if np.any(center_b != center_a):
                return center_b, bool(center_b_from_body)
        return scan.sample_center_near_with_source(
            rng,
            center_a,
            radius_mm=max(local_radius_mm, self.base_patch_mm * 2.0),
            patch_vox=scan.patch_shape_vox,
        )

    @staticmethod
    def _path_suffix(path: str) -> str:
        p = Path(path)
        if p.name.endswith(".nii.gz"):
            return ".nii.gz"
        if p.suffix:
            return p.suffix
        return ".nii"

    def _stage_to_scratch(self, path: str, worker_id: int, slot_idx: int, role: str) -> str:
        if self.scratch_dir is None:
            return path
        slot_dir = Path(self.scratch_dir) / f"study_worker_{worker_id}" / f"slot_{slot_idx:03d}" / role
        slot_dir.mkdir(parents=True, exist_ok=True)
        for entry in slot_dir.glob("*"):
            if entry.is_file():
                entry.unlink(missing_ok=True)
        suffix = self._path_suffix(path)
        dst = slot_dir / f"scan{suffix}"
        tmp = slot_dir / f"scan{suffix}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            shutil.copy2(path, tmp)
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return str(dst)

    def _load_scan(self, record: ScanRecord, worker_id: int, slot_idx: int, role: str) -> tuple[Any, str]:
        if self.scratch_dir is None:
            return load_nifti_scan(
                record,
                base_patch_mm=self.base_patch_mm,
                use_totalseg_body_centers=self.use_totalseg_body_centers,
            )
        source_path = record.nifti_path or resolve_nifti_path(record.series_path)
        staged_path = self._stage_to_scratch(source_path, worker_id, slot_idx, role)
        staged_record = ScanRecord(
            scan_id=record.scan_id,
            series_id=record.series_id,
            study_id=record.study_id,
            modality=record.modality,
            series_path=record.series_path,
            nifti_path=staged_path,
        )
        return load_nifti_scan(
            staged_record,
            base_patch_mm=self.base_patch_mm,
            use_totalseg_body_centers=self.use_totalseg_body_centers,
        )

    def _select_series_pair(self, group: _StudyGroup, rng: random.Random) -> tuple[ScanRecord, ScanRecord, str]:
        records = list(group.records)
        record_x = rng.choice(records)
        distinct = [record for record in records if record.series_id != record_x.series_id]
        if not distinct:
            return record_x, record_x, "duplicate"
        return record_x, rng.choice(distinct), "paired_world"

    def _load_study_slot(
        self,
        group: _StudyGroup,
        *,
        rng: random.Random,
        worker_id: int,
        slot_idx: int,
    ) -> tuple[dict[str, Any], int, int]:
        record_x, record_y, mode = self._select_series_pair(group, rng)
        scan_x, path_x = self._load_scan(record_x, worker_id, slot_idx, "x")
        loaded_delta = 1
        loaded_with_body_delta = int(bool(str(getattr(scan_x, "body_sampling_source", ""))))
        if record_y.series_id == record_x.series_id:
            scan_y = scan_x
            path_y = path_x
        else:
            scan_y, path_y = self._load_scan(record_y, worker_id, slot_idx, "y")
            loaded_delta += 1
            loaded_with_body_delta += int(bool(str(getattr(scan_y, "body_sampling_source", ""))))
        return (
            {
                "study_id": group.study_id,
                "record_x": record_x,
                "record_y": record_y,
                "scan_x": scan_x,
                "scan_y": scan_y,
                "resolved_nifti_path_x": path_x,
                "resolved_nifti_path_y": path_y,
                "visits": 0,
                "fallback_mode": mode,
            },
            loaded_delta,
            loaded_with_body_delta,
        )

    def _try_same_world_view(
        self,
        scan: Any,
        world_center_pt: np.ndarray,
        *,
        seed: int,
    ) -> Mapping[str, Any] | None:
        center_vox = np.rint(world_points_to_voxel(np.asarray(world_center_pt, dtype=np.float32), scan.affine)[0]).astype(np.int64)
        if not scan._patch_has_overlap(center_vox, scan.patch_shape_vox):  # noqa: SLF001
            return None
        min_idx, max_idx = scan._center_bounds_for_full_patch(scan.patch_shape_vox)  # noqa: SLF001
        center_vox = np.clip(center_vox, min_idx, max_idx)
        if not scan._patch_has_overlap(center_vox, scan.patch_shape_vox):  # noqa: SLF001
            return None
        return scan.train_sample(
            self.n_patches,
            seed=seed,
            subset_center_vox=center_vox,
            sampled_body_center=scan.contains_body_center_vox(center_vox),
        )

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id = 0
            n_workers = 1
        else:
            worker_id = worker.id
            n_workers = worker.num_workers

        shard_groups = [g for g in self.study_groups if shard_worker(g.study_id, n_workers) == worker_id]
        if not shard_groups:
            return

        rng = random.Random(self.seed + worker_id)
        rng.shuffle(shard_groups)

        effective_pool = min(max(1, self.warm_pool_size), len(shard_groups))
        slots: list[dict[str, Any]] = []
        next_idx = 0
        sample_count = 0
        rr_index = 0

        pending_completed = 0
        pending_failed = 0
        pending_wait_ms = 0.0
        pending_attempted = 0
        pending_broken = 0
        pending_loaded = 0
        pending_loaded_with_body = 0

        attempted_series = 0
        broken_series = 0
        broken_seen: set[str] = set()
        broken_names: list[str] = []

        def _next_group() -> _StudyGroup:
            nonlocal next_idx
            group = shard_groups[next_idx % len(shard_groups)]
            next_idx += 1
            return group

        while len(slots) < effective_pool and next_idx < len(shard_groups):
            group = _next_group()
            try:
                pending_attempted += len(group.records)
                attempted_series += len(group.records)
                slot, loaded_delta, loaded_with_body_delta = self._load_study_slot(
                    group,
                    rng=rng,
                    worker_id=worker_id,
                    slot_idx=len(slots),
                )
            except Exception as exc:
                pending_broken += len(group.records)
                broken_series += len(group.records)
                for record in group.records:
                    self._note_broken(
                        broken_seen,
                        broken_names,
                        record.series_path,
                        exc,
                        stage="initial_load",
                        worker_id=worker_id,
                        attempted_series=attempted_series,
                        broken_series=broken_series,
                    )
                self._check_abort_threshold(attempted_series, broken_series, broken_names)
                continue
            slots.append(slot)
            pending_loaded += loaded_delta
            pending_loaded_with_body += loaded_with_body_delta

        if not slots:
            raise RuntimeError("Study warm pool bootstrap failed: no valid study groups")

        try:
            while True:
                slot_idx = rr_index % len(slots)
                rr_index += 1
                slot = slots[slot_idx]
                slot["visits"] += 1

                sample_seed = self.seed + worker_id * 1_000_000 + sample_count
                np_rng_state = np.random.get_state()
                py_rng_state = random.getstate()
                random.seed(sample_seed)
                np.random.seed(sample_seed)

                try:
                    scan_x = slot["scan_x"]
                    result_a = scan_x.train_sample(self.n_patches, seed=sample_seed * 4)
                    pair_center_b_vox, pair_center_b_from_body = self._sample_pair_center_vox(
                        scan_x,
                        np.asarray(result_a["prism_center_vox"], dtype=np.int64),
                        seed=sample_seed * 4 + 1,
                        sample_index=sample_count,
                    )
                    result_b = scan_x.train_sample(
                        self.n_patches,
                        seed=sample_seed * 4 + 2,
                        subset_center_vox=pair_center_b_vox,
                        sampled_body_center=pair_center_b_from_body,
                    )

                    cross_valid = False
                    cross_mode = str(slot["fallback_mode"])
                    scan_y = slot["scan_y"]
                    if slot["record_y"].series_id != slot["record_x"].series_id:
                        result_ap = self._try_same_world_view(
                            scan_y,
                            np.asarray(result_a["prism_center_pt"], dtype=np.float32),
                            seed=sample_seed * 4 + 3,
                        )
                        result_bp = self._try_same_world_view(
                            scan_y,
                            np.asarray(result_b["prism_center_pt"], dtype=np.float32),
                            seed=sample_seed * 4 + 4,
                        )
                        if result_ap is not None and result_bp is not None:
                            cross_valid = True
                            cross_mode = "paired_world"
                        else:
                            result_ap = scan_y.train_sample(self.n_patches, seed=sample_seed * 4 + 5)
                            result_bp = scan_y.train_sample(self.n_patches, seed=sample_seed * 4 + 6)
                            cross_mode = "unpaired"
                    else:
                        result_ap = scan_y.train_sample(self.n_patches, seed=sample_seed * 4 + 5)
                        result_bp = scan_y.train_sample(self.n_patches, seed=sample_seed * 4 + 6)
                except SmallScanError as exc:
                    broken_series += 1
                    pending_broken += 1
                    self._note_broken(
                        broken_seen,
                        broken_names,
                        slot["record_x"].series_path,
                        exc,
                        stage="sample",
                        worker_id=worker_id,
                        attempted_series=attempted_series,
                        broken_series=broken_series,
                    )
                    self._check_abort_threshold(attempted_series, broken_series, broken_names)
                    sample_count += 1
                    continue
                finally:
                    random.setstate(py_rng_state)
                    np.random.set_state(np_rng_state)

                replacement_requested = False
                if slot["visits"] >= self.visits_per_scan:
                    replacement_requested = True
                    group = _next_group()
                    t0 = time.perf_counter()
                    pending_attempted += len(group.records)
                    attempted_series += len(group.records)
                    try:
                        new_slot, loaded_delta, loaded_with_body_delta = self._load_study_slot(
                            group,
                            rng=rng,
                            worker_id=worker_id,
                            slot_idx=slot_idx,
                        )
                    except Exception as exc:
                        pending_failed += 1
                        broken_series += len(group.records)
                        pending_broken += len(group.records)
                        for record in group.records:
                            self._note_broken(
                                broken_seen,
                                broken_names,
                                record.series_path,
                                exc,
                                stage="replacement",
                                worker_id=worker_id,
                                attempted_series=attempted_series,
                                broken_series=broken_series,
                            )
                        self._check_abort_threshold(attempted_series, broken_series, broken_names)
                    else:
                        slots[slot_idx] = new_slot
                        pending_completed += 1
                        pending_loaded += loaded_delta
                        pending_loaded_with_body += loaded_with_body_delta
                    pending_wait_ms += (time.perf_counter() - t0) * 1000.0

                sample_count += 1

                yield build_study4_dataset_item(
                    result_a=result_a,
                    result_ap=result_ap,
                    result_b=result_b,
                    result_bp=result_bp,
                    study_id=str(slot["study_id"]),
                    series_id_x=str(slot["record_x"].series_id),
                    series_id_y=str(slot["record_y"].series_id),
                    cross_valid=bool(cross_valid),
                    cross_mode=cross_mode,
                    replacement_completed_count_delta=pending_completed,
                    replacement_failed_count_delta=pending_failed,
                    replacement_wait_time_ms_delta=pending_wait_ms,
                    attempted_series_delta=pending_attempted,
                    broken_series_delta=pending_broken,
                    loaded_series_delta=pending_loaded,
                    loaded_with_body_delta=pending_loaded_with_body,
                    replacement_requested=replacement_requested,
                )

                pending_completed = 0
                pending_failed = 0
                pending_wait_ms = 0.0
                pending_attempted = 0
                pending_broken = 0
                pending_loaded = 0
                pending_loaded_with_body = 0
        finally:
            if self.scratch_dir is not None:
                worker_scratch = Path(self.scratch_dir) / f"study_worker_{worker_id}"
                shutil.rmtree(worker_scratch, ignore_errors=True)
