"""Stable hashing helpers."""

from __future__ import annotations

import hashlib


def stable_int_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def shard_worker(scan_id: str, n_workers: int) -> int:
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    return stable_int_hash(scan_id) % n_workers
