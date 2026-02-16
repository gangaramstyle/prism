"""Home quota guard helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def compute_dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = 0
    for root, _, files in os.walk(path):
        for filename in files:
            fp = Path(root) / filename
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                continue
    return total_bytes / (1024.0 ** 3)


def compute_home_usage_gb() -> float | None:
    home = str(Path.home())
    try:
        proc = subprocess.run(["du", "-sk", home], check=True, capture_output=True, text=True)
        kb = int(proc.stdout.split()[0])
        return kb / (1024.0 ** 2)
    except Exception:
        return None


def quota_state(home_usage_gb: float | None, soft_limit_gb: float, hard_limit_gb: float) -> str:
    if home_usage_gb is None:
        return "unknown"
    if home_usage_gb >= hard_limit_gb:
        return "hard"
    if home_usage_gb >= soft_limit_gb:
        return "soft"
    return "ok"
