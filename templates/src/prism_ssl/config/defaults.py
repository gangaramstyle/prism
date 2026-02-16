"""Config loading and override helpers."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from prism_ssl.config.schema import (
    CheckpointConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    QuotaConfig,
    RunConfig,
    RunMetadataConfig,
    RuntimeConfig,
    TrainConfig,
    WandbConfig,
)


def _section_from_dict(cls: type, payload: dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in payload.items() if k in allowed}
    if cls is DataConfig and "modality_filter" in filtered:
        filtered["modality_filter"] = tuple(str(x) for x in filtered["modality_filter"])
    return cls(**filtered)


def load_run_config(config_path: str) -> RunConfig:
    """Load YAML config into typed dataclasses."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    run_cfg = _section_from_dict(RunMetadataConfig, raw.get("run", {}))
    wandb_cfg = _section_from_dict(WandbConfig, raw.get("wandb", {}))
    data_cfg = _section_from_dict(DataConfig, raw.get("data", {}))
    train_cfg = _section_from_dict(TrainConfig, raw.get("train", {}))
    model_cfg = _section_from_dict(ModelConfig, raw.get("model", {}))
    loss_cfg = _section_from_dict(LossConfig, raw.get("loss", {}))
    ckpt_cfg = _section_from_dict(CheckpointConfig, raw.get("checkpoint", {}))
    quota_cfg = _section_from_dict(QuotaConfig, raw.get("quota", {}))
    runtime_cfg = _section_from_dict(RuntimeConfig, raw.get("runtime", {}))

    # Keep global seed synchronized by default.
    if train_cfg.seed != run_cfg.seed:
        train_cfg.seed = run_cfg.seed

    return RunConfig(
        run=run_cfg,
        wandb=wandb_cfg,
        data=data_cfg,
        train=train_cfg,
        model=model_cfg,
        loss=loss_cfg,
        checkpoint=ckpt_cfg,
        quota=quota_cfg,
        runtime=runtime_cfg,
    )


def _cast_like(value: Any, current: Any) -> Any:
    if value is None:
        return current
    if isinstance(current, tuple):
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return tuple(str(value).split(","))
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if isinstance(value, list):
            return value
        return [x for x in str(value).split(",") if x]
    return value


def apply_overrides(config: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    """Apply dotted-path overrides like `train.batch_size` in-place."""
    for dotted_key, new_value in overrides.items():
        if new_value is None:
            continue
        parts = dotted_key.split(".")
        target: Any = config
        for part in parts[:-1]:
            if not hasattr(target, part):
                target = None
                break
            target = getattr(target, part)
        if target is None:
            continue
        leaf = parts[-1]
        if not hasattr(target, leaf):
            continue
        current = getattr(target, leaf)
        setattr(target, leaf, _cast_like(new_value, current))

    # Keep seeds aligned unless explicitly overridden separately.
    if config.train.seed == config.run.seed:
        config.train.seed = config.run.seed
    return config


def flatten_config(config: RunConfig) -> dict[str, Any]:
    """Flatten nested config for W&B logging."""
    flat: dict[str, Any] = {}

    def _walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                _walk(f"{prefix}.{key}" if prefix else key, value)
            return
        if isinstance(obj, (list, tuple)):
            flat[prefix] = list(obj)
            return
        if hasattr(obj, "__dataclass_fields__"):
            _walk(prefix, asdict(obj))
            return
        flat[prefix] = obj

    _walk("", config)
    return flat
