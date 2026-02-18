"""Typed configuration schema for Prism SSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScanRecord:
    """Logical scan unit used by the streaming dataset."""

    scan_id: str
    series_id: str
    modality: str
    series_path: str
    nifti_path: str = ""


@dataclass
class RunMetadataConfig:
    name: str = "prism_ssl_baseline"
    seed: int = 42


@dataclass
class WandbConfig:
    mode: str = "online"
    project: str = "nvreason-prism-ssl"
    entity: str = ""
    tags: list[str] = field(default_factory=list)
    run_name: str = ""
    resume_id: str = ""


@dataclass
class DataConfig:
    catalog_path: str = "data/pmbb_catalog.csv.gz"
    n_scans: int = 5000
    modality_filter: tuple[str, ...] = ("CT", "MR")
    n_patches: int = 1024
    patch_mm: float = 16.0
    method: str = "optimized_fused"
    storage_mode: str = "sharded"
    workers: int = 8
    warm_pool_size: int = 16
    visits_per_scan: int = 100
    max_prefetch_replacements: int = 2
    position_frame_for_model: str = "aug"
    apply_native_orientation_hint: bool = True
    rotation_augmentation_max_degrees: float = 10.0
    use_local_scratch: bool = True
    strict_background_errors: bool = False
    broken_abort_ratio: float = 0.10
    broken_abort_min_attempts: int = 200
    max_broken_series_log: int = 2000


@dataclass
class LossConfig:
    w_distance: float = 1.0
    w_rotation: float = 1.0
    w_window: float = 1.0
    w_supcon_target: float = 0.2
    supcon_temperature: float = 0.1
    supcon_warmup_steps: int = 2000
    supcon_ramp_steps: int = 3000
    normalize_targets: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 50_000
    log_every: int = 100
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-2
    precision: str = "bf16"
    seed: int = 42
    device: str = "auto"
    allow_failures: bool = False


@dataclass
class ModelConfig:
    name: str = "patch_mlp"
    d_model: int = 256
    num_layers: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    proj_dim: int = 128


@dataclass
class CheckpointConfig:
    artifact_every_steps: int = 5000
    max_local_checkpoints: int = 1
    local_ckpt_dir: str = "~/prism_ssl/checkpoints/<run_id>"
    no_resume: bool = False


@dataclass
class QuotaConfig:
    home_soft_limit_gb: float = 25.0
    home_hard_limit_gb: float = 42.0


@dataclass
class RuntimeConfig:
    tmp_run_dir: str = ""
    summary_output: str = ""


@dataclass
class RunConfig:
    run: RunMetadataConfig = field(default_factory=RunMetadataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class RunSummary:
    status: str
    final_step: int
    throughput_effective_patches_per_sec: float
    attempted_series: int
    broken_series: int
    broken_ratio: float
    broken_series_log_path: str
    replacement_completed_count: int
    replacement_failed_count: int
    replacement_wait_time_ms_total: float
    extra: dict[str, Any] = field(default_factory=dict)
