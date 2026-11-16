"""Typed configuration objects for the EnzFormer2-Act workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


WEIGHT_MODES = frozenset({"uniform"})
WEIGHT_NORMALIZATIONS = frozenset({"batch_sum", "token_mean"})
PRECISIONS = frozenset({"none", "fp32", "bf16", "fp16"})
LORA_TARGET_MODULES = frozenset({"layernorm_qkv", "out_proj"})


def _betas() -> list[float]:
    return [0.9, 0.95]


def _finite_float(value: float, name: str, *, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite and{suffix}; got {value!r}.")
    return result


@dataclass(slots=True)
class ActDataConfig:
    """Sequence-manifest settings for uniform-weight Act adaptation."""

    manifest_path: Path | None = None
    sequence_root: Path | None = None
    sequence_column: str | None = None
    seq_id_column: str | None = None
    sequence_lookup_path: Path | None = None
    weight_mode: str = "uniform"
    asset_status: str = "included"

    def __post_init__(self) -> None:
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(
                f"data.weight_mode must be one of {sorted(WEIGHT_MODES)}; got {self.weight_mode!r}."
            )
        if self.asset_status not in {
            "included",
            "external",
            "separate_archive",
            "restricted",
            "not_redistributed",
        }:
            raise ValueError(f"data.asset_status is not supported: {self.asset_status!r}.")


@dataclass(slots=True)
class ActModelConfig:
    """ESM-C and LoRA settings for the unconditional masked-language path."""

    backbone_model_name: str = "esmc_600m"
    freeze_backbone: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["layernorm_qkv", "out_proj"])

    def __post_init__(self) -> None:
        if not self.backbone_model_name:
            raise ValueError("model.backbone_model_name must be non-empty.")
        if not self.freeze_backbone:
            raise ValueError("model.freeze_backbone must be true for EnzFormer2-Act.")
        if int(self.lora_rank) <= 0:
            raise ValueError("model.lora_rank must be positive.")
        if int(self.lora_alpha) <= 0:
            raise ValueError("model.lora_alpha must be positive.")
        self.lora_dropout = _finite_float(self.lora_dropout, "model.lora_dropout", minimum=0.0)
        if self.lora_dropout >= 1.0:
            raise ValueError("model.lora_dropout must be less than 1.")
        self.lora_target_modules = [str(value) for value in self.lora_target_modules if str(value)]
        if tuple(self.lora_target_modules) != ("layernorm_qkv", "out_proj"):
            raise ValueError(
                "model.lora_target_modules must contain exactly layernorm_qkv and out_proj."
            )


@dataclass(slots=True)
class ActMaskingConfig:
    """Target sampling and context masking settings."""

    targets_per_sequence: int = 4
    context_mask_policy: str = "radius_plus_random"
    context_mask_radius: int = 0
    extra_random_mask_rate: float = 0.05
    share_context_across_targets: bool = True

    def __post_init__(self) -> None:
        if int(self.targets_per_sequence) <= 0:
            raise ValueError("masking.targets_per_sequence must be positive.")
        if self.context_mask_policy not in {"target_only", "radius", "radius_plus_random"}:
            raise ValueError("masking.context_mask_policy is not supported.")
        if int(self.context_mask_radius) < 0:
            raise ValueError("masking.context_mask_radius must be non-negative.")
        self.extra_random_mask_rate = _finite_float(
            self.extra_random_mask_rate, "masking.extra_random_mask_rate", minimum=0.0
        )
        if self.extra_random_mask_rate > 1.0:
            raise ValueError("masking.extra_random_mask_rate must be at most 1.")
        if self.context_mask_policy == "target_only" and (
            self.context_mask_radius > 0 or self.extra_random_mask_rate > 0.0
        ):
            raise ValueError(
                "masking.context_mask_radius and masking.extra_random_mask_rate are "
                "ignored when context_mask_policy='target_only'."
            )
        if self.context_mask_policy == "radius" and self.extra_random_mask_rate > 0.0:
            raise ValueError(
                "masking.extra_random_mask_rate is ignored when context_mask_policy='radius'."
            )


@dataclass(slots=True)
class ActLossConfig:
    """Cross-entropy reduction settings."""

    sample_weight_normalization: str = "token_mean"

    def __post_init__(self) -> None:
        if self.sample_weight_normalization not in WEIGHT_NORMALIZATIONS:
            raise ValueError(
                f"loss.sample_weight_normalization must be one of {sorted(WEIGHT_NORMALIZATIONS)}."
            )


@dataclass(slots=True)
class ActRuntimeConfig:
    """Portable training runtime and optimizer settings."""

    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 3
    validation_fraction: float = 0.05
    optimizer: str = "adamw"
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    betas: list[float] = field(default_factory=_betas)
    eps: float = 1.0e-8
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    amp_dtype: str = "fp32"
    seed: int = 13
    device: str | None = None
    pin_memory: bool = False
    persistent_workers: bool = False
    backbone_max_masked_rows_per_call: int | None = None
    keep_per_epoch_checkpoints: bool = True

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("runtime.batch_size must be positive.")
        if int(self.num_workers) < 0:
            raise ValueError("runtime.num_workers must be non-negative.")
        if int(self.epochs) <= 0:
            raise ValueError("runtime.epochs must be positive.")
        self.validation_fraction = _finite_float(
            self.validation_fraction, "runtime.validation_fraction", minimum=0.0
        )
        if self.validation_fraction >= 1.0:
            raise ValueError("runtime.validation_fraction must be less than 1.")
        if self.optimizer != "adamw":
            raise ValueError("runtime.optimizer must be 'adamw'.")
        self.lr = _finite_float(self.lr, "runtime.lr", minimum=0.0)
        if self.lr == 0.0:
            raise ValueError("runtime.lr must be positive.")
        self.weight_decay = _finite_float(self.weight_decay, "runtime.weight_decay", minimum=0.0)
        self.eps = _finite_float(self.eps, "runtime.eps", minimum=0.0)
        if self.eps == 0.0:
            raise ValueError("runtime.eps must be positive.")
        self.grad_clip_norm = _finite_float(
            self.grad_clip_norm, "runtime.grad_clip_norm", minimum=0.0
        )
        if int(self.grad_accum_steps) <= 0:
            raise ValueError("runtime.grad_accum_steps must be positive.")
        if self.scheduler not in {"none", "cosine", "linear"}:
            raise ValueError("runtime.scheduler must be none, cosine, or linear.")
        self.warmup_ratio = _finite_float(self.warmup_ratio, "runtime.warmup_ratio", minimum=0.0)
        if self.warmup_ratio > 1.0:
            raise ValueError("runtime.warmup_ratio must be at most 1.")
        if self.amp_dtype not in PRECISIONS:
            raise ValueError(f"runtime.amp_dtype must be one of {sorted(PRECISIONS)}.")
        self.betas = [float(value) for value in self.betas]
        if len(self.betas) != 2 or not all(0.0 <= value < 1.0 for value in self.betas):
            raise ValueError("runtime.betas must contain two values in [0, 1).")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("runtime.persistent_workers requires runtime.num_workers > 0.")
        if (
            self.backbone_max_masked_rows_per_call is not None
            and int(self.backbone_max_masked_rows_per_call) <= 0
        ):
            raise ValueError("runtime.backbone_max_masked_rows_per_call must be positive when set.")


@dataclass(slots=True)
class ActCheckpointConfig:
    """Checkpoint paths for training or scoring."""

    output_dir: Path | None = None
    init_ckpt: Path | None = None
    resume_ckpt: Path | None = None
    checkpoint: Path | None = None


@dataclass(slots=True)
class ActScanConfig:
    """Single-mutant scoring settings."""

    chunk_size: int = 64
    seq_id: str | None = None
    input_manifest: Path | None = None
    output: Path | None = None

    def __post_init__(self) -> None:
        if int(self.chunk_size) <= 0:
            raise ValueError("scan.chunk_size must be positive.")


@dataclass(slots=True)
class ActTrainConfig:
    """Complete EnzFormer2-Act training configuration."""

    data: ActDataConfig = field(default_factory=ActDataConfig)
    model: ActModelConfig = field(default_factory=ActModelConfig)
    masking: ActMaskingConfig = field(default_factory=ActMaskingConfig)
    loss: ActLossConfig = field(default_factory=ActLossConfig)
    runtime: ActRuntimeConfig = field(default_factory=ActRuntimeConfig)
    checkpoint: ActCheckpointConfig = field(default_factory=ActCheckpointConfig)


@dataclass(slots=True)
class ActScoreConfig:
    """Complete EnzFormer2-Act scoring configuration."""

    data: ActDataConfig = field(default_factory=ActDataConfig)
    runtime: ActRuntimeConfig = field(default_factory=lambda: ActRuntimeConfig(batch_size=1))
    checkpoint: ActCheckpointConfig = field(default_factory=ActCheckpointConfig)
    scan: ActScanConfig = field(default_factory=ActScanConfig)


__all__ = [
    "PRECISIONS",
    "LORA_TARGET_MODULES",
    "WEIGHT_MODES",
    "WEIGHT_NORMALIZATIONS",
    "ActCheckpointConfig",
    "ActDataConfig",
    "ActLossConfig",
    "ActMaskingConfig",
    "ActModelConfig",
    "ActRuntimeConfig",
    "ActScanConfig",
    "ActScoreConfig",
    "ActTrainConfig",
]
