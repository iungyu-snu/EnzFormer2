"""YAML loading and validation for the public EnzFormer2-Act commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .schemas import (
    PRECISIONS,
    ActCheckpointConfig,
    ActDataConfig,
    ActLossConfig,
    ActMaskingConfig,
    ActModelConfig,
    ActRuntimeConfig,
    ActScanConfig,
    ActScoreConfig,
    ActTrainConfig,
)


_TRAIN_KEYS = frozenset({"data", "model", "masking", "loss", "runtime", "checkpoint"})
_SCORE_KEYS = frozenset({"data", "runtime", "checkpoint", "scan"})
_DATA_KEYS = frozenset(
    {
        "manifest_path",
        "sequence_root",
        "sequence_column",
        "seq_id_column",
        "sequence_lookup_path",
        "weight_mode",
        "asset_status",
    }
)
_MODEL_KEYS = frozenset(
    {
        "backbone_model_name",
        "freeze_backbone",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
    }
)
_MASKING_KEYS = frozenset(
    {
        "targets_per_sequence",
        "context_mask_policy",
        "context_mask_radius",
        "extra_random_mask_rate",
        "share_context_across_targets",
    }
)
_LOSS_KEYS = frozenset({"sample_weight_normalization"})
_RUNTIME_KEYS = frozenset(
    {
        "batch_size",
        "num_workers",
        "epochs",
        "validation_fraction",
        "optimizer",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "grad_clip_norm",
        "grad_accum_steps",
        "scheduler",
        "warmup_ratio",
        "amp_dtype",
        "seed",
        "device",
        "pin_memory",
        "persistent_workers",
        "backbone_max_masked_rows_per_call",
        "keep_per_epoch_checkpoints",
    }
)
_CHECKPOINT_KEYS = frozenset({"output_dir", "init_ckpt", "resume_ckpt", "checkpoint"})
_SCAN_KEYS = frozenset({"chunk_size", "seq_id", "input_manifest", "output"})
_ASSET_STATUSES = frozenset(
    {"included", "external", "separate_archive", "restricted", "not_redistributed"}
)


def _config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _path(value: Any, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _load_yaml_mapping(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = _config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("The top-level configuration must be a mapping.")
    return dict(payload), config_path.parent


def _section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {name!r} must be a mapping.")
    return dict(value)


def _assert_keys(section: str, payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unexpected = sorted(set(payload).difference(allowed))
    if unexpected:
        raise ValueError(f"Unsupported keys in {section}: {unexpected}.")


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean; got {value!r}.")


def _data(payload: Mapping[str, Any], *, base_dir: Path) -> ActDataConfig:
    _assert_keys("data", payload, _DATA_KEYS)
    asset_status = str(payload.get("asset_status", "included"))
    if asset_status not in _ASSET_STATUSES:
        raise ValueError(f"data.asset_status must be one of {sorted(_ASSET_STATUSES)}.")
    result = ActDataConfig(
        manifest_path=_path(payload.get("manifest_path"), base_dir=base_dir),
        sequence_root=_path(payload.get("sequence_root"), base_dir=base_dir),
        sequence_column=None
        if payload.get("sequence_column") is None
        else str(payload["sequence_column"]),
        seq_id_column=None
        if payload.get("seq_id_column") is None
        else str(payload["seq_id_column"]),
        sequence_lookup_path=_path(payload.get("sequence_lookup_path"), base_dir=base_dir),
        weight_mode=str(payload.get("weight_mode", "uniform")),
        asset_status=asset_status,
    )
    return result


def _model(payload: Mapping[str, Any]) -> ActModelConfig:
    _assert_keys("model", payload, _MODEL_KEYS)
    return ActModelConfig(
        backbone_model_name=str(payload.get("backbone_model_name", "esmc_600m")),
        freeze_backbone=_bool(payload.get("freeze_backbone", True), "model.freeze_backbone"),
        lora_rank=int(payload.get("lora_rank", 32)),
        lora_alpha=int(payload.get("lora_alpha", 32)),
        lora_dropout=float(payload.get("lora_dropout", 0.05)),
        lora_target_modules=list(payload.get("lora_target_modules", ["layernorm_qkv", "out_proj"])),
    )


def _masking(payload: Mapping[str, Any]) -> ActMaskingConfig:
    _assert_keys("masking", payload, _MASKING_KEYS)
    return ActMaskingConfig(
        targets_per_sequence=int(payload.get("targets_per_sequence", 4)),
        context_mask_policy=str(payload.get("context_mask_policy", "radius_plus_random")),
        context_mask_radius=int(payload.get("context_mask_radius", 0)),
        extra_random_mask_rate=float(payload.get("extra_random_mask_rate", 0.05)),
        share_context_across_targets=_bool(
            payload.get("share_context_across_targets", True),
            "masking.share_context_across_targets",
        ),
    )


def _loss(payload: Mapping[str, Any]) -> ActLossConfig:
    _assert_keys("loss", payload, _LOSS_KEYS)
    return ActLossConfig(
        sample_weight_normalization=str(payload.get("sample_weight_normalization", "token_mean"))
    )


def _runtime(payload: Mapping[str, Any], *, batch_default: int) -> ActRuntimeConfig:
    _assert_keys("runtime", payload, _RUNTIME_KEYS)
    values = dict(payload)
    values.setdefault("batch_size", batch_default)
    if "pin_memory" in values:
        values["pin_memory"] = _bool(values["pin_memory"], "runtime.pin_memory")
    if "persistent_workers" in values:
        values["persistent_workers"] = _bool(
            values["persistent_workers"], "runtime.persistent_workers"
        )
    if "keep_per_epoch_checkpoints" in values:
        values["keep_per_epoch_checkpoints"] = _bool(
            values["keep_per_epoch_checkpoints"], "runtime.keep_per_epoch_checkpoints"
        )
    if "amp_dtype" in values and str(values["amp_dtype"]) not in PRECISIONS:
        raise ValueError(f"runtime.amp_dtype must be one of {sorted(PRECISIONS)}.")
    return ActRuntimeConfig(**values)


def _checkpoint(payload: Mapping[str, Any], *, base_dir: Path) -> ActCheckpointConfig:
    _assert_keys("checkpoint", payload, _CHECKPOINT_KEYS)
    return ActCheckpointConfig(
        output_dir=_path(payload.get("output_dir"), base_dir=base_dir),
        init_ckpt=_path(payload.get("init_ckpt"), base_dir=base_dir),
        resume_ckpt=_path(payload.get("resume_ckpt"), base_dir=base_dir),
        checkpoint=_path(payload.get("checkpoint"), base_dir=base_dir),
    )


def _scan(payload: Mapping[str, Any], *, base_dir: Path) -> ActScanConfig:
    _assert_keys("scan", payload, _SCAN_KEYS)
    return ActScanConfig(
        chunk_size=int(payload.get("chunk_size", 64)),
        seq_id=None if payload.get("seq_id") is None else str(payload["seq_id"]),
        input_manifest=_path(payload.get("input_manifest"), base_dir=base_dir),
        output=_path(payload.get("output"), base_dir=base_dir),
    )


def load_act_train_config(path: str | Path) -> ActTrainConfig:
    """Load and type-check an Act training configuration."""

    payload, base_dir = _load_yaml_mapping(path)
    _assert_keys("top level", payload, _TRAIN_KEYS)
    config = ActTrainConfig(
        data=_data(_section(payload, "data"), base_dir=base_dir),
        model=_model(_section(payload, "model")),
        masking=_masking(_section(payload, "masking")),
        loss=_loss(_section(payload, "loss")),
        runtime=_runtime(_section(payload, "runtime"), batch_default=16),
        checkpoint=_checkpoint(_section(payload, "checkpoint"), base_dir=base_dir),
    )
    if config.checkpoint.output_dir is None:
        raise ValueError("checkpoint.output_dir is required for Act training.")
    return config


def load_act_score_config(path: str | Path) -> ActScoreConfig:
    """Load and type-check an Act scoring configuration."""

    payload, base_dir = _load_yaml_mapping(path)
    _assert_keys("top level", payload, _SCORE_KEYS)
    return ActScoreConfig(
        data=_data(_section(payload, "data"), base_dir=base_dir),
        runtime=_runtime(_section(payload, "runtime"), batch_default=1),
        checkpoint=_checkpoint(_section(payload, "checkpoint"), base_dir=base_dir),
        scan=_scan(_section(payload, "scan"), base_dir=base_dir),
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Act sequence manifests must be CSV or TSV: {path}")


def validate_act_train_config(
    config: ActTrainConfig, *, require_assets: bool = False
) -> dict[str, Any]:
    """Validate configuration and, when installed, the sequence-manifest contract.

    ``require_assets=False`` is intended for a no-model configuration check. A
    separately distributed paper corpus may therefore be absent while the
    configuration remains syntactically valid. Training always calls this
    function with ``require_assets=True``.
    """

    if config.data.manifest_path is None:
        raise ValueError("data.manifest_path is required for Act training.")

    manifest = config.data.manifest_path
    frame: pd.DataFrame | None = None
    sequence_column: str | None = config.data.sequence_column
    path_column_present = False
    if manifest.is_file():
        frame = _read_table(manifest)
        if frame.empty:
            raise ValueError(f"Act sequence manifest is empty: {manifest}")
        id_column = config.data.seq_id_column or next(
            (
                column
                for column in ("seq_id", "basename", "accession", "id")
                if column in frame.columns
            ),
            None,
        )
        if id_column is None:
            raise ValueError(
                "Act sequence manifest needs a seq_id, basename, accession, or id column."
            )
        ids = frame[id_column].astype(str).str.strip()
        if ids.eq("").any():
            raise ValueError("Act sequence manifest contains an empty sequence identifier.")
        if ids.duplicated().any():
            duplicates = sorted(set(ids[ids.duplicated()].tolist()))
            raise ValueError(
                f"Act sequence manifest contains duplicate identifiers: {duplicates[:5]}"
            )
        sequence_column = sequence_column or next(
            (column for column in ("sequence", "aa_sequence", "seq") if column in frame.columns),
            None,
        )
        path_column_present = any(
            column in frame.columns for column in ("sequence_path", "fasta_path", "file", "path")
        )
        if (
            sequence_column is None
            and not path_column_present
            and config.data.sequence_root is None
            and config.data.sequence_lookup_path is None
        ):
            raise ValueError(
                "Manifest must contain a sequence column/path or configure a sequence directory."
            )

    paths: list[Path] = [manifest]
    if frame is None or (sequence_column is None and not path_column_present):
        if config.data.sequence_lookup_path is not None:
            paths.append(config.data.sequence_lookup_path)
        elif config.data.sequence_root is not None:
            paths.append(config.data.sequence_root)
    missing = [str(path) for path in paths if path is not None and not path.exists()]
    if missing and (
        require_assets or getattr(config.data, "asset_status", "included") == "included"
    ):
        raise FileNotFoundError("Required Act input is missing: " + ", ".join(missing))

    summary: dict[str, Any] = {"missing_paths": missing, "manifest_rows": None, "sequence_ids": []}
    if frame is not None:
        summary.update({"manifest_rows": int(len(frame)), "sequence_ids": ids.tolist()})
        if not missing or (sequence_column is not None and config.data.weight_mode == "uniform"):
            from enzformer2.data.raw_sequence_dataset import RawSequenceDataset

            RawSequenceDataset(
                manifest,
                sequence_root=config.data.sequence_root,
                sequence_column=sequence_column,
                seq_id_column=config.data.seq_id_column,
                sequence_lookup_path=config.data.sequence_lookup_path,
            )
    if config.checkpoint.output_dir is None:
        raise ValueError("checkpoint.output_dir is required for Act training.")
    if config.runtime.amp_dtype not in PRECISIONS:
        raise ValueError(f"Unsupported precision: {config.runtime.amp_dtype}")
    return summary


def act_train_config_to_namespace(config: ActTrainConfig) -> argparse.Namespace:
    """Expose a small namespace for the training loop and run metadata."""

    values: dict[str, Any] = {
        "manifest_path": config.data.manifest_path,
        "sequence_root": config.data.sequence_root,
        "sequence_column": config.data.sequence_column,
        "seq_id_column": config.data.seq_id_column,
        "sequence_lookup_path": config.data.sequence_lookup_path,
        "weight_mode": config.data.weight_mode,
        "backbone_model_name": config.model.backbone_model_name,
        "freeze_backbone": config.model.freeze_backbone,
        "lora_rank": config.model.lora_rank,
        "lora_alpha": config.model.lora_alpha,
        "lora_dropout": config.model.lora_dropout,
        "lora_target_modules": list(config.model.lora_target_modules),
        "targets_per_sequence": config.masking.targets_per_sequence,
        "context_mask_policy": config.masking.context_mask_policy,
        "context_mask_radius": config.masking.context_mask_radius,
        "extra_random_mask_rate": config.masking.extra_random_mask_rate,
        "share_context_across_targets": config.masking.share_context_across_targets,
        "sample_weight_normalization": config.loss.sample_weight_normalization,
        "output_dir": config.checkpoint.output_dir,
        "init_ckpt": config.checkpoint.init_ckpt,
        "resume_ckpt": config.checkpoint.resume_ckpt,
    }
    values.update(asdict(config.runtime))
    values["act_config"] = config
    return argparse.Namespace(**values)


def act_score_config_to_namespace(config: ActScoreConfig) -> argparse.Namespace:
    """Expose scoring settings to the scanner."""

    values: dict[str, Any] = {
        "manifest_path": config.data.manifest_path,
        "sequence_root": config.data.sequence_root,
        "sequence_column": config.data.sequence_column,
        "seq_id_column": config.data.seq_id_column,
        "sequence_lookup_path": config.data.sequence_lookup_path,
        "weight_mode": config.data.weight_mode,
        "checkpoint": config.checkpoint.checkpoint,
        "input_manifest": config.scan.input_manifest,
        "seq_id": config.scan.seq_id,
        "output": config.scan.output,
        "chunk_size": config.scan.chunk_size,
    }
    values.update(asdict(config.runtime))
    return argparse.Namespace(**values)


def config_to_serializable_dict(config: Any) -> dict[str, Any]:
    """Convert a dataclass configuration to JSON-safe primitive values."""

    if not is_dataclass(config):
        raise TypeError("config_to_serializable_dict expects a dataclass instance.")

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return {key: convert(item) for key, item in asdict(value).items()}
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(config)


__all__ = [
    "act_score_config_to_namespace",
    "act_train_config_to_namespace",
    "config_to_serializable_dict",
    "load_act_score_config",
    "load_act_train_config",
    "validate_act_train_config",
]
