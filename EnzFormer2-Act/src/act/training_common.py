"""Shared runtime, checkpoint, and logging helpers for Act workflows."""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.optim import Optimizer


def configure_runtime_logging(level: int = logging.INFO) -> None:
    """Configure one concise foreground logger when the application starts."""

    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def log_line(message: str) -> None:
    logging.getLogger("enzformer2").info(message)


def resolve_device(requested: str | None) -> torch.device:
    """Resolve a requested device without selecting unavailable hardware."""

    if requested is None or str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_loader_kwargs(
    *,
    num_workers: int,
    device: torch.device,
    collate_fn: Any,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> dict[str, Any]:
    """Build portable DataLoader options."""

    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0.")
    return {
        "num_workers": int(num_workers),
        "collate_fn": collate_fn,
        "pin_memory": bool(pin_memory and device.type == "cuda"),
        **({"persistent_workers": bool(persistent_workers)} if num_workers > 0 else {}),
    }


def load_trusted_checkpoint(path: str | Path, *, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint explicitly supplied by the user."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def get_best_metric_from_checkpoint(
    checkpoint: Mapping[str, Any], *, default_metric_name: str
) -> tuple[float, str]:
    """Read the best validation metric stored in a checkpoint."""

    if "best_val_metric" in checkpoint:
        return float(checkpoint["best_val_metric"]), str(
            checkpoint.get("best_metric_name", default_metric_name)
        )
    if "best_val_ce" in checkpoint:
        return float(checkpoint["best_val_ce"]), "val_target_ce"
    return float("inf"), default_metric_name


def model_state_for_checkpoint(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], str]:
    """Return the compact adapter state used by Act checkpoints."""

    state = model.state_dict()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    compact = {name: tensor for name, tensor in state.items() if name in trainable}
    return compact, "trainable_only"


def load_model_state_into_scorer(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    state_dict_mode: str = "trainable_only",
) -> None:
    """Load an adapter checkpoint and reject unexpected or missing parameters."""

    if state_dict_mode != "trainable_only":
        raise ValueError("Act checkpoints use state_dict_mode='trainable_only'.")
    model_keys = set(model.state_dict())
    unknown = sorted(set(state_dict).difference(model_keys))
    if unknown:
        raise ValueError(f"Checkpoint contains unexpected parameters: {unknown}")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = sorted(trainable.difference(state_dict))
    if missing:
        raise ValueError(f"Checkpoint is missing trainable parameters: {missing}")
    model.load_state_dict(dict(state_dict), strict=False)


def save_training_checkpoint(
    path: Path,
    *,
    workflow: str,
    model: torch.nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_val_metric: float,
    best_metric_name: str,
    best_epoch: int,
    epochs_without_improvement: int,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any] | None,
    runtime_config: Mapping[str, Any] | None,
    masking_config: Mapping[str, Any] | None,
    checkpoint_config: Mapping[str, Any] | None,
    loss_config: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
    args: Mapping[str, Any] | None = None,
    model_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Write a self-describing compact Act training checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict() if model_state_dict is None else dict(model_state_dict)
    payload: dict[str, Any] = {
        "workflow": str(workflow),
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_metric": float(best_val_metric),
        "best_metric_name": str(best_metric_name),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "model_config": dict(model_config),
        "data_config": None if data_config is None else dict(data_config),
        "runtime_config": None if runtime_config is None else dict(runtime_config),
        "masking_config": None if masking_config is None else dict(masking_config),
        "checkpoint_config": None if checkpoint_config is None else dict(checkpoint_config),
    }
    if loss_config is not None:
        payload["loss_config"] = dict(loss_config)
    if extra_state is not None:
        payload["extra_state"] = dict(extra_state)
    if args is not None:
        payload["args"] = dict(args)
    torch.save(payload, path)


def save_epoch_family_checkpoints(
    output_dir: Path,
    *,
    epoch: int,
    improved: bool,
    checkpoint_kwargs: Mapping[str, Any],
) -> None:
    """Save an epoch checkpoint and update last/best aliases."""

    save_training_checkpoint(output_dir / f"epoch_{int(epoch):04d}.pt", **dict(checkpoint_kwargs))
    if improved:
        save_training_checkpoint(output_dir / "best.pt", **dict(checkpoint_kwargs))
    save_training_checkpoint(output_dir / "last.pt", **dict(checkpoint_kwargs))


def write_history_json(path: Path, history: list[dict[str, Any]]) -> None:
    """Write metric history as indented JSON."""

    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def namespace_to_serializable_dict(namespace: Namespace | Mapping[str, Any]) -> dict[str, Any]:
    """Convert a namespace to JSON-safe values."""

    values = dict(vars(namespace)) if isinstance(namespace, Namespace) else dict(namespace)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    values.pop("act_config", None)
    return {key: convert(value) for key, value in values.items()}


def compute_benchmark_mean(*, spearman_rho: float | None, ndcg_mean: float | None) -> float | None:
    """Combine normalized Spearman and NDCG for the optional eCTL hook."""

    if spearman_rho is None or ndcg_mean is None:
        return None
    return (((float(spearman_rho) + 1.0) / 2.0) + float(ndcg_mean)) / 2.0


__all__ = [
    "build_loader_kwargs",
    "compute_benchmark_mean",
    "configure_runtime_logging",
    "get_best_metric_from_checkpoint",
    "load_model_state_into_scorer",
    "load_trusted_checkpoint",
    "log_line",
    "model_state_for_checkpoint",
    "namespace_to_serializable_dict",
    "resolve_device",
    "save_epoch_family_checkpoints",
    "save_training_checkpoint",
    "write_history_json",
]
