"""Train the EnzFormer2-Act masked-language LoRA adapter."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from enzformer2.config import (
    ActTrainConfig,
    act_train_config_to_namespace,
    config_to_serializable_dict,
    load_act_train_config,
    validate_act_train_config,
)
from enzformer2.data.collate import collate_raw_sequence_batch
from enzformer2.data.raw_sequence_dataset import RawSequenceDataset
from enzformer2.data.sampling import TargetPositionSampler
from enzformer2.models.conditional_masked_scorer import ESMCMaskedConditionalScorer
from enzformer2.models.esmc_backbone import ESMCBackboneWrapper
from enzformer2.act.training_common import (
    build_loader_kwargs,
    configure_runtime_logging,
    get_best_metric_from_checkpoint,
    load_model_state_into_scorer,
    load_trusted_checkpoint,
    log_line,
    model_state_for_checkpoint,
    namespace_to_serializable_dict,
    resolve_device,
    save_epoch_family_checkpoints,
    write_history_json,
)
from enzformer2.utils import set_global_seed


configure_runtime_logging()

SELECTION_METRIC_NAME = "val_target_ce"
COMPLETION_MARKER_NAME = "training_complete.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="enzformer2 act train",
        description="Train an EnzFormer2-Act LoRA adapter from an explicit YAML configuration.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate YAML and installed input metadata without loading the ESM-C model.",
    )
    return parser.parse_args(argv)


def resolve_act_args(cli_args: argparse.Namespace) -> tuple[argparse.Namespace, ActTrainConfig]:
    config = load_act_train_config(cli_args.config)
    args = act_train_config_to_namespace(config)
    args.config = Path(cli_args.config).expanduser().resolve()
    if cli_args.device is not None:
        args.device = str(cli_args.device)
    if cli_args.output_dir is not None:
        output_dir = Path(cli_args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = _REPO_ROOT / output_dir
        args.output_dir = output_dir.resolve()
    return args, config


def _resolve_output_dir(value: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _split_indices(count: int, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if count <= 0:
        raise ValueError("The Act sequence corpus contains no sequences.")
    if fraction <= 0.0:
        return list(range(count)), []
    if count < 2:
        raise ValueError("At least two sequences are required for a non-empty validation split.")
    validation_count = max(1, int(round(count * fraction)))
    validation_count = min(validation_count, count - 1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(count, generator=generator).tolist()
    return order[validation_count:], order[:validation_count]


def build_corpus_loaders(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    """Load and split the frozen sequence manifest, failing on empty inputs."""

    dataset = RawSequenceDataset(
        args.manifest_path,
        sequence_root=args.sequence_root,
        sequence_column=args.sequence_column,
        seq_id_column=args.seq_id_column,
        sequence_lookup_path=args.sequence_lookup_path,
    )
    if args.weight_mode != "uniform":
        raise ValueError("This release supports only data.weight_mode='uniform' (qwoff).")
    for record in dataset.records:
        record["sample_weight"] = 1.0

    weight_values = torch.tensor(
        [float(record["sample_weight"]) for record in dataset.records], dtype=torch.float64
    )

    train_indices, validation_indices = _split_indices(
        len(dataset), float(args.validation_fraction), int(args.seed)
    )
    if not train_indices:
        raise ValueError("The Act training split is empty.")
    if args.validation_fraction > 0.0 and not validation_indices:
        raise ValueError("The Act validation split is empty.")
    loader_options = build_loader_kwargs(
        num_workers=int(args.num_workers),
        device=device,
        collate_fn=collate_raw_sequence_batch,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(args.seed)),
        **loader_options,
    )
    validation_loader = None
    if validation_indices:
        validation_loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=int(args.batch_size),
            shuffle=False,
            **loader_options,
        )
    summary = {
        "total_sequences": len(dataset),
        "train_sequences": len(train_indices),
        "validation_sequences": len(validation_indices),
        "weight_mode": args.weight_mode,
        "weight_distribution": {
            "count": int(weight_values.numel()),
            "min": float(weight_values.min().item()),
            "max": float(weight_values.max().item()),
            "mean": float(weight_values.mean().item()),
            "std": float(weight_values.std(unbiased=False).item()),
        },
    }
    return train_loader, validation_loader, summary


def weighted_target_loss(
    target_log_probs: torch.Tensor,
    wt_token_ids: torch.Tensor,
    target_valid_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    normalization: str,
) -> torch.Tensor:
    """Compute weighted target NLL with explicit reduction semantics."""

    if normalization not in {"batch_sum", "token_mean"}:
        raise ValueError(f"Unknown sample-weight normalization: {normalization!r}")
    if target_log_probs.ndim != 3 or wt_token_ids.ndim != 2:
        raise ValueError("target_log_probs must be [B, M, V] and wt_token_ids must be [B, M].")
    if wt_token_ids.shape != target_valid_mask.shape:
        raise ValueError("WT token ids and target-valid mask must have matching shapes.")
    per_target = -target_log_probs.gather(-1, wt_token_ids.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    valid = target_valid_mask.to(dtype=per_target.dtype)
    weights = sample_weights.to(device=per_target.device, dtype=per_target.dtype).view(-1, 1)
    weighted = per_target * valid * weights
    if normalization == "batch_sum":
        denominator = (valid * weights).sum()
    else:
        denominator = valid.sum()
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("No valid target residues are available for the loss.")
    return weighted.sum() / denominator


def _autocast_context(device: torch.device, precision: str):
    if precision in {"none", "fp32"}:
        return contextlib.nullcontext()
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("runtime.amp_dtype='fp16' requires CUDA; use fp32 on CPU.")
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported precision: {precision!r}")


def _forward_loss(
    model: ESMCMaskedConditionalScorer,
    batch: Mapping[str, Any],
    sampler: TargetPositionSampler,
    *,
    args: argparse.Namespace,
    training: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid_mask = batch["valid_mask"]
    target_sample = sampler.sample_with_metadata(
        valid_mask,
        seq_ids=batch["seq_id"],
        deterministic=not training,
    )
    target_positions = target_sample.positions
    target_valid_mask = target_sample.target_valid_mask
    model_batch = {
        **batch,
        "target_positions": target_positions,
        "target_valid_mask": target_valid_mask,
        "context_mask_policy": args.context_mask_policy if training else "target_only",
        "context_mask_radius": args.context_mask_radius if training else 0,
        "extra_random_mask_rate": args.extra_random_mask_rate if training else 0.0,
        "share_context_across_targets": (
            bool(args.share_context_across_targets) if training else False
        ),
        "backbone_max_masked_rows_per_call": args.backbone_max_masked_rows_per_call,
    }
    with _autocast_context(device, str(args.amp_dtype)):
        outputs = model(model_batch)
        log_probs = outputs["common_target_log_probs"]
        wt_ids = outputs["target_wt_token_ids"].to(device=log_probs.device)
        valid_targets = outputs.get("target_valid_mask", target_valid_mask).to(
            device=log_probs.device
        )
        loss = weighted_target_loss(
            log_probs,
            wt_ids,
            valid_targets,
            batch["sample_weights"],
            normalization=str(args.sample_weight_normalization),
        )
    with torch.no_grad():
        valid = valid_targets.bool()
        target_nll = -log_probs.gather(-1, wt_ids.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        target_nll = target_nll.masked_select(valid)
        top1 = log_probs.argmax(dim=-1).eq(wt_ids).masked_select(valid).float()
    return loss, {
        "target_ce": float(target_nll.mean().item()),
        "target_top1": float(top1.mean().item()),
        "num_targets": float(valid.sum().item()),
        "masked_positions": float(
            outputs.get("total_masked_positions_per_seq", torch.zeros(1)).float().mean().item()
        ),
    }


def run_epoch(
    model: ESMCMaskedConditionalScorer,
    loader: DataLoader,
    sampler: TargetPositionSampler,
    *,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {
        "target_ce": 0.0,
        "target_top1": 0.0,
        "num_targets": 0.0,
        "masked_positions": 0.0,
    }
    batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        loss_context = contextlib.nullcontext() if training else torch.inference_mode()
        with loss_context:
            loss, metrics = _forward_loss(
                model, batch, sampler, args=args, training=training, device=device
            )
        if training:
            scaled_loss = loss / float(args.grad_accum_steps)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if (batch_index + 1) % int(args.grad_accum_steps) == 0 or batch_index + 1 == len(
                loader
            ):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if args.grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        batches += 1
        for key, value in metrics.items():
            totals[key] += value
    if batches == 0:
        raise ValueError("The DataLoader produced no batches.")
    totals["target_ce"] /= batches
    totals["target_top1"] /= batches
    totals["num_targets"] /= batches
    totals["masked_positions"] /= batches
    totals["total_loss"] = totals["target_ce"]
    return totals


def _build_optimizer(model: ESMCMaskedConditionalScorer, args: argparse.Namespace) -> AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable LoRA parameters were found.")
    return AdamW(
        parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=tuple(float(value) for value in args.betas),
        eps=float(args.eps),
    )


def _build_scheduler(
    optimizer: AdamW,
    *,
    name: str,
    warmup_ratio: float,
    total_steps: int,
) -> LambdaLR | None:
    if name == "none":
        return None
    warmup_steps = int(round(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if name == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return LambdaLR(optimizer, scale)


def _load_initial_state(
    path: Path | None, model: ESMCMaskedConditionalScorer, device: torch.device
) -> None:
    if path is None:
        return
    checkpoint = load_trusted_checkpoint(path, device=device)
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint has no model_state mapping: {path}")
    load_model_state_into_scorer(
        model, state, state_dict_mode=str(checkpoint.get("state_dict_mode", "trainable_only"))
    )


def _write_completion_marker(
    output_dir: Path, best_epoch: int, best_metric: float, history: list[dict[str, Any]]
) -> None:
    (output_dir / COMPLETION_MARKER_NAME).write_text(
        json.dumps(
            {
                "status": "completed",
                "best_epoch": int(best_epoch),
                "best_val_target_ce": float(best_metric),
                "history_length": len(history),
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    cli_args = parse_args(argv)
    args, config = resolve_act_args(cli_args)
    if cli_args.validate_config:
        summary = validate_act_train_config(config, require_assets=False)
        print(json.dumps({"status": "valid", **summary}, indent=2))
        return
    summary = validate_act_train_config(config, require_assets=True)
    if args.resume_ckpt is not None and args.init_ckpt is not None:
        raise ValueError("Use either checkpoint.resume_ckpt or checkpoint.init_ckpt, not both.")
    device = resolve_device(args.device)
    if args.amp_dtype == "fp16" and device.type != "cuda":
        raise ValueError("runtime.amp_dtype='fp16' requires CUDA; use fp32 on CPU.")
    set_global_seed(int(args.seed))
    args.output_dir = _resolve_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, split_summary = build_corpus_loaders(args, device=device)
    backbone = ESMCBackboneWrapper.from_pretrained(args.backbone_model_name, device=device)
    model = ESMCMaskedConditionalScorer(
        backbone_model_name=args.backbone_model_name,
        freeze_backbone=bool(args.freeze_backbone),
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_target_modules=tuple(args.lora_target_modules),
        backbone=backbone,
    ).to(device)
    _load_initial_state(args.init_ckpt, model, device)
    optimizer = _build_optimizer(model, args)
    scheduler = _build_scheduler(
        optimizer,
        name=str(args.scheduler),
        warmup_ratio=float(args.warmup_ratio),
        total_steps=max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs),
    )
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if device.type == "cuda" and args.amp_dtype == "fp16"
        else None
    )
    sampler = TargetPositionSampler(
        int(args.targets_per_sequence),
        training=True,
        base_seed=int(args.seed),
    )
    validation_sampler = TargetPositionSampler(
        int(args.targets_per_sequence),
        training=False,
        base_seed=int(args.seed),
    )
    history_path = args.output_dir / "history.json"
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = -1
    start_epoch = 0
    if args.resume_ckpt is not None:
        checkpoint = load_trusted_checkpoint(args.resume_ckpt, device=device)
        state = checkpoint.get("model_state")
        if not isinstance(state, Mapping):
            raise ValueError("Resume checkpoint lacks model_state.")
        load_model_state_into_scorer(model, state)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_metric, _ = get_best_metric_from_checkpoint(
            checkpoint, default_metric_name=SELECTION_METRIC_NAME
        )
        best_epoch = int(checkpoint.get("best_epoch", -1))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    log_line(
        f"act-train device={device} precision_requested={args.amp_dtype} "
        f"precision_resolved={args.amp_dtype if args.amp_dtype in {'none', 'fp32'} else args.amp_dtype} "
        f"train_sequences={split_summary['train_sequences']} "
        f"validation_sequences={split_summary['validation_sequences']}"
    )
    for epoch in range(start_epoch, int(args.epochs)):
        train_metrics = run_epoch(
            model,
            train_loader,
            sampler,
            args=args,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        validation_metrics = (
            run_epoch(model, validation_loader, validation_sampler, args=args, device=device)
            if validation_loader is not None
            else train_metrics
        )
        current_metric = float(validation_metrics["target_ce"])
        improved = True
        if improved:
            best_metric = current_metric
            best_epoch = epoch
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "best_epoch": best_epoch,
            "best_val_target_ce": best_metric,
        }
        history.append(row)
        write_history_json(history_path, history)
        log_line(
            f"epoch={epoch} train_target_ce={train_metrics['target_ce']:.4f} "
            f"validation_target_ce={validation_metrics['target_ce']:.4f} "
            f"validation_target_top1={validation_metrics['target_top1']:.4f}"
        )
        state_dict, state_mode = model_state_for_checkpoint(model)
        checkpoint_kwargs = {
            "workflow": "act_train",
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "best_val_metric": best_metric,
            "best_metric_name": SELECTION_METRIC_NAME,
            "best_epoch": best_epoch,
            "epochs_without_improvement": 0,
            "model_config": config_to_serializable_dict(config.model),
            "data_config": config_to_serializable_dict(config.data),
            "runtime_config": {
                **config_to_serializable_dict(config.runtime),
                "device_resolved": str(device),
                "precision_requested": args.amp_dtype,
                "precision_resolved": args.amp_dtype,
            },
            "masking_config": config_to_serializable_dict(config.masking),
            "checkpoint_config": config_to_serializable_dict(config.checkpoint),
            "loss_config": config_to_serializable_dict(config.loss),
            "extra_state": {
                "trainable_param_names": [
                    name for name, parameter in model.named_parameters() if parameter.requires_grad
                ],
                "injected_lora_modules": list(model.injected_lora_module_names()),
                "state_dict_mode": state_mode,
                "score_column": "masked_log_odds",
                "split_summary": split_summary,
                "config_validation": summary,
            },
            "args": namespace_to_serializable_dict(args),
            "model_state_dict": state_dict,
        }
        if bool(args.keep_per_epoch_checkpoints):
            save_epoch_family_checkpoints(
                args.output_dir, epoch=epoch, improved=improved, checkpoint_kwargs=checkpoint_kwargs
            )
        elif improved:
            from enzformer2.act.training_common import save_training_checkpoint

            save_training_checkpoint(args.output_dir / "best.pt", **checkpoint_kwargs)
    write_history_json(history_path, history)
    _write_completion_marker(args.output_dir, best_epoch, best_metric, history)


if __name__ == "__main__":  # pragma: no cover
    main()
