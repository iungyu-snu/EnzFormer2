"""Regression trainer for ESMC+LoRA ddG regressors.

Design:

  * The frozen ESMC backbone is loaded once via :class:`ESMCMaskedLM`.
  * Custom LoRA is injected into the backbone in place; non-LoRA params are
    frozen with :func:`freeze_non_lora`.
  * A Siamese-delta regressor is built around the same wrapper.
  * The optimizer trains LoRA and regressor parameters, including the prior
    coefficient when enabled; full backbone weights are never written to disk.
  * Targets are optionally standardized using **train-only** statistics
    (``standardize_train``); predictions are inverse-transformed back to
    kcal/mol before being written to ``predictions_*.csv`` or ``metrics.json``.
  * Mixed precision is enabled only on CUDA. CPU runs fall back to fp32.
  * ESMC's public ``encode`` helper is single-sequence, but the wrapper uses
    batched token tensors for the regression hidden-state path, so
    ``batch_size`` controls the actual microbatch size.
  * The test split is never loaded by this trainer.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddg_lora.data.regression_dataset import RegressionDataset, RegressionExample
from ddg_lora.models.esmc_wrapper import ESMCMaskedLM
from ddg_lora.models.lora import (
    LoRAConfig,
    freeze_non_lora,
    inject_lora,
    save_lora_state_dict,
    trainable_param_report,
)
from ddg_lora.models.regressors import REGRESSOR_KINDS, build_regressor
from ddg_lora.utils.io import dump_json, dump_yaml, ensure_run_dir
from ddg_lora.utils.logging import get_logger
from ddg_lora.utils.seed import set_seed

log = get_logger("regression_trainer")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    log_every: int
    eval_every: int
    save_every: int
    loss: str = "smooth_l1"          # "smooth_l1" | "mse"
    smooth_l1_beta: float = 1.0
    target_transform: str = "standardize_train"  # "standardize_train" | "none"
    precision: str = "bf16"          # cuda autocast dtype


@dataclass
class TargetTransform:
    kind: str          # "standardize_train" | "none"
    mean: float = 0.0
    std: float = 1.0

    def encode(self, ddg: float) -> float:
        if self.kind == "none":
            return float(ddg)
        return float((ddg - self.mean) / max(self.std, 1e-12))

    def decode(self, std_value: float) -> float:
        if self.kind == "none":
            return float(std_value)
        return float(std_value * self.std + self.mean)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TargetTransform":
        return cls(
            kind=str(d.get("kind", "none")),
            mean=float(d.get("mean", 0.0)),
            std=float(d.get("std", 1.0)),
        )


@dataclass
class TrainerState:
    step: int = 0
    best_val_metric: float = float("inf")    # tracks lowest val MAE in kcal/mol
    metrics: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loss + AMP helpers
# ---------------------------------------------------------------------------


def _build_loss_fn(name: str, *, smooth_l1_beta: float):
    if name == "smooth_l1":
        return lambda pred, target: F.smooth_l1_loss(
            pred, target, beta=float(smooth_l1_beta)
        )
    if name == "mse":
        return F.mse_loss
    raise ValueError(f"unknown regression loss {name!r}")


def _amp_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", enabled=False)


def _predict_examples(model: nn.Module, examples: list[RegressionExample]) -> torch.Tensor:
    """Predict a microbatch, using a model's batched path when available."""
    if not examples:
        raise ValueError("examples must be non-empty")
    predict_batch = getattr(model, "predict_batch", None)
    if callable(predict_batch):
        pred = predict_batch(examples)
    else:
        pred = torch.stack([model.predict_one(ex) for ex in examples])
    if pred.dim() == 0:
        pred = pred.unsqueeze(0)
    return pred


def _iter_dataset_batches(
    dataset: RegressionDataset, batch_size: int
) -> list[RegressionExample]:
    """Yield dataset examples in contiguous microbatches for evaluation."""
    batch_size = max(1, int(batch_size))
    batch: list[RegressionExample] = []
    for ex in dataset.iter_examples():
        batch.append(ex)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Target standardization
# ---------------------------------------------------------------------------


def fit_target_transform(
    train_dataset: RegressionDataset, kind: str
) -> TargetTransform:
    if kind == "none":
        return TargetTransform(kind="none", mean=0.0, std=1.0)
    if kind != "standardize_train":
        raise ValueError(f"unknown target_transform {kind!r}")
    df = train_dataset.dataframe
    if df.empty:
        log.warning("training set is empty; falling back to identity target transform")
        return TargetTransform(kind="standardize_train", mean=0.0, std=1.0)
    arr = df["ddg"].astype(float).to_numpy()
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-6:
        log.warning("training ddg std=%.3e is near zero; using std=1.0 instead", std)
        std = 1.0
    log.info("target_transform=standardize_train mean=%.4f std=%.4f", mean, std)
    return TargetTransform(kind="standardize_train", mean=mean, std=std)


def _masked_logodds_feature_values(
    df,
    *,
    mode: str,
) -> np.ndarray:
    required = ["masked_logodds_forward", "masked_logodds_reverse"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "masked_logodds is enabled but the training parquet lacks "
            f"precomputed columns: {missing}. Run scripts/precompute_esmc_masked_logodds.py "
            "and point data.train_path/validation_path at the enriched parquets."
        )
    forward = df["masked_logodds_forward"].astype(float).to_numpy()
    reverse = df["masked_logodds_reverse"].astype(float).to_numpy()
    finite = np.isfinite(forward) & np.isfinite(reverse)
    if not finite.all():
        raise ValueError(
            f"masked_logodds columns contain {(~finite).sum()} non-finite rows"
        )
    if mode == "wt_context":
        return forward.reshape(-1, 1)
    if mode == "symmetric":
        return (0.5 * (forward - reverse)).reshape(-1, 1)
    if mode == "both":
        return np.stack([forward, reverse], axis=1)
    raise ValueError(f"unknown masked_logodds mode {mode!r}")


def fit_masked_logodds_stats(
    train_dataset: RegressionDataset,
    regressor_cfg: dict[str, Any],
) -> tuple[list[float], list[float]] | None:
    raw = regressor_cfg.get("masked_logodds") or {}
    if not bool(raw.get("enabled", False)):
        return None
    mode = str(raw.get("mode", "wt_context"))
    transform = str(raw.get("transform", "train_zscore"))
    values = _masked_logodds_feature_values(train_dataset.dataframe, mode=mode)
    if transform in {"raw", "none"}:
        mean = np.zeros(values.shape[1], dtype=float)
        std = np.ones(values.shape[1], dtype=float)
    elif transform in {"train_zscore", "train_zscore_clip5"}:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    else:
        raise ValueError(f"unknown masked_logodds transform {transform!r}")
    out = ([float(x) for x in mean], [float(x) for x in std])
    log.info("masked_logodds stats mode=%s mean=%s std=%s", mode, out[0], out[1])
    return out


def _masked_logodds_snapshot(model: nn.Module) -> dict[str, Any] | None:
    inner = getattr(model, "inner", model)
    cfg = getattr(inner, "masked_logodds", None)
    if cfg is None:
        return None
    record = cfg.to_dict()
    if hasattr(inner, "masked_logodds_mean"):
        record["mean"] = [
            float(x) for x in inner.masked_logodds_mean.detach().cpu().reshape(-1)
        ]
        record["std"] = [
            float(x) for x in inner.masked_logodds_std.detach().cpu().reshape(-1)
        ]
    return record


def apply_cuda_memory_limit(model_cfg: dict[str, Any]) -> None:
    """Optionally cap this process's CUDA allocator from ``model`` config."""
    raw_limit = model_cfg.get("cuda_memory_limit_gb")
    if raw_limit is None:
        return
    if not torch.cuda.is_available():
        return

    device_spec = str(model_cfg.get("device", "auto"))
    device = torch.device("cuda" if device_spec == "auto" else device_spec)
    if device.type != "cuda":
        return

    limit_gb = float(raw_limit)
    if limit_gb <= 0:
        raise ValueError("model.cuda_memory_limit_gb must be positive")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
    fraction = min(1.0, max(0.01, limit_gb / total_gb))
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    log.info(
        "CUDA memory cap: %.1fGB requested on cuda:%d (%.1fGB total, fraction=%.3f)",
        limit_gb,
        device_index,
        total_gb,
        fraction,
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_regression(
    model: nn.Module,
    backbone_module: nn.Module,
    dataset: RegressionDataset,
    target_transform: TargetTransform,
    *,
    device: torch.device,
    precision: str,
    batch_size: int = 1,
) -> dict[str, float]:
    """Mean MAE / RMSE in kcal/mol on the given dataset."""
    if len(dataset) == 0:
        return {"val_mae": float("nan"), "val_rmse": float("nan"), "val_n": 0}

    was_training_backbone = backbone_module.training
    was_training_head = model.training
    backbone_module.eval()
    model.eval()
    try:
        abs_errors: list[float] = []
        sq_errors: list[float] = []
        for batch in _iter_dataset_batches(dataset, batch_size):
            with _amp_context(device, precision):
                pred_std = _predict_examples(model, batch)
            pred_np = pred_std.detach().float().cpu().numpy().reshape(-1)
            for pred_value, ex in zip(pred_np, batch, strict=True):
                pred_kcal = target_transform.decode(float(pred_value))
                err = pred_kcal - float(ex.ddg)
                abs_errors.append(abs(err))
                sq_errors.append(err * err)
        return {
            "val_mae": float(np.mean(abs_errors)),
            "val_rmse": float(math.sqrt(float(np.mean(sq_errors)))),
            "val_n": int(len(abs_errors)),
        }
    finally:
        if was_training_backbone:
            backbone_module.train()
        if was_training_head:
            model.train()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_one_regressor(
    *,
    wrapper: ESMCMaskedLM,
    model: nn.Module,
    train_dataset: RegressionDataset,
    val_dataset: RegressionDataset,
    target_transform: TargetTransform,
    train_cfg: TrainConfig,
    out_dir: str | Path,
    config_snapshot: dict | None = None,
) -> TrainerState:
    """End-to-end training loop for a single regressor.

    Caller is expected to have:
      - injected LoRA into ``wrapper.model`` and called :func:`freeze_non_lora`
      - constructed a Siamese-delta ``model`` wrapping ``wrapper``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_fp = open(metrics_path, "w", encoding="utf-8")

    if config_snapshot is not None:
        dump_yaml(config_snapshot, out_dir / "train_config.yaml")
    dump_json(target_transform.to_dict(), out_dir / "target_transform.json")
    masked_logodds_record = _masked_logodds_snapshot(model)
    if masked_logodds_record is not None:
        dump_json(masked_logodds_record, out_dir / "masked_logodds.json")

    device = wrapper.device
    backbone_module = wrapper.model

    param_rep = trainable_param_report(backbone_module)
    regressor_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Backbone trainable: %d / %d (%.4f%%) across %d LoRA modules; regressor trainable: %d",
        param_rep.trainable, param_rep.total,
        100.0 * param_rep.trainable_ratio, param_rep.lora_modules, regressor_trainable,
    )
    if param_rep.trainable == 0:
        raise RuntimeError(
            "backbone has no trainable LoRA parameters; check inject_lora and freeze_non_lora"
        )

    trainable_params = [p for p in backbone_module.parameters() if p.requires_grad]
    trainable_params += [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    loss_fn = _build_loss_fn(train_cfg.loss, smooth_l1_beta=train_cfg.smooth_l1_beta)
    state = TrainerState()

    n_train = len(train_dataset)
    log.info("train examples: %d, val examples: %d", n_train, len(val_dataset))
    log.info(
        "epochs=%d batch_size=%d grad_accum=%d effective_step_size=%d",
        train_cfg.epochs, train_cfg.batch_size, train_cfg.gradient_accumulation_steps,
        train_cfg.batch_size * train_cfg.gradient_accumulation_steps,
    )
    batch_size = max(1, int(train_cfg.batch_size))
    accum = max(1, train_cfg.gradient_accumulation_steps)

    for epoch in range(train_cfg.epochs):
        rng = torch.Generator(device="cpu").manual_seed(epoch + 1)
        perm = torch.randperm(n_train, generator=rng).tolist()

        backbone_module.train()
        model.train()
        epoch_loss_sum = 0.0
        epoch_main_loss_sum = 0.0
        epoch_examples = 0
        microbatch_inner = 0
        epoch_t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        for start in range(0, n_train, batch_size):
            batch_indices = perm[start : start + batch_size]
            examples = [train_dataset[idx] for idx in batch_indices]
            target_std = torch.tensor(
                [target_transform.encode(ex.ddg) for ex in examples],
                device=device,
                dtype=torch.float32,
            )

            with _amp_context(device, train_cfg.precision):
                pred_std = _predict_examples(model, examples)
                main_loss_raw = loss_fn(pred_std.float(), target_std)
                loss_total_raw = main_loss_raw
                loss = loss_total_raw / accum
            loss.backward()
            batch_loss = float(loss_total_raw.item())
            epoch_loss_sum += batch_loss * len(examples)
            epoch_main_loss_sum += float(main_loss_raw.item()) * len(examples)
            epoch_examples += len(examples)
            microbatch_inner += 1

            if microbatch_inner % accum == 0:
                if train_cfg.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, train_cfg.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.step += 1

                if state.step % train_cfg.log_every == 0:
                    avg = epoch_loss_sum / max(1, epoch_examples)
                    elapsed = time.time() - epoch_t0
                    rate = epoch_examples / elapsed if elapsed > 0 else 0.0
                    log.info(
                        "epoch %d step %d (%d/%d): train_loss=%.4f rate=%.2f ex/s",
                        epoch,
                        state.step,
                        min(start + len(examples), n_train),
                        n_train,
                        avg,
                        rate,
                    )

                if train_cfg.eval_every and state.step % train_cfg.eval_every == 0:
                    val = evaluate_regression(
                        model, backbone_module, val_dataset,
                        target_transform, device=device, precision=train_cfg.precision,
                        batch_size=batch_size,
                    )
                    train_regression_mean = (
                        epoch_main_loss_sum / max(1, epoch_examples)
                    )
                    rec = {
                        "epoch": epoch,
                        "step": state.step,
                        "train_loss": epoch_loss_sum / max(1, epoch_examples),
                        "train_loss_total": epoch_loss_sum / max(1, epoch_examples),
                        "train_loss_main": train_regression_mean,
                        "train_loss_regression": train_regression_mean,
                        **val,
                    }
                    state.metrics.append(rec)
                    metrics_fp.write(json.dumps(rec) + "\n")
                    metrics_fp.flush()
                    log.info("eval: %s", rec)
                    _maybe_save_best(
                        wrapper, model, out_dir, val.get("val_mae", float("nan")), state
                    )
                    backbone_module.train()
                    model.train()

                if train_cfg.save_every and state.step % train_cfg.save_every == 0:
                    save_lora_state_dict(
                        backbone_module, out_dir / f"adapter_step{state.step}.pt"
                    )
                    torch.save(
                        model.head.state_dict(), out_dir / f"head_step{state.step}.pt"
                    )

        # Flush any remaining gradient at epoch boundary.
        if microbatch_inner % accum != 0:
            if train_cfg.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, train_cfg.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state.step += 1

        val = evaluate_regression(
            model, backbone_module, val_dataset,
            target_transform, device=device, precision=train_cfg.precision,
            batch_size=batch_size,
        )
        epoch_regression_mean = epoch_main_loss_sum / max(1, epoch_examples)
        rec = {
            "epoch": epoch,
            "step": state.step,
            "epoch_train_loss": epoch_loss_sum / max(1, epoch_examples),
            "epoch_train_loss_total": epoch_loss_sum / max(1, epoch_examples),
            "epoch_train_loss_main": epoch_regression_mean,
            "epoch_train_loss_regression": epoch_regression_mean,
            **val,
        }
        state.metrics.append(rec)
        metrics_fp.write(json.dumps(rec) + "\n")
        metrics_fp.flush()
        log.info("epoch %d done: %s", epoch, rec)
        _maybe_save_best(
            wrapper, model, out_dir, val.get("val_mae", float("nan")), state
        )

    save_lora_state_dict(backbone_module, out_dir / "adapter.pt")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    torch.save(model.state_dict(), out_dir / "regressor.pt")
    metrics_fp.close()
    dump_json(
        {
            "steps": state.step,
            "best_val_mae_kcal": state.best_val_metric,
            "target_transform": target_transform.to_dict(),
            "masked_logodds": _masked_logodds_snapshot(model),
        },
        out_dir / "summary.json",
    )
    log.info(
        "Training done. best_val_mae=%.4f, total_steps=%d",
        state.best_val_metric, state.step,
    )
    return state


def _maybe_save_best(
    wrapper: ESMCMaskedLM,
    model: nn.Module,
    out_dir: Path,
    val_metric: float,
    state: TrainerState,
) -> None:
    if math.isnan(val_metric):
        return
    if val_metric < state.best_val_metric:
        state.best_val_metric = val_metric
        save_lora_state_dict(wrapper.model, out_dir / "best_adapter.pt")
        torch.save(model.head.state_dict(), out_dir / "best_head.pt")
        torch.save(model.state_dict(), out_dir / "best_regressor.pt")
        log.info("New best regressor saved (val_mae=%.4f)", val_metric)


# ---------------------------------------------------------------------------
# Config-driven entrypoint
# ---------------------------------------------------------------------------


def _build_dataset(
    parquet_path: str | Path,
    *,
    include_neutral: bool,
    max_examples: int | None,
    max_sequence_length: int | None,
) -> RegressionDataset:
    return RegressionDataset(
        parquet_path,
        include_neutral=include_neutral,
        max_examples=max_examples,
        max_sequence_length=max_sequence_length,
    )


def _infer_hidden_dim(wrapper: ESMCMaskedLM) -> int:
    """Probe the backbone with a tiny sequence to discover D."""
    probe_seq = "ACDEFGHIKL"
    with torch.inference_mode():
        H = wrapper.forward_residue_hidden_states(probe_seq)
    if H.dim() != 2 or H.shape[0] != len(probe_seq):
        raise RuntimeError(
            f"hidden-state probe returned shape {tuple(H.shape)}; "
            f"expected [{len(probe_seq)}, D]"
        )
    return int(H.shape[1])


def run_from_config(cfg: dict) -> TrainerState:
    """Build everything from a regression config and run training."""
    set_seed(cfg["experiment"]["seed"])

    # Backbone + LoRA setup.
    model_cfg = cfg["model"]
    apply_cuda_memory_limit(model_cfg)
    wrapper = ESMCMaskedLM.load(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
    )
    lcfg = cfg["lora"]
    target_patterns = tuple(
        lcfg.get("target_patterns") or lcfg.get("target_module_patterns") or []
    )
    inject_lora(
        wrapper.model,
        LoRAConfig(
            rank=int(lcfg["r"]),
            alpha=int(lcfg["alpha"]),
            dropout=float(lcfg.get("dropout", 0.0)),
            target_module_patterns=target_patterns,
            fail_if_no_match=bool(lcfg.get("fail_if_no_match", True)),
        ),
    )
    freeze_non_lora(wrapper.model)

    # Regressor.
    rcfg = cfg["regressor"]
    kind = rcfg["kind"]
    if kind not in REGRESSOR_KINDS:
        raise ValueError(f"regressor.kind={kind!r} not in {REGRESSOR_KINDS}")
    hidden_dim = _infer_hidden_dim(wrapper)
    log.info("backbone hidden_dim=D=%d", hidden_dim)
    model = build_regressor(
        kind,
        backbone=wrapper,
        hidden_dim=hidden_dim,
        head_dropout=float(rcfg.get("head_dropout", 0.1)),
        head_hidden_dims=tuple(rcfg.get("head_hidden_dims", (512, 128))),
        use_context=bool(rcfg.get("use_context", False)),
        context_window_k=int(rcfg.get("context_window_k", 5)),
        masked_logodds=rcfg.get("masked_logodds"),
    ).to(wrapper.device)

    # Datasets — train + validation only, never test.
    data_cfg = cfg["data"]
    if "test_path" in data_cfg:
        raise ValueError(
            "data.test_path is forbidden in train configs. "
            "The test split is loaded only by enzformer2-temp evaluate."
        )
    include_neutral = bool(data_cfg.get("include_neutral", True))
    max_train = data_cfg.get("max_train_examples")
    max_val = data_cfg.get("max_validation_examples")
    max_seq_len = data_cfg.get("max_sequence_length")

    train_ds = _build_dataset(
        data_cfg["train_path"],
        include_neutral=include_neutral,
        max_examples=max_train,
        max_sequence_length=max_seq_len,
    )
    val_ds = _build_dataset(
        data_cfg["validation_path"],
        include_neutral=include_neutral,
        max_examples=max_val,
        max_sequence_length=max_seq_len,
    )

    # Target transform is fit on the configured train slice only.
    t_cfg = cfg["training"]
    target_transform = fit_target_transform(
        train_ds, kind=str(t_cfg.get("target_transform", "standardize_train"))
    )
    masked_stats = fit_masked_logodds_stats(train_ds, rcfg)
    if masked_stats is not None:
        inner = getattr(model, "inner", model)
        inner.set_masked_logodds_stats(*masked_stats)

    train_cfg = TrainConfig(
        epochs=int(t_cfg["epochs"]),
        batch_size=int(t_cfg["batch_size"]),
        gradient_accumulation_steps=int(t_cfg["gradient_accumulation_steps"]),
        learning_rate=float(t_cfg["learning_rate"]),
        weight_decay=float(t_cfg["weight_decay"]),
        max_grad_norm=float(t_cfg["max_grad_norm"]),
        log_every=int(t_cfg["log_every"]),
        eval_every=int(t_cfg["eval_every"]),
        save_every=int(t_cfg["save_every"]),
        loss=str(t_cfg.get("loss", "smooth_l1")),
        smooth_l1_beta=float(t_cfg.get("smooth_l1_beta", 1.0)),
        target_transform=str(t_cfg.get("target_transform", "standardize_train")),
        precision=str(model_cfg.get("precision", "bf16")),
    )

    out_dir = ensure_run_dir(cfg["experiment"]["output_dir"], exist_ok=True)
    return train_one_regressor(
        wrapper=wrapper,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        target_transform=target_transform,
        train_cfg=train_cfg,
        out_dir=out_dir,
        config_snapshot=cfg,
    )
