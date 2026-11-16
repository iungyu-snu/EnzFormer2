"""Shared helpers for ddG regressor evaluation scripts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ddg_lora.data.regression_dataset import RegressionDataset, RegressionExample
from ddg_lora.evaluation.metrics import (
    binary_classification_metrics,
    ndcg_at_k,
    per_protein_spearman_median,
    precision_at_k,
    precision_at_top_fraction,
    ranking_metrics,
    regression_errors,
)


@dataclass
class LoadedRegressor:
    model: object
    backbone_module: object
    target_transform: object
    precision: str
    batch_size: int
    regressor_kind: str


def load_regression_checkpoint(checkpoint_dir: str | Path) -> LoadedRegressor:
    """Load a trained LoRA regressor checkpoint once for evaluation."""
    import torch

    from ddg_lora.models.esmc_wrapper import ESMCMaskedLM
    from ddg_lora.models.lora import (
        LoRAConfig,
        freeze_non_lora,
        inject_lora,
        load_lora_state_dict,
    )
    from ddg_lora.models.regressors import REGRESSOR_KINDS, build_regressor
    from ddg_lora.training.regression_trainer import (
        TargetTransform,
        apply_cuda_memory_limit,
    )
    from ddg_lora.utils.io import load_yaml
    from ddg_lora.utils.seed import set_seed

    ckpt_dir = Path(checkpoint_dir)
    required = (
        ckpt_dir / "train_config.yaml",
        ckpt_dir / "target_transform.json",
        ckpt_dir / "best_adapter.pt",
        ckpt_dir / "best_head.pt",
    )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint artifacts: {missing}")

    cfg = load_yaml(ckpt_dir / "train_config.yaml")
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    target_transform = TargetTransform.from_dict(
        json.loads((ckpt_dir / "target_transform.json").read_text())
    )

    model_cfg = cfg["model"]
    apply_cuda_memory_limit(model_cfg)
    wrapper = ESMCMaskedLM.load(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
    )
    lora_cfg = cfg["lora"]
    target_patterns = tuple(
        lora_cfg.get("target_patterns") or lora_cfg.get("target_module_patterns") or []
    )
    inject_lora(
        wrapper.model,
        LoRAConfig(
            rank=int(lora_cfg["r"]),
            alpha=int(lora_cfg["alpha"]),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            target_module_patterns=target_patterns,
            fail_if_no_match=True,
        ),
    )
    freeze_non_lora(wrapper.model)
    load_lora_state_dict(wrapper.model, ckpt_dir / "best_adapter.pt", strict=True)

    reg_cfg = cfg["regressor"]
    kind = str(reg_cfg["kind"])
    if kind not in REGRESSOR_KINDS:
        raise ValueError(f"unknown regressor kind {kind!r}; expected {REGRESSOR_KINDS}")
    with torch.inference_mode():
        hidden = wrapper.forward_residue_hidden_states("ACDEFGHIKL")
    model = build_regressor(
        kind,
        backbone=wrapper,
        hidden_dim=int(hidden.shape[1]),
        head_dropout=float(reg_cfg.get("head_dropout", 0.1)),
        head_hidden_dims=tuple(reg_cfg.get("head_hidden_dims", (512, 128))),
        use_context=bool(reg_cfg.get("use_context", False)),
        context_window_k=int(reg_cfg.get("context_window_k", 5)),
        masked_logodds=reg_cfg.get("masked_logodds"),
    ).to(wrapper.device)
    best_regressor = ckpt_dir / "best_regressor.pt"
    if best_regressor.exists():
        model.load_state_dict(torch.load(best_regressor, map_location="cpu"))
    else:
        masked_path = ckpt_dir / "masked_logodds.json"
        if masked_path.exists():
            record = json.loads(masked_path.read_text(encoding="utf-8"))
            inner = getattr(model, "inner", model)
            if bool(record.get("enabled", False)) and hasattr(
                inner, "set_masked_logodds_stats"
            ):
                inner.set_masked_logodds_stats(record["mean"], record["std"])
        head_state = torch.load(ckpt_dir / "best_head.pt", map_location="cpu")
        model.head.load_state_dict(head_state)
    wrapper.model.eval()
    model.eval()

    return LoadedRegressor(
        model=model,
        backbone_module=wrapper.model,
        target_transform=target_transform,
        precision=str(model_cfg.get("precision", "bf16")),
        batch_size=int(cfg["training"].get("batch_size", 1)),
        regressor_kind=kind,
    )


def predict_examples(
    model: object,
    backbone_module: object,
    examples: Sequence[RegressionExample],
    *,
    target_transform: object,
    precision: str,
    batch_size: int,
) -> list[float]:
    """Return decoded kcal/mol ddG predictions for examples."""
    if not examples:
        return []
    import torch

    device_type = next(backbone_module.parameters()).device.type
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device_type == "cuda" and precision in {"bf16", "fp16"}
    predict_batch = getattr(model, "predict_batch", None)
    decode = getattr(target_transform, "decode")

    out: list[float] = []
    model.eval()
    backbone_module.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), max(1, int(batch_size))):
            batch = list(examples[start : start + max(1, int(batch_size))])
            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                if callable(predict_batch):
                    pred_std = predict_batch(batch)
                else:
                    pred_std = torch.stack([model.predict_one(ex) for ex in batch])
            values = pred_std.detach().float().cpu().numpy().reshape(-1)
            out.extend(float(decode(float(v))) for v in values)
    return out


def prediction_dataframe_from_dataset(
    dataset: RegressionDataset,
    ddg_hat: Sequence[float],
) -> pd.DataFrame:
    if len(dataset) != len(ddg_hat):
        raise ValueError(f"dataset length {len(dataset)} != prediction length {len(ddg_hat)}")
    rows: list[dict] = []
    for pred, ex in zip(ddg_hat, dataset.iter_examples(), strict=True):
        rows.append(
            {
                "protein_id": ex.protein_id,
                "mutation": ex.mutation,
                "position_0idx": ex.position_0idx,
                "wt_aa": ex.wt_aa,
                "mut_aa": ex.mut_aa,
                "ddg": ex.ddg,
                "label3": ex.label3,
                "label_binary": ex.label_binary,
                "ddg_hat": float(pred),
                "score": -float(pred),
            }
        )
    return pd.DataFrame(rows)


def predict_dataset(
    loaded: LoadedRegressor,
    dataset: RegressionDataset,
    *,
    batch_size: int | None = None,
) -> pd.DataFrame:
    examples = list(dataset.iter_examples())
    preds = predict_examples(
        loaded.model,
        loaded.backbone_module,
        examples,
        target_transform=loaded.target_transform,
        precision=loaded.precision,
        batch_size=int(batch_size or loaded.batch_size),
    )
    return prediction_dataframe_from_dataset(dataset, preds)


def load_threshold_value(path: str | Path) -> float:
    record = json.loads(Path(path).read_text())
    if "threshold_score_neg_ddg_hat" not in record:
        raise ValueError(f"threshold json missing threshold_score_neg_ddg_hat: {path}")
    threshold = record["threshold_score_neg_ddg_hat"]
    if threshold is None:
        raise ValueError(f"threshold json has null threshold_score_neg_ddg_hat: {path}")
    return float(threshold)


def binary_block(df: pd.DataFrame, *, threshold: float | None) -> dict:
    binary_df = df.dropna(subset=["label_binary"]).copy()
    if binary_df.empty:
        return {"n": 0}
    binary_df["label_binary"] = binary_df["label_binary"].astype(int)
    y_true = binary_df["label_binary"].tolist()
    y_score = binary_df["score"].tolist()
    if threshold is None:
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true_arr = np.asarray(y_true)
        n_pos = int((y_true_arr == 1).sum())
        n_neg = int((y_true_arr == 0).sum())
        return {
            "n": int(len(binary_df)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": float(roc_auc_score(y_true, y_score)) if n_pos and n_neg else float("nan"),
            "auprc": float(average_precision_score(y_true, y_score)) if n_pos else float("nan"),
        }
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    cm = binary_classification_metrics(y_true, y_score, y_pred)
    return {
        "accuracy": cm.accuracy,
        "balanced_accuracy": cm.balanced_accuracy,
        "f1": cm.f1,
        "precision": cm.precision,
        "recall": cm.recall,
        "mcc": cm.mcc,
        "auroc": cm.auroc,
        "auprc": cm.auprc,
        "confusion": cm.confusion,
        "n": cm.n,
        "n_pos": cm.n_pos,
        "n_neg": cm.n_neg,
    }


def regression_block(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    rk = ranking_metrics(df["ddg"].tolist(), df["score"].tolist())
    pp = per_protein_spearman_median(df, score_col="score")
    err = regression_errors(df["ddg"].tolist(), df["ddg_hat"].tolist())
    return {
        "n": int(len(df)),
        "mae": err["mae"],
        "rmse": err["rmse"],
        "spearman_neg_score_vs_ddg": rk.spearman,
        "pearson_neg_score_vs_ddg": rk.pearson,
        "per_protein_spearman_median": pp["per_protein_spearman_median"],
        "n_proteins_in_per_protein": pp["n_proteins"],
    }


def screening_block(df: pd.DataFrame, ks: Iterable[int] = (10, 20, 50, 100)) -> dict:
    binary_df = df.dropna(subset=["label_binary"]).copy()
    ndcg_ks = (10, 20, 50)
    top_fracs = (0.01,)
    if binary_df.empty:
        out = {f"precision_at_{k}": float("nan") for k in ks}
        out.update({f"ndcg_at_{k}": float("nan") for k in ndcg_ks})
        out.update(precision_at_top_fraction([], [], fracs=top_fracs))
        return out
    y_true = binary_df["label_binary"].astype(int).tolist()
    y_score = binary_df["score"].tolist()
    out = precision_at_k(y_true, y_score, ks=ks)
    out.update(ndcg_at_k(y_true, y_score, ks=ndcg_ks))
    out.update(precision_at_top_fraction(y_true, y_score, fracs=top_fracs))
    return out


def compute_prediction_metrics(df: pd.DataFrame, *, threshold: float | None) -> dict:
    return {
        "regression": regression_block(df),
        "binary": binary_block(df, threshold=threshold),
        "screening": screening_block(df),
    }
