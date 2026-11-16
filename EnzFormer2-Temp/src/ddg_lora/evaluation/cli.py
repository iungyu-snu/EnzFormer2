#!/usr/bin/env python
"""Evaluate a trained ddG regressor on validation, calibrate a binary
threshold on validation only, then touch the test split exactly once.

Run::

    enzformer2-temp evaluate \\
        --checkpoint-dir checkpoints/enzformer2_temp \\
        --data-dir data/processed/megascale_thermompnn \\
        --out-dir artifacts/siamese_delta_esmc600m_lora/evaluation

Outputs (under ``--out-dir``):
  * ``predictions_validation.csv`` and ``predictions_test.csv``
  * ``metrics.json``
  * ``threshold.json``

The score used for ranking and binary classification is ``-ddg_hat`` because
lower ddG_hat means more stabilizing. Neutral rows are kept in the prediction
CSVs but excluded from binary metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from ddg_lora.constants import DEFAULT_DESTABILIZING_MIN, DEFAULT_STABILIZING_MAX
from ddg_lora.data.regression_dataset import RegressionDataset
from ddg_lora.ensemble import bundle_members, predict_datasets, provenance, validate_member
from ddg_lora.evaluation.regression_eval import compute_prediction_metrics
from ddg_lora.evaluation.calibration import calibrate_threshold
from ddg_lora.evaluation.metrics import (
    binary_classification_metrics,
    per_protein_spearman_median,
    precision_at_k,
    ranking_metrics,
    regression_errors,
)
from ddg_lora.models.esmc_wrapper import ESMCMaskedLM
from ddg_lora.training.regression_trainer import (
    TargetTransform,
)
from ddg_lora.utils.io import dump_json
from ddg_lora.utils.logging import get_logger

log = get_logger("evaluate_regression")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True,
                   help="Run directory containing best_adapter.pt, best_head.pt, "
                        "train_config.yaml, target_transform.json.")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing validation.parquet and test.parquet.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threshold-metric", default="mcc",
                   choices=["mcc", "balanced_accuracy", "f1", "youden_j"])
    p.add_argument("--max-validation-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Evaluation microbatch size. Defaults to 16.")
    p.add_argument("--skip-test", action="store_true",
                   help="Only evaluate validation. Used by hyperparameter search to avoid touching test.")
    return p.parse_args(argv)


def _hidden_dim_probe(wrapper: ESMCMaskedLM) -> int:
    probe_seq = "ACDEFGHIKL"
    with torch.inference_mode():
        H = wrapper.forward_residue_hidden_states(probe_seq)
    return int(H.shape[1])


def _predict_split(
    model: torch.nn.Module,
    backbone_module: torch.nn.Module,
    dataset: RegressionDataset,
    target_transform: TargetTransform,
    *,
    batch_size: int,
    precision: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    backbone_module.eval()
    model.eval()
    device_type = next(backbone_module.parameters()).device.type
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device_type == "cuda" and precision in {"bf16", "fp16"}

    def _predict_batch(batch):
        predict_batch = getattr(model, "predict_batch", None)
        if callable(predict_batch):
            return predict_batch(batch)
        return torch.stack([model.predict_one(ex) for ex in batch])

    with torch.inference_mode():
        batch = []
        for ex in dataset.iter_examples():
            batch.append(ex)
            if len(batch) < max(1, batch_size):
                continue
            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                pred_std = _predict_batch(batch)
            pred_values = pred_std.detach().float().cpu().numpy().reshape(-1)
            for pred_value, item in zip(pred_values, batch, strict=True):
                pred_kcal = target_transform.decode(float(pred_value))
                rows.append(
                    {
                        "protein_id": item.protein_id,
                        "mutation": item.mutation,
                        "position_0idx": item.position_0idx,
                        "wt_aa": item.wt_aa,
                        "mut_aa": item.mut_aa,
                        "ddg": item.ddg,
                        "label3": item.label3,
                        "label_binary": item.label_binary,
                        "ddg_hat": pred_kcal,
                        "score": -pred_kcal,        # higher -> more stabilizing
                    }
                )
            batch = []
        if batch:
            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                pred_std = _predict_batch(batch)
            pred_values = pred_std.detach().float().cpu().numpy().reshape(-1)
            for pred_value, item in zip(pred_values, batch, strict=True):
                pred_kcal = target_transform.decode(float(pred_value))
                rows.append(
                    {
                        "protein_id": item.protein_id,
                        "mutation": item.mutation,
                        "position_0idx": item.position_0idx,
                        "wt_aa": item.wt_aa,
                        "mut_aa": item.mut_aa,
                        "ddg": item.ddg,
                        "label3": item.label3,
                        "label_binary": item.label_binary,
                        "ddg_hat": pred_kcal,
                        "score": -pred_kcal,
                    }
                )
    return pd.DataFrame(rows)


def _binary_block(
    df: pd.DataFrame, *, threshold: float | None
) -> dict:
    """Compute binary metrics on stab/destab rows only."""
    binary_df = df.dropna(subset=["label_binary"]).copy()
    binary_df["label_binary"] = binary_df["label_binary"].astype(int)
    if binary_df.empty:
        return {"n": 0}
    y_true = binary_df["label_binary"].tolist()
    y_score = binary_df["score"].tolist()
    if threshold is None:
        return {"n": len(binary_df)}
    y_pred = [1 if s >= threshold else 0 for s in y_score]
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


def _regression_block(df: pd.DataFrame) -> dict:
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


def _screening_block(df: pd.DataFrame, ks=(10, 50, 100)) -> dict:
    binary_df = df.dropna(subset=["label_binary"]).copy()
    if binary_df.empty:
        return {f"precision_at_{k}": float("nan") for k in ks}
    y_true = binary_df["label_binary"].astype(int).tolist()
    y_score = binary_df["score"].tolist()
    return precision_at_k(y_true, y_score, ks=ks)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ckpt_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    members = bundle_members(ckpt_dir, require_final=False)
    checkpoint_record = provenance(ckpt_dir, members)
    cfg = validate_member(members[0], require_final=False)
    kind = cfg["regressor"]["kind"]
    eval_batch_size = args.batch_size

    # ----- validation pass: regression metrics + threshold calibration -----
    val_path = data_dir / "validation.parquet"
    test_path = data_dir / "test.parquet"
    if not val_path.exists():
        log.error("missing validation parquet: %s", val_path)
        return 3
    if not args.skip_test and not test_path.exists():
        log.error("missing test parquet: %s", test_path)
        return 3

    log.info("loading validation: %s", val_path)
    val_ds = RegressionDataset(
        val_path, include_neutral=True, max_examples=args.max_validation_examples,
    )
    val_df = predict_datasets(members, {"validation": val_ds}, batch_size=eval_batch_size)["validation"]
    val_df.to_parquet(out_dir / "predictions_validation.parquet", index=False)
    val_df.to_csv(out_dir / "predictions_validation.csv", index=False)

    val_reg = _regression_block(val_df)
    val_screen = compute_prediction_metrics(val_df, threshold=None)["screening"]

    binary_val = val_df.dropna(subset=["label_binary"]).copy()
    if binary_val.empty:
        log.warning("validation set has no labeled binary rows; threshold not fit")
        threshold = None
        threshold_metric_value = float("nan")
        threshold_metric_name = args.threshold_metric
    else:
        cal = calibrate_threshold(
            binary_val["score"].tolist(),
            binary_val["label_binary"].astype(int).tolist(),
            metric=args.threshold_metric,
        )
        threshold = float(cal.threshold)
        threshold_metric_value = float(cal.metric_value)
        threshold_metric_name = cal.metric_name
        log.info(
            "validation threshold=%.4f (%s=%.4f)",
            threshold, threshold_metric_name, threshold_metric_value,
        )

    val_bin = _binary_block(val_df, threshold=threshold)

    threshold_record = {
        "fit_split": "validation", "checkpoint": checkpoint_record,
        "threshold_score_neg_ddg_hat": threshold,
        "metric_name": threshold_metric_name,
        "metric_value": threshold_metric_value,
        "binary_label_convention": {
            "stabilizing_threshold_kcal": DEFAULT_STABILIZING_MAX,
            "destabilizing_threshold_kcal": DEFAULT_DESTABILIZING_MIN,
            "score": "score = -ddg_hat (kcal/mol); higher score => more stabilizing",
        },
    }
    dump_json(threshold_record, out_dir / "threshold.json")

    if args.skip_test:
        metrics = {
            "regressor_kind": kind,
            "ddg_sign_convention": "thermompnn_positive_destabilizing",
            "validation": {
                "regression": val_reg,
                "binary": val_bin,
                "screening": val_screen,
            },
            "test": None,
            "threshold": threshold_record,
        }
        dump_json(metrics, out_dir / "metrics.json")
        log.info("Wrote validation-only %s", out_dir / "metrics.json")
        return 0

    # ----- test pass (touched exactly once with the validation-fit threshold) ---
    log.info("loading test: %s", test_path)
    test_ds = RegressionDataset(
        test_path, include_neutral=True, max_examples=args.max_test_examples,
    )
    test_df = predict_datasets(members, {"test": test_ds}, batch_size=eval_batch_size)["test"]
    test_df.to_parquet(out_dir / "predictions_test.parquet", index=False)
    test_df.to_csv(out_dir / "predictions_test.csv", index=False)

    test_reg = _regression_block(test_df)
    test_screen = compute_prediction_metrics(test_df, threshold=None)["screening"]
    test_bin = _binary_block(test_df, threshold=threshold)

    metrics = {
        "regressor_kind": kind,
        "ddg_sign_convention": "thermompnn_positive_destabilizing",
        "validation": {
            "regression": val_reg,
            "binary": val_bin,
            "screening": val_screen,
        },
        "test": {
            "regression": test_reg,
            "binary": test_bin,
            "screening": test_screen,
        },
        "threshold": threshold_record,
    }
    dump_json(metrics, out_dir / "metrics.json")
    log.info("Wrote %s", out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
