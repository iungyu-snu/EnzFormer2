"""Classification, ranking, and regression metrics.

We rely on scikit-learn for the standard metrics and on scipy for Spearman.
Per-protein Spearman is computed via groupby on `protein_id`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float
    mcc: float
    auroc: float
    auprc: float
    confusion: dict
    n: int
    n_pos: int
    n_neg: int


def binary_classification_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    y_pred: Iterable[int],
) -> ClassificationMetrics:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
        matthews_corrcoef, precision_score, recall_score, roc_auc_score,
        average_precision_score,
    )

    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score), dtype=float)
    y_pred = np.asarray(list(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) if y_true.size else np.zeros((2, 2))
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    auroc = float(roc_auc_score(y_true, y_score)) if n_pos and n_neg else float("nan")
    auprc = float(average_precision_score(y_true, y_score)) if n_pos else float("nan")
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)) if y_true.size else float("nan"),
        auroc=auroc,
        auprc=auprc,
        confusion={
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        n=int(y_true.size),
        n_pos=n_pos,
        n_neg=n_neg,
    )


# ---------------------------------------------------------------------------
# Regression-style ranking metrics (ddg vs -score)
# ---------------------------------------------------------------------------

@dataclass
class RankingMetrics:
    spearman: float
    pearson: float
    spearman_pvalue: float


def ranking_metrics(ddg: Iterable[float], score: Iterable[float]) -> RankingMetrics:
    """Spearman and Pearson between ddg and `-score`.

    Higher `score` should correspond to lower (more negative) ddg if the
    adapter pair captures stabilization.
    """
    from scipy.stats import spearmanr, pearsonr

    ddg = np.asarray(list(ddg), dtype=float)
    score = np.asarray(list(score), dtype=float)
    if ddg.size < 2 or np.allclose(ddg, ddg[0]) or np.allclose(score, score[0]):
        return RankingMetrics(float("nan"), float("nan"), float("nan"))
    rs, p = spearmanr(ddg, -score)
    rp, _ = pearsonr(ddg, -score)
    return RankingMetrics(float(rs), float(rp), float(p))


def per_protein_spearman_median(
    df: pd.DataFrame, *, ddg_col: str = "ddg", score_col: str = "score_logodds",
    protein_col: str = "protein_id",
) -> dict[str, float]:
    """Return median Spearman across proteins (groups with <=2 rows are ignored)."""
    if df.empty or protein_col not in df.columns:
        return {"per_protein_spearman_median": float("nan"), "n_proteins": 0}
    from scipy.stats import spearmanr

    out: list[float] = []
    for _, group in df.groupby(protein_col):
        if len(group) < 3:
            continue
        if group[ddg_col].nunique() < 2 or group[score_col].nunique() < 2:
            continue
        rs, _ = spearmanr(group[ddg_col].values, -group[score_col].values)
        if not math.isnan(rs):
            out.append(float(rs))
    return {
        "per_protein_spearman_median": float(np.median(out)) if out else float("nan"),
        "n_proteins": len(out),
    }


def precision_at_k(
    y_true: Iterable[int], y_score: Iterable[float], ks: Iterable[int]
) -> dict[str, float]:
    """For each K in `ks`, fraction of the top-K (by descending score) that are positives."""
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score), dtype=float)
    if y_true.size == 0:
        return {f"precision_at_{k}": float("nan") for k in ks}
    order = np.argsort(-y_score)  # descending
    sorted_true = y_true[order]
    out: dict[str, float] = {}
    for k in ks:
        k_eff = min(int(k), sorted_true.size)
        out[f"precision_at_{k}"] = float(sorted_true[:k_eff].sum() / max(1, k_eff))
    return out


def _top_fraction_label(frac: float) -> str:
    pct = float(frac) * 100.0
    return f"{pct:g}".replace(".", "_") + "pct"


def precision_at_top_fraction(
    y_true: Iterable[int], y_score: Iterable[float], fracs: Iterable[float] = (0.01,)
) -> dict[str, float]:
    """Precision among the top ``frac`` of items ranked by descending score.

    The cutoff count is ``ceil(frac * n)`` (at least 1 when ``n > 0``), so for a
    1% fraction on 1781 binary-labeled rows we score the top 18 predictions.
    Keys are formatted like ``precision_at_top_1pct`` for ``frac=0.01``.
    """
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score), dtype=float)
    n = int(y_true.size)
    order = np.argsort(-y_score) if n else np.array([], dtype=int)
    sorted_true = y_true[order] if n else y_true
    out: dict[str, float] = {}
    for frac in fracs:
        key = f"precision_at_top_{_top_fraction_label(frac)}"
        if n == 0:
            out[key] = float("nan")
            continue
        k = max(1, int(math.ceil(float(frac) * n)))
        out[key] = float(sorted_true[:k].sum() / k)
    return out


def ndcg_at_k(
    y_true: Iterable[float], y_score: Iterable[float], ks: Iterable[int]
) -> dict[str, float]:
    """Normalized DCG at each K, ranking items by descending `y_score`.

    Relevance gains are the (non-negative) `y_true` values; with binary
    stabilizing labels this matches the precision/top-percent screening view but
    rewards putting positives nearer the top. DCG uses the standard
    ``rel_i / log2(i + 1)`` discount and is normalized by the ideal DCG (items
    sorted by descending relevance). Returns NaN for a K when no positive
    relevance exists (ideal DCG is zero).
    """
    y_true = np.asarray(list(y_true), dtype=float)
    y_score = np.asarray(list(y_score), dtype=float)
    if y_true.size == 0:
        return {f"ndcg_at_{k}": float("nan") for k in ks}
    order = np.argsort(-y_score)  # descending by predicted score
    gains = y_true[order]
    ideal = np.sort(y_true)[::-1]
    discounts = 1.0 / np.log2(np.arange(2, y_true.size + 2))
    out: dict[str, float] = {}
    for k in ks:
        k_eff = min(int(k), y_true.size)
        dcg = float(np.sum(gains[:k_eff] * discounts[:k_eff]))
        idcg = float(np.sum(ideal[:k_eff] * discounts[:k_eff]))
        out[f"ndcg_at_{k}"] = float(dcg / idcg) if idcg > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Regression errors
# ---------------------------------------------------------------------------

def regression_errors(
    ddg_true: Iterable[float], ddg_hat: Iterable[float]
) -> dict[str, float]:
    t = np.asarray(list(ddg_true), dtype=float)
    h = np.asarray(list(ddg_hat), dtype=float)
    if t.size == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    return {
        "mae": float(np.mean(np.abs(t - h))),
        "rmse": float(np.sqrt(np.mean((t - h) ** 2))),
    }
