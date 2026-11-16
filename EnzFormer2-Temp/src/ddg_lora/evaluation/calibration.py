"""Threshold calibration on the validation split.

Hard rule: callers MUST pass the validation split. Never the test split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class CalibrationResult:
    threshold: float
    metric_value: float
    metric_name: str


def _grid(scores: Sequence[float], n: int) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        return np.array([lo], dtype=float)
    return np.linspace(lo, hi, max(2, int(n)))


def calibrate_threshold_youden_j(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    grid_size: int = 401,
) -> tuple[float, float]:
    """Pick the threshold that maximizes Youden's J = TPR - FPR. (Numpy-free)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if len(scores) == 0:
        raise ValueError("empty input to calibrate_threshold_youden_j")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    pos = sum(1 for y in labels if y == 1)
    neg = sum(1 for y in labels if y == 0)
    if pos == 0 or neg == 0:
        raise ValueError("validation set must contain both classes")

    lo, hi = min(scores), max(scores)
    if lo == hi:
        return lo, 0.0

    step = (hi - lo) / (grid_size - 1)
    best_t = lo
    best_j = -1.0
    for i in range(grid_size):
        t = lo + i * step
        tp = sum(1 for s, y in zip(scores, labels) if y == 1 and s >= t)
        fp = sum(1 for s, y in zip(scores, labels) if y == 0 and s >= t)
        tpr = tp / pos
        fpr = fp / neg
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return best_t, best_j


def calibrate_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    *,
    metric: str = "mcc",
    grid_size: int = 401,
) -> CalibrationResult:
    """Pick the threshold that maximizes one of {mcc, balanced_accuracy, f1, youden_j}.

    Hard requirement: caller passes the *validation* scores/labels. Doing this
    on the test split is a leakage bug.
    """
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, matthews_corrcoef,
    )

    scores_a = np.asarray(list(scores), dtype=float)
    labels_a = np.asarray(list(labels))
    if scores_a.size == 0:
        raise ValueError("empty validation set")
    if not (np.unique(labels_a) == np.array([0, 1])).all():
        # Allow a single-class set only if the metric is youden_j (which already errors).
        if set(np.unique(labels_a)) - {0, 1}:
            raise ValueError(f"labels must be {0,1}; got {sorted(set(labels_a.tolist()))}")
    if metric == "youden_j":
        t, j = calibrate_threshold_youden_j(scores_a.tolist(), labels_a.tolist(), grid_size=grid_size)
        return CalibrationResult(threshold=float(t), metric_value=float(j), metric_name="youden_j")

    grid = _grid(scores_a.tolist(), grid_size)
    best_t = float(grid[0])
    best_v = -1e18

    def _value_for_threshold(t: float) -> float:
        y_pred = (scores_a >= t).astype(int)
        if metric == "mcc":
            return float(matthews_corrcoef(labels_a, y_pred))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(labels_a, y_pred))
        if metric == "f1":
            return float(f1_score(labels_a, y_pred, zero_division=0))
        raise ValueError(f"unknown metric: {metric!r}")

    for t in grid:
        v = _value_for_threshold(float(t))
        if v > best_v:
            best_v = v
            best_t = float(t)
    return CalibrationResult(threshold=best_t, metric_value=best_v, metric_name=metric)
