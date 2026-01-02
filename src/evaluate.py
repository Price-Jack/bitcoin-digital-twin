"""Evaluation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> tuple[float, dict]:
    """Search thresholds and return the best by F1."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    best_t = 0.5
    best = {"F1": -1.0}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        metrics = classification_metrics(y_true, y_pred)
        if metrics["F1"] > best["F1"]:
            best = metrics
            best_t = float(t)

    return best_t, best
