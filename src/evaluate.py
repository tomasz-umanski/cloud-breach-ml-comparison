"""Metric computation shared by all models."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Return the full set of classification metrics for a probability vector."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # Curves stored for plotting (down-sampled to keep JSON small).
        "roc_curve": _downsample(fpr, tpr),
        "pr_curve": _downsample(rec_curve, prec_curve),
    }


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> dict:
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x, y = x[idx], y[idx]
    return {"x": x.tolist(), "y": y.tolist()}


def measure_inference_time(model, X: np.ndarray, repeats: int = 3) -> float:
    """Average wall-clock seconds to score the whole array ``X`` once."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(X)
        times.append(time.perf_counter() - start)
    return float(np.mean(times))
