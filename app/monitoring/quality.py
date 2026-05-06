from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _clean_labels_and_scores(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    mask = np.isfinite(labels) & np.isfinite(scores)
    return labels[mask].astype(int), scores[mask]


def calculate_quality_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float | None | int]:
    labels, scores = _clean_labels_and_scores(y_true, y_score)
    sample_count = int(len(labels))

    if sample_count == 0:
        return {
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "sample_count": 0,
            "positive_count": 0,
        }

    predictions = (scores >= threshold).astype(int)
    has_two_classes = len(np.unique(labels)) == 2

    return {
        "roc_auc": (
            float(roc_auc_score(labels, scores)) if has_two_classes else None
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sample_count": sample_count,
        "positive_count": int(labels.sum()),
    }


def calculate_prediction_summary(logs: list[dict[str, object]]) -> dict[str, object]:
    total_predictions = len(logs)
    if total_predictions == 0:
        return {
            "total_predictions": 0,
            "average_probability": None,
            "high_risk_share": 0.0,
            "risk_band_counts": {"low": 0, "medium": 0, "high": 0},
        }

    probabilities = np.asarray(
        [float(log["churn_probability"]) for log in logs],
        dtype=float,
    )
    risk_bands = [str(log["risk_band"]) for log in logs]
    risk_band_counts = {
        "low": risk_bands.count("low"),
        "medium": risk_bands.count("medium"),
        "high": risk_bands.count("high"),
    }

    return {
        "total_predictions": total_predictions,
        "average_probability": float(np.mean(probabilities)),
        "high_risk_share": risk_band_counts["high"] / total_predictions,
        "risk_band_counts": risk_band_counts,
    }
