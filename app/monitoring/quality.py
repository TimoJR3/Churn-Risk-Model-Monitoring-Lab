from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


LOW_RISK_MAX = 0.3
HIGH_RISK_MIN = 0.7


def _clean_scores(values: Sequence[float | int | None]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _extract_probability(log: Mapping[str, object]) -> float | None:
    value = log.get("probability", log.get("churn_probability"))
    if value is None:
        return None

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(probability):
        return None

    return min(max(probability, 0.0), 1.0)


def _risk_band(probability: float) -> str:
    if probability < LOW_RISK_MAX:
        return "low"
    if probability < HIGH_RISK_MIN:
        return "medium"
    return "high"


def calculate_prediction_summary(
    logs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize prediction probabilities from prediction log records."""
    probabilities = [
        probability
        for log in logs
        if (probability := _extract_probability(log)) is not None
    ]
    total_predictions = len(probabilities)
    risk_band_counts = {"low": 0, "medium": 0, "high": 0}

    for probability in probabilities:
        risk_band_counts[_risk_band(probability)] += 1

    average_probability = (
        float(np.mean(probabilities)) if probabilities else None
    )
    high_risk_share = (
        risk_band_counts["high"] / total_predictions
        if total_predictions
        else 0.0
    )

    return {
        "total_predictions": total_predictions,
        "average_probability": average_probability,
        "high_risk_share": float(high_risk_share),
        "risk_band_counts": risk_band_counts,
    }


def calculate_quality_metrics(
    y_true: Sequence[int | bool] | None,
    y_score: Sequence[float | int | None],
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Calculate binary classification quality metrics from labels and scores."""
    scores = _clean_scores(y_score)
    if y_true is None:
        return {
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    labels = np.asarray(y_true, dtype=int)
    if labels.size == 0 or scores.size == 0:
        return {
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    size = min(labels.size, scores.size)
    labels = labels[:size]
    scores = scores[:size]
    predictions = (scores >= threshold).astype(int)
    unique_labels = np.unique(labels)

    roc_auc = (
        float(roc_auc_score(labels, scores))
        if unique_labels.size == 2
        else None
    )

    return {
        "roc_auc": roc_auc,
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
