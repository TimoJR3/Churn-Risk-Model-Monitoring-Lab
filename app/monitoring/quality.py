from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOW_RISK_MAX = 0.35
HIGH_RISK_MIN = 0.65


def _clean_labels_and_scores(
    y_true: Sequence[int | bool] | None,
    y_score: Sequence[float | int | None],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(y_score, dtype=float)
    score_mask = np.isfinite(scores)

    if y_true is None:
        return np.asarray([], dtype=int), scores[score_mask]

    labels = np.asarray(y_true, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    mask = np.isfinite(labels) & score_mask
    return labels[mask].astype(int), scores[mask]


def _extract_probability(log: Mapping[str, object]) -> float | None:
    value = log.get("churn_probability", log.get("probability"))
    if value is None:
        return None

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(probability):
        return None

    return min(max(probability, 0.0), 1.0)


def _risk_band_from_probability(probability: float) -> str:
    if probability < LOW_RISK_MAX:
        return "low"
    if probability < HIGH_RISK_MIN:
        return "medium"
    return "high"


def calculate_quality_metrics(
    y_true: Sequence[int | bool] | None,
    y_score: Sequence[float | int | None],
    threshold: float = 0.5,
) -> dict[str, float | None | int]:
    labels, scores = _clean_labels_and_scores(y_true, y_score)
    sample_count = int(len(scores) if y_true is None else len(labels))
    positive_count = int(labels.sum()) if len(labels) else 0

    if y_true is None or sample_count == 0:
        return {
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "sample_count": sample_count,
            "positive_count": positive_count,
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
        "positive_count": positive_count,
    }


def calculate_prediction_summary(
    logs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    probabilities = []
    risk_band_counts = {"low": 0, "medium": 0, "high": 0}

    for log in logs:
        probability = _extract_probability(log)
        if probability is None:
            continue

        probabilities.append(probability)
        risk_band = str(log.get("risk_band") or "").lower()
        if risk_band not in risk_band_counts:
            risk_band = _risk_band_from_probability(probability)
        risk_band_counts[risk_band] += 1

    total_predictions = len(probabilities)
    high_risk_share = (
        risk_band_counts["high"] / total_predictions
        if total_predictions
        else 0.0
    )

    return {
        "total_predictions": total_predictions,
        "average_probability": (
            float(np.mean(probabilities)) if probabilities else None
        ),
        "high_risk_share": float(high_risk_share),
        "risk_band_counts": risk_band_counts,
    }
