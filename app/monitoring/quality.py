from __future__ import annotations

<<<<<<< HEAD
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
=======
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
>>>>>>> 43b3088ee8037c6bf7d408b8c31043284e38a3a6
        return {
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
<<<<<<< HEAD
            "sample_count": 0,
            "positive_count": 0,
        }

    predictions = (scores >= threshold).astype(int)
    has_two_classes = len(np.unique(labels)) == 2

    return {
        "roc_auc": (
            float(roc_auc_score(labels, scores)) if has_two_classes else None
        ),
=======
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
>>>>>>> 43b3088ee8037c6bf7d408b8c31043284e38a3a6
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
<<<<<<< HEAD
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
=======
>>>>>>> 43b3088ee8037c6bf7d408b8c31043284e38a3a6
    }
