from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.routers import monitoring as monitoring_router
from app.monitoring.drift import calculate_psi, drift_status
from app.monitoring.quality import (
    calculate_prediction_summary,
    calculate_quality_metrics,
)


def test_psi_identical_distributions_is_near_zero() -> None:
    values = [1, 2, 3, 4, 5, None]

    result = calculate_psi(values, values, buckets=5)

    assert result["psi"] == pytest.approx(0.0)
    assert result["status"] == "stable"


def test_psi_shifted_distribution_is_greater_than_identical() -> None:
    expected = [1, 2, 3, 4, 5] * 20
    shifted = [6, 7, 8, 9, 10] * 20

    identical = calculate_psi(expected, expected, buckets=5)
    drifted = calculate_psi(expected, shifted, buckets=5)

    assert drifted["psi"] > identical["psi"]


def test_drift_status_thresholds() -> None:
    assert drift_status(0.099) == "stable"
    assert drift_status(0.1) == "warning"
    assert drift_status(0.249) == "warning"
    assert drift_status(0.25) == "drift"


def test_quality_metrics_normal_case() -> None:
    result = calculate_quality_metrics(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.4, 0.6, 0.9],
        threshold=0.5,
    )

    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["sample_count"] == 4
    assert result["positive_count"] == 2


def test_quality_metrics_single_class_does_not_fail() -> None:
    result = calculate_quality_metrics(
        y_true=[0, 0, 0],
        y_score=[0.1, 0.2, 0.3],
        threshold=0.5,
    )

    assert result["roc_auc"] is None
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["sample_count"] == 3


def test_prediction_summary_counts_probability_logs() -> None:
    result = calculate_prediction_summary(
        [
            {"probability": 0.1},
            {"probability": 0.4},
            {"probability": 0.8},
        ]
    )

    assert result["total_predictions"] == 3
    assert result["average_probability"] == pytest.approx(1.3 / 3)
    assert result["high_risk_share"] == pytest.approx(1 / 3)
    assert result["risk_band_counts"] == {"low": 1, "medium": 1, "high": 1}


def test_monitoring_summary_uses_fake_logs(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        monitoring_router,
        "fetch_recent_prediction_logs",
        lambda limit=100: [
            {"churn_probability": 0.2, "risk_band": "low"},
            {"churn_probability": 0.7, "risk_band": "high"},
            {"churn_probability": 0.5, "risk_band": "medium"},
        ],
    )

    response = client.get("/monitoring/summary")
    payload = response.json()

    assert response.status_code == 200
    assert payload["total_predictions"] == 3
    assert payload["average_probability"] == pytest.approx(
        (0.2 + 0.7 + 0.5) / 3
    )
    assert payload["high_risk_share"] == pytest.approx(1 / 3)
    assert payload["risk_band_counts"] == {"low": 1, "medium": 1, "high": 1}
