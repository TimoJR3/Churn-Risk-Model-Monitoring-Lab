import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.monitoring.drift import calculate_psi, drift_status
from app.monitoring.quality import (
    calculate_prediction_summary,
    calculate_quality_metrics,
)


client = TestClient(app)


def test_psi_identical_distributions_is_close_to_zero() -> None:
    expected = [1, 2, 3, 4, 5, None]
    actual = [1, 2, 3, 4, 5, None]

    result = calculate_psi(expected, actual, buckets=5)

    assert result["psi"] == pytest.approx(0.0)


def test_psi_shifted_distribution_is_higher_than_identical() -> None:
    expected = [1, 2, 3, 4, 5] * 20
    identical = [1, 2, 3, 4, 5] * 20
    shifted = [6, 7, 8, 9, 10] * 20

    identical_psi = calculate_psi(expected, identical, buckets=5)["psi"]
    shifted_psi = calculate_psi(expected, shifted, buckets=5)["psi"]

    assert shifted_psi > identical_psi


def test_drift_status_thresholds() -> None:
    assert drift_status(0.099) == "stable"
    assert drift_status(0.1) == "warning"
    assert drift_status(0.249) == "warning"
    assert drift_status(0.25) == "drift"


def test_quality_metrics_for_valid_binary_case() -> None:
    result = calculate_quality_metrics(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.4, 0.35, 0.8],
        threshold=0.5,
    )

    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)


def test_quality_metrics_single_class_labels_do_not_fail() -> None:
    result = calculate_quality_metrics(
        y_true=[1, 1, 1],
        y_score=[0.2, 0.8, 0.9],
        threshold=0.5,
    )

    assert result["roc_auc"] is None
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(0.8)


def test_prediction_summary_counts_fake_logs() -> None:
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
    assert result["risk_band_counts"] == {
        "low": 1,
        "medium": 1,
        "high": 1,
    }


def test_monitoring_summary_endpoint_works_on_fake_logs() -> None:
    response = client.get("/monitoring/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_predictions"] == 3
    assert set(payload["risk_band_counts"]) == {"low", "medium", "high"}
