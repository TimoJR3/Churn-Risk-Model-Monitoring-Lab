from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.ml import inference


def sample_payload() -> dict[str, object]:
    return {
        "signup_date": "2025-09-01",
        "country": "US",
        "plan_type": "standard",
        "monthly_fee": 19.99,
        "days_active_last_30": 12,
        "sessions_last_30": 30,
        "support_tickets_last_30": 1,
        "payments_failed_last_90": 0,
        "avg_session_duration": 24.5,
        "feature_usage_score": 61.0,
        "last_login_days_ago": 4,
    }


def test_risk_band_thresholds() -> None:
    assert inference.get_risk_band(0.349) == "low"
    assert inference.get_risk_band(0.35) == "medium"
    assert inference.get_risk_band(0.649) == "medium"
    assert inference.get_risk_band(0.65) == "high"


def test_predict_endpoint_returns_prediction_with_fake_artifacts(
    client: TestClient,
    monkeypatch,
) -> None:
    class FakePreprocessor:
        def transform(self, data):
            return np.array([[1.0]])

        def get_feature_names_out(self):
            return ["feature_a"]

    class FakeModel:
        def predict_proba(self, data):
            return np.array([[0.42, 0.58]])

    monkeypatch.setattr(
        inference,
        "load_prediction_artifacts",
        lambda: inference.PredictionArtifacts(
            model=FakeModel(),
            preprocessor=FakePreprocessor(),
            model_artifact_name="trained_model.pkl",
            model_version="fake_model",
        ),
    )

    response = client.post("/predict", json=sample_payload())
    payload = response.json()

    assert response.status_code == 200
    assert payload["churn_probability"] == 0.58
    assert payload["churn_prediction"] == 1
    assert payload["risk_band"] == "medium"
    assert payload["threshold"] == 0.5
    assert payload["model_artifact_name"] == "trained_model.pkl"
    assert payload["explanation"]


def test_predict_endpoint_rejects_empty_payload(client: TestClient) -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_missing_artifacts_raise_controlled_error(monkeypatch, tmp_path) -> None:
    inference.load_prediction_artifacts.cache_clear()
    monkeypatch.setattr(inference, "TRAINED_MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(
        inference,
        "PREPROCESSOR_PATH",
        tmp_path / "preprocessor.pkl",
    )
    monkeypatch.setattr(inference, "METRICS_PATH", tmp_path / "metrics.json")

    try:
        try:
            inference.load_prediction_artifacts()
        except inference.ModelArtifactsUnavailableError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected missing artifacts error.")

        assert "Model artifacts are missing" in message
        assert "python -m app.ml.training --source csv --n-splits 3" in message
    finally:
        inference.load_prediction_artifacts.cache_clear()


def test_predict_endpoint_returns_503_for_missing_artifacts(
    client: TestClient,
    monkeypatch,
) -> None:
    def raise_missing_artifacts() -> inference.PredictionArtifacts:
        raise inference.ModelArtifactsUnavailableError(
            "Model artifacts are missing. Run training first."
        )

    monkeypatch.setattr(
        inference,
        "load_prediction_artifacts",
        raise_missing_artifacts,
    )

    response = client.post("/predict", json=sample_payload())
    payload = response.json()

    assert response.status_code == 503
    assert payload["detail"]["error"] == "model_artifacts_unavailable"
    assert "traceback" not in str(payload).lower()
