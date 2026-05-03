from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.api.routers import prediction as prediction_router
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


def test_model_metadata_uses_metrics_and_artifact_status(
    client: TestClient,
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "trained_model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    metrics_path = tmp_path / "metrics.json"
    model_path.write_text("model", encoding="utf-8")
    preprocessor_path.write_text("preprocessor", encoding="utf-8")
    metrics_path.write_text(
        (
            '{"best_model": "random_forest", '
            '"validation_metrics": {"roc_auc": 0.91, "f1": 0.44}, '
            '"generated_at": "2026-05-04T10:00:00Z"}'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(inference, "TRAINED_MODEL_PATH", model_path)
    monkeypatch.setattr(inference, "PREPROCESSOR_PATH", preprocessor_path)
    monkeypatch.setattr(inference, "METRICS_PATH", metrics_path)

    response = client.get("/model/metadata")
    payload = response.json()

    assert response.status_code == 200
    assert payload["best_model"] == "random_forest"
    assert payload["validation_metrics"]["roc_auc"] == 0.91
    assert payload["artifacts"]["trained_model"]["exists"] is True
    assert payload["artifacts"]["preprocessor"]["path"] == str(preprocessor_path)
    assert payload["threshold"] == 0.5
    assert payload["generated_at"] == "2026-05-04T10:00:00Z"


def test_predict_batch_returns_row_count(
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

    response = client.post(
        "/predict/batch",
        json=[sample_payload(), sample_payload()],
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["row_count"] == 2
    assert len(payload["items"]) == 2
    assert payload["items"][0]["risk_band"] == "medium"


def test_predict_batch_size_limit_returns_controlled_error(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(prediction_router.settings, "prediction_batch_size", 1)

    response = client.post(
        "/predict/batch",
        json=[sample_payload(), sample_payload()],
    )
    payload = response.json()

    assert response.status_code == 413
    assert payload["detail"]["error"] == "batch_size_limit_exceeded"
    assert payload["detail"]["limit"] == 1


def test_predict_batch_invalid_item_returns_validation_error(
    client: TestClient,
) -> None:
    invalid_payload = sample_payload()
    invalid_payload["country"] = "INVALID"

    response = client.post(
        "/predict/batch",
        json=[sample_payload(), invalid_payload],
    )
    payload = response.json()

    assert response.status_code == 422
    assert payload["detail"]
    assert payload["detail"][0]["loc"][0:2] == ["body", 1]
