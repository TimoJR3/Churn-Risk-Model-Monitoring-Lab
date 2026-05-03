from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select

from app.api.routers import prediction as prediction_router
from app.db import prediction_logs
from app.ml import inference
from app.schemas.prediction import PredictionRequest, PredictionResponse


def sample_payload() -> dict[str, object]:
    return {
        "user_id": 1001,
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

    saved_logs = []

    def fake_save_prediction_log(
        request: PredictionRequest,
        response: PredictionResponse,
        model_version: str | None,
    ) -> None:
        saved_logs.append(
            {
                "request": request,
                "response": response,
                "model_version": model_version,
            }
        )

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
    monkeypatch.setattr(
        prediction_router,
        "save_prediction_log",
        fake_save_prediction_log,
    )

    response = client.post("/predict", json=sample_payload())
    payload = response.json()

    assert response.status_code == 200
    assert payload["churn_probability"] == 0.58
    assert payload["churn_prediction"] == 1
    assert payload["risk_band"] == "medium"
    assert payload["threshold"] == 0.5
    assert payload["model_version"] == "fake_model"
    assert payload["model_artifact_name"] == "trained_model.pkl"
    assert payload["explanation"]
    assert len(saved_logs) == 1
    assert saved_logs[0]["model_version"] == "fake_model"


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


def test_hash_user_id_is_deterministic() -> None:
    first = prediction_logs.hash_user_id(1001)
    second = prediction_logs.hash_user_id("1001")

    assert first == second
    assert first != "1001"


def test_save_prediction_log_inserts_row_with_sanitized_features() -> None:
    engine = create_engine("sqlite:///:memory:")
    prediction_logs.metadata.create_all(engine)

    request = PredictionRequest.model_validate(sample_payload())
    response = PredictionResponse(
        churn_probability=0.58,
        churn_prediction=1,
        risk_band="medium",
        threshold=0.5,
        model_version="fake_model",
        model_artifact_name="trained_model.pkl",
        explanation="test",
    )

    request_id = prediction_logs.save_prediction_log(
        request=request,
        response=response,
        model_version=response.model_version,
        engine=engine,
    )

    with engine.begin() as connection:
        row = connection.execute(
            select(prediction_logs.prediction_logs_table)
        ).mappings().one()

    assert str(row["request_id"]) == str(request_id)
    assert row["user_id_hash"] == prediction_logs.hash_user_id(1001)
    assert float(row["churn_probability"]) == 0.58
    assert row["churn_prediction"] == 1
    assert row["risk_band"] == "medium"
    assert row["input_features"]["country"] == "US"
    assert "user_id" not in row["input_features"]


def test_predict_returns_503_when_log_save_fails(
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
    monkeypatch.setattr(
        prediction_router,
        "save_prediction_log",
        lambda **kwargs: (_ for _ in ()).throw(
            prediction_logs.PredictionLogUnavailableError("DB unavailable")
        ),
    )

    response = client.post("/predict", json=sample_payload())
    payload = response.json()

    assert response.status_code == 503
    assert payload["detail"]["error"] == "prediction_log_unavailable"


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


def test_recent_predictions_limit_validation(client: TestClient) -> None:
    too_low = client.get("/predictions/recent?limit=0")
    too_high = client.get("/predictions/recent?limit=101")

    assert too_low.status_code == 422
    assert too_high.status_code == 422


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
