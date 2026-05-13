from pydantic import ValidationError

from app.schemas.prediction import (
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)


def valid_prediction_payload() -> dict[str, object]:
    return {
        "user_id": "demo-1001",
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


def test_prediction_request_schema_accepts_valid_payload() -> None:
    request = PredictionRequest.model_validate(valid_prediction_payload())

    assert request.country == "US"
    assert request.plan_type == "standard"
    assert request.days_active_last_30 == 12


def test_prediction_request_schema_rejects_invalid_feature_values() -> None:
    payload = valid_prediction_payload()
    payload["feature_usage_score"] = 120

    try:
        PredictionRequest.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors()
    else:
        raise AssertionError("Expected schema validation error.")

    assert errors[0]["loc"] == ("feature_usage_score",)


def test_prediction_response_schema_bounds_probability_and_prediction() -> None:
    response = PredictionResponse.model_validate(
        {
            "churn_probability": 0.42,
            "churn_prediction": 0,
            "risk_band": "medium",
            "threshold": 0.5,
            "model_version": "random_forest",
            "model_artifact_name": "trained_model.pkl",
            "explanation": "demo",
        }
    )

    assert response.churn_probability == 0.42
    assert response.risk_band == "medium"


def test_batch_prediction_response_schema_requires_non_negative_row_count() -> None:
    try:
        BatchPredictionResponse.model_validate({"row_count": -1, "items": []})
    except ValidationError as exc:
        errors = exc.errors()
    else:
        raise AssertionError("Expected schema validation error.")

    assert errors[0]["loc"] == ("row_count",)
