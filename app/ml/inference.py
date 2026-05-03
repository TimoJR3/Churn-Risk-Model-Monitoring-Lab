from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import pandas as pd

from app.ml.features import add_feature_engineering
from app.ml.preprocessing import DROP_COLUMNS, transform_to_dataframe
from app.ml.training import METRICS_PATH, PREPROCESSOR_PATH, TRAINED_MODEL_PATH
from app.schemas.prediction import (
    ArtifactStatus,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
    RiskBand,
)


ARTIFACT_COMMAND = "python -m app.ml.training --source csv --n-splits 3"


class ModelArtifactsUnavailableError(RuntimeError):
    """Raised when prediction artifacts are missing or unavailable."""


@dataclass(frozen=True)
class PredictionArtifacts:
    model: object
    preprocessor: object
    model_artifact_name: str
    model_version: str


def get_risk_band(probability: float) -> RiskBand:
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "medium"
    return "high"


def _read_model_version() -> str:
    if not METRICS_PATH.exists():
        return "unknown"

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return str(metrics.get("best_model", "unknown"))


def _read_metrics() -> dict[str, object]:
    if not METRICS_PATH.exists():
        return {}

    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_prediction_artifacts() -> PredictionArtifacts:
    missing_paths = [
        path
        for path in (TRAINED_MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH)
        if not path.exists()
    ]
    if missing_paths:
        missing_names = ", ".join(path.name for path in missing_paths)
        raise ModelArtifactsUnavailableError(
            "Model artifacts are missing: "
            f"{missing_names}. Run `{ARTIFACT_COMMAND}` before prediction."
        )

    return PredictionArtifacts(
        model=joblib.load(TRAINED_MODEL_PATH),
        preprocessor=joblib.load(PREPROCESSOR_PATH),
        model_artifact_name=TRAINED_MODEL_PATH.name,
        model_version=_read_model_version(),
    )


def _request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    return pd.DataFrame([request.model_dump(mode="json")])


def predict_churn(
    request: PredictionRequest,
    threshold: float,
) -> PredictionResponse:
    artifacts = load_prediction_artifacts()
    raw_features = _request_to_dataframe(request)
    engineered = add_feature_engineering(raw_features)
    model_features = engineered.drop(columns=DROP_COLUMNS, errors="ignore")
    transformed = transform_to_dataframe(artifacts.preprocessor, model_features)

    probability = float(artifacts.model.predict_proba(transformed)[0, 1])
    prediction = int(probability >= threshold)
    risk_band = get_risk_band(probability)

    return PredictionResponse(
        churn_probability=probability,
        churn_prediction=prediction,
        risk_band=risk_band,
        threshold=threshold,
        model_version=artifacts.model_version,
        model_artifact_name=artifacts.model_artifact_name,
        explanation=(
            f"Risk band is {risk_band}; threshold {threshold:.2f} "
            f"maps probabilities at or above threshold to churn."
        ),
    )


def get_model_metadata(threshold: float) -> ModelMetadataResponse:
    metrics = _read_metrics()

    return ModelMetadataResponse(
        best_model=(
            str(metrics["best_model"])
            if metrics.get("best_model") is not None
            else None
        ),
        validation_metrics=dict(metrics.get("validation_metrics", {})),
        artifacts={
            "trained_model": ArtifactStatus(
                path=str(TRAINED_MODEL_PATH),
                exists=TRAINED_MODEL_PATH.exists(),
            ),
            "preprocessor": ArtifactStatus(
                path=str(PREPROCESSOR_PATH),
                exists=PREPROCESSOR_PATH.exists(),
            ),
            "metrics": ArtifactStatus(
                path=str(METRICS_PATH),
                exists=METRICS_PATH.exists(),
            ),
        },
        threshold=threshold,
        generated_at=(
            str(metrics["generated_at"])
            if metrics.get("generated_at") is not None
            else None
        ),
    )
