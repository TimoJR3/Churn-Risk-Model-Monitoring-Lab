<<<<<<< HEAD
from fastapi import APIRouter, HTTPException, status

from app.db.prediction_logs import (
    PredictionLogUnavailableError,
    fetch_recent_prediction_logs,
)
from app.monitoring.drift import calculate_psi
=======
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.monitoring.drift import calculate_psi, drift_status
>>>>>>> 43b3088ee8037c6bf7d408b8c31043284e38a3a6
from app.monitoring.quality import (
    calculate_prediction_summary,
    calculate_quality_metrics,
)
<<<<<<< HEAD
from app.schemas.monitoring import (
    DriftRequest,
    DriftResponse,
    MonitoringSummaryResponse,
    QualityRequest,
    QualityResponse,
)


router = APIRouter(tags=["monitoring"])


@router.get("/monitoring/summary", response_model=MonitoringSummaryResponse)
def monitoring_summary() -> MonitoringSummaryResponse:
    try:
        logs = fetch_recent_prediction_logs(limit=100)
    except PredictionLogUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "prediction_log_unavailable",
                "message": str(exc),
            },
        ) from exc

    return MonitoringSummaryResponse(**calculate_prediction_summary(logs))


@router.post("/monitoring/drift", response_model=DriftResponse)
def monitoring_drift(request: DriftRequest) -> DriftResponse:
    return DriftResponse(
        **calculate_psi(
            expected=request.expected,
            actual=request.actual,
            buckets=request.buckets,
        )
    )


@router.post("/monitoring/quality", response_model=QualityResponse)
def monitoring_quality(request: QualityRequest) -> QualityResponse:
    try:
        metrics = calculate_quality_metrics(
            y_true=request.y_true,
            y_score=request.y_score,
            threshold=request.threshold,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "invalid_quality_input",
                "message": str(exc),
            },
        ) from exc

    return QualityResponse(**metrics)
=======


router = APIRouter(prefix="/monitoring", tags=["monitoring"])

PredictionLog = dict[str, float | str | int | bool | None]

FAKE_PREDICTION_LOGS: list[PredictionLog] = [
    {"prediction_id": 1, "probability": 0.18},
    {"prediction_id": 2, "probability": 0.46},
    {"prediction_id": 3, "probability": 0.81},
]


class PredictionSummaryResponse(BaseModel):
    total_predictions: int
    average_probability: float | None
    high_risk_share: float
    risk_band_counts: dict[Literal["low", "medium", "high"], int]


class DriftRequest(BaseModel):
    expected: list[float | None]
    actual: list[float | None]
    buckets: int = Field(default=10, ge=1, le=100)


class DriftResponse(BaseModel):
    psi: float
    status: Literal["stable", "warning", "drift"]
    buckets: int
    expected_count: int
    actual_count: int


class QualityRequest(BaseModel):
    y_true: list[int] | None = None
    y_score: list[float | None]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class QualityResponse(BaseModel):
    roc_auc: float | None
    precision: float | None
    recall: float | None
    f1: float | None


@router.get("/summary", response_model=PredictionSummaryResponse)
def get_monitoring_summary() -> dict[str, object]:
    """Return a minimal prediction monitoring summary."""
    return calculate_prediction_summary(FAKE_PREDICTION_LOGS)


@router.post("/drift", response_model=DriftResponse)
def post_monitoring_drift(request: DriftRequest) -> dict[str, object]:
    """Calculate PSI and drift status for one numeric feature."""
    result = calculate_psi(
        expected=request.expected,
        actual=request.actual,
        buckets=request.buckets,
    )
    result["status"] = drift_status(float(result["psi"]))
    return result


@router.post("/quality", response_model=QualityResponse)
def post_monitoring_quality(request: QualityRequest) -> dict[str, float | None]:
    """Calculate model quality metrics from labels and prediction scores."""
    return calculate_quality_metrics(
        y_true=request.y_true,
        y_score=request.y_score,
        threshold=request.threshold,
    )
>>>>>>> 43b3088ee8037c6bf7d408b8c31043284e38a3a6
