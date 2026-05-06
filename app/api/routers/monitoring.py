from fastapi import APIRouter, HTTPException, status

from app.db.prediction_logs import (
    PredictionLogUnavailableError,
    fetch_recent_prediction_logs,
)
from app.monitoring.drift import calculate_psi
from app.monitoring.quality import (
    calculate_prediction_summary,
    calculate_quality_metrics,
)
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
