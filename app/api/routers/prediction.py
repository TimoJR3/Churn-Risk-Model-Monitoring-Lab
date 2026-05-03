from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Query, status

from app.core.config import settings
from app.db.prediction_logs import (
    PredictionLogUnavailableError,
    fetch_recent_prediction_logs,
    save_prediction_log,
)
from app.ml.inference import (
    ModelArtifactsUnavailableError,
    get_model_metadata,
    predict_churn,
)
from app.schemas.prediction import (
    BatchPredictionResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
    RecentPredictionLogsResponse,
)


router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        response = predict_churn(
            request=request,
            threshold=settings.prediction_threshold,
        )
        if settings.save_predictions:
            save_prediction_log(
                request=request,
                response=response,
                model_version=response.model_version,
            )
        return response
    except ModelArtifactsUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_artifacts_unavailable",
                "message": str(exc),
            },
        ) from exc
    except PredictionLogUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "prediction_log_unavailable",
                "message": str(exc),
            },
        ) from exc


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    requests: Annotated[list[PredictionRequest], Body(min_length=1)],
) -> BatchPredictionResponse:
    if len(requests) > settings.prediction_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "batch_size_limit_exceeded",
                "message": (
                    "Batch size exceeds configured limit "
                    f"of {settings.prediction_batch_size}."
                ),
                "limit": settings.prediction_batch_size,
            },
        )

    try:
        items = [
            predict_churn(
                request=request,
                threshold=settings.prediction_threshold,
            )
            for request in requests
        ]
    except ModelArtifactsUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_artifacts_unavailable",
                "message": str(exc),
            },
        ) from exc

    return BatchPredictionResponse(row_count=len(items), items=items)


@router.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    return get_model_metadata(threshold=settings.prediction_threshold)


@router.get(
    "/predictions/recent",
    response_model=RecentPredictionLogsResponse,
)
def recent_predictions(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> RecentPredictionLogsResponse:
    try:
        rows = fetch_recent_prediction_logs(limit=limit)
    except PredictionLogUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "prediction_log_unavailable",
                "message": str(exc),
            },
        ) from exc

    return RecentPredictionLogsResponse(row_count=len(rows), items=rows)
