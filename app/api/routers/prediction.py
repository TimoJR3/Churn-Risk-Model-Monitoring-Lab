from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.ml.inference import (
    ModelArtifactsUnavailableError,
    predict_churn,
)
from app.schemas.prediction import PredictionRequest, PredictionResponse


router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return predict_churn(
            request=request,
            threshold=settings.prediction_threshold,
        )
    except ModelArtifactsUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_artifacts_unavailable",
                "message": str(exc),
            },
        ) from exc
