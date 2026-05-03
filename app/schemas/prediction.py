from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


RiskBand = Literal["low", "medium", "high"]


class PredictionRequest(BaseModel):
    signup_date: date
    country: Literal["US", "DE", "FR", "BR", "IN", "PL", "NL", "ES"]
    plan_type: Literal["basic", "standard", "premium"]
    monthly_fee: float = Field(gt=0)
    days_active_last_30: int = Field(ge=0, le=30)
    sessions_last_30: int = Field(ge=0)
    support_tickets_last_30: int | None = Field(default=None, ge=0)
    payments_failed_last_90: int = Field(ge=0)
    avg_session_duration: float | None = Field(default=None, ge=0)
    feature_usage_score: float | None = Field(default=None, ge=0, le=100)
    last_login_days_ago: int = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: int = Field(ge=0, le=1)
    risk_band: RiskBand
    threshold: float = Field(ge=0, le=1)
    model_artifact_name: str
    explanation: str


class BatchPredictionResponse(BaseModel):
    row_count: int = Field(ge=0)
    items: list[PredictionResponse]


class ArtifactStatus(BaseModel):
    path: str
    exists: bool


class ModelMetadataResponse(BaseModel):
    best_model: str | None = None
    validation_metrics: dict[str, object]
    artifacts: dict[str, ArtifactStatus]
    threshold: float = Field(ge=0, le=1)
    generated_at: str | None = None
