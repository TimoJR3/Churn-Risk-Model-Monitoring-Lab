from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DriftStatus = Literal["stable", "warning", "drift"]


class DriftRequest(BaseModel):
    expected: list[float]
    actual: list[float]
    buckets: int = Field(default=10, ge=1, le=100)


class DriftResponse(BaseModel):
    psi: float = Field(ge=0)
    status: DriftStatus
    bucket_count: int = Field(ge=0)
    expected_count: int = Field(ge=0)
    actual_count: int = Field(ge=0)


class QualityRequest(BaseModel):
    y_true: list[int]
    y_score: list[float]
    threshold: float = Field(default=0.5, ge=0, le=1)


class QualityResponse(BaseModel):
    roc_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    sample_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)


class MonitoringSummaryResponse(BaseModel):
    total_predictions: int = Field(ge=0)
    average_probability: float | None = None
    high_risk_share: float = Field(ge=0, le=1)
    risk_band_counts: dict[str, int]
