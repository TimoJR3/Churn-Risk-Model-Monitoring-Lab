from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    desc,
    insert,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import get_engine
from app.schemas.prediction import PredictionRequest, PredictionResponse


metadata = MetaData()

prediction_logs_table = Table(
    "prediction_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(36), nullable=False),
    Column("user_id_hash", Text, nullable=True),
    Column("churn_probability", Numeric, nullable=False),
    Column("churn_prediction", Integer, nullable=False),
    Column("risk_band", Text, nullable=False),
    Column("threshold", Numeric, nullable=False),
    Column("model_version", Text, nullable=True),
    Column("input_features", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


class PredictionLogUnavailableError(RuntimeError):
    """Raised when prediction log persistence is unavailable."""


def hash_user_id(user_id: object) -> str:
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()


def sanitize_input_features(request: PredictionRequest) -> dict[str, Any]:
    features = request.model_dump(mode="json")
    features.pop("user_id", None)
    return features


def save_prediction_log(
    request: PredictionRequest,
    response: PredictionResponse,
    model_version: str | None,
    engine: Engine | None = None,
) -> UUID:
    request_id = uuid4()
    payload = {
        "request_id": str(request_id),
        "user_id_hash": (
            hash_user_id(request.user_id)
            if request.user_id is not None
            else None
        ),
        "churn_probability": response.churn_probability,
        "churn_prediction": response.churn_prediction,
        "risk_band": response.risk_band,
        "threshold": response.threshold,
        "model_version": model_version,
        "input_features": sanitize_input_features(request),
        "created_at": datetime.now(UTC),
    }

    database_engine = engine or get_engine()
    try:
        with database_engine.begin() as connection:
            connection.execute(insert(prediction_logs_table).values(**payload))
    except SQLAlchemyError as exc:
        raise PredictionLogUnavailableError(
            "Prediction was generated, but saving prediction log failed. "
            "Check PostgreSQL availability or set SAVE_PREDICTIONS=false."
        ) from exc

    return request_id


def fetch_recent_prediction_logs(
    limit: int = 20,
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    database_engine = engine or get_engine()
    query = (
        select(prediction_logs_table)
        .order_by(desc(prediction_logs_table.c.created_at))
        .limit(limit)
    )

    try:
        with database_engine.begin() as connection:
            rows: Sequence[Any] = connection.execute(query).mappings().all()
    except SQLAlchemyError as exc:
        raise PredictionLogUnavailableError(
            "Could not read recent prediction logs. Check PostgreSQL "
            "availability."
        ) from exc

    result = []
    for row in rows:
        item = dict(row)
        input_features = dict(item.get("input_features") or {})
        input_features.pop("user_id", None)
        item["input_features"] = input_features
        result.append(item)

    return result
