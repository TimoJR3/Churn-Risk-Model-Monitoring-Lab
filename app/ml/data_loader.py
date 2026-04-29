from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_database_url() -> str:
    return (
        "postgresql+psycopg2://"
        f"{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}"
        f"/{settings.postgres_db}"
    )


def load_from_csv(
    path: Path | None = None,
) -> pd.DataFrame:
    data_path = path or RAW_DATA_DIR / "synthetic_churn_dataset.csv"
    return pd.read_csv(data_path, parse_dates=["signup_date"])


def load_from_postgres() -> pd.DataFrame:
    query = """
        SELECT
            u.user_id,
            u.signup_date,
            u.country,
            u.plan_type,
            u.monthly_fee,
            f.days_active_last_30,
            f.sessions_last_30,
            f.support_tickets_last_30,
            f.payments_failed_last_90,
            f.avg_session_duration,
            f.feature_usage_score,
            f.last_login_days_ago,
            u.churn::int AS churn
        FROM users AS u
        JOIN user_features AS f
            ON u.user_id = f.user_id
        WHERE f.snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM user_features
        )
        ORDER BY u.user_id
    """
    engine = create_engine(get_database_url())
    return pd.read_sql(query, engine, parse_dates=["signup_date"])


def load_churn_data(
    source: str = "csv",
    csv_path: Path | None = None,
) -> pd.DataFrame:
    if source == "csv":
        return load_from_csv(csv_path)

    if source == "postgres":
        return load_from_postgres()

    raise ValueError("source must be 'csv' or 'postgres'.")
