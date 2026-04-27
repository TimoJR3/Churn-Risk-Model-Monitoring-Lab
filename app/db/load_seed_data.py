from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def get_database_url() -> str:
    return (
        "postgresql+psycopg2://"
        f"{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}"
        f"/{settings.postgres_db}"
    )


def load_sql_file(engine, path: Path) -> None:
    with engine.begin() as connection:
        connection.execute(text(path.read_text(encoding="utf-8")))


def load_seed_data() -> None:
    users_path = PROCESSED_DATA_DIR / "users.csv"
    features_path = PROCESSED_DATA_DIR / "user_features.csv"

    if not users_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Seed CSV files are missing. Run data generation first."
        )

    engine = create_engine(get_database_url())

    load_sql_file(engine, PROJECT_ROOT / "app" / "db" / "schema.sql")
    load_sql_file(engine, PROJECT_ROOT / "app" / "db" / "seed_monitoring.sql")

    users = pd.read_csv(users_path, parse_dates=["signup_date"])
    features = pd.read_csv(features_path, parse_dates=["snapshot_date"])

    with engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE TABLE user_features, users "
                "RESTART IDENTITY CASCADE"
            )
        )

    users.to_sql(
        "users",
        engine,
        if_exists="append",
        index=False,
        method="multi",
    )
    features.to_sql(
        "user_features",
        engine,
        if_exists="append",
        index=False,
        method="multi",
    )


if __name__ == "__main__":
    load_seed_data()
