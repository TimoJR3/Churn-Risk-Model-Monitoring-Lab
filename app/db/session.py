from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from app.core.config import settings


def get_database_url() -> str:
    return (
        "postgresql+psycopg2://"
        f"{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}"
        f"/{settings.postgres_db}"
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_database_url())
