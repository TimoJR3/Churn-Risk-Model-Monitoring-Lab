from app.core.config import Settings


def test_settings_accept_database_url_and_max_batch_size_alias() -> None:
    settings = Settings(
        _env_file=None,
        DATABASE_URL="postgresql+psycopg2://user:pass@db:5432/app",
        MAX_BATCH_SIZE=25,
    )

    assert settings.database_url == "postgresql+psycopg2://user:pass@db:5432/app"
    assert settings.prediction_batch_size == 25
