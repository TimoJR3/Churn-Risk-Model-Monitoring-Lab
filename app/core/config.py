from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "Churn Risk & Model Monitoring Lab"
    app_env: str = "local"

    postgres_user: str = "churn_user"
    postgres_password: str = "churn_password"
    postgres_db: str = "churn_lab"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    prediction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
