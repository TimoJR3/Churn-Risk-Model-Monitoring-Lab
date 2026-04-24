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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
