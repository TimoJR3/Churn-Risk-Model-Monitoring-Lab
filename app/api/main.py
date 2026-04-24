from fastapi import FastAPI

from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health status."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }
