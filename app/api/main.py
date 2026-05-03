from fastapi import FastAPI

from app.api.routers.prediction import router as prediction_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)
app.include_router(prediction_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health status."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }
