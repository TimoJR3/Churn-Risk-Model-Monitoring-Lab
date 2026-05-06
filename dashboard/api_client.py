from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st


DEFAULT_API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 5.0
RISK_BAND_LABELS = {
    "low": "Low risk",
    "medium": "Medium risk",
    "high": "High risk",
}


@dataclass(frozen=True)
class ApiResult:
    """Safe API response wrapper for Streamlit rendering."""

    ok: bool
    data: Any | None = None
    status_code: int | None = None
    error: str | None = None


def normalize_api_base_url(value: str | None) -> str:
    """Normalize an API base URL value from env or sidebar input."""
    if not value or not value.strip():
        return DEFAULT_API_BASE_URL
    return value.strip().rstrip("/")


def format_probability(value: float | int | None) -> str:
    """Format a probability as a dashboard-friendly percentage."""
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def risk_band_display(value: str | None) -> str:
    """Return a human-readable risk band label."""
    if value is None:
        return "Unknown risk"
    return RISK_BAND_LABELS.get(value.lower(), "Unknown risk")


def _error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "API is unavailable. Check that the FastAPI service is running."
    if isinstance(exc, httpx.TimeoutException):
        return "API request timed out. Check backend logs and connectivity."
    return str(exc)


@st.cache_data(ttl=60)
def get_json(api_base_url: str, path: str) -> ApiResult:
    """GET JSON from the API with short Streamlit caching."""
    try:
        response = httpx.get(
            f"{api_base_url}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return ApiResult(ok=True, data=response.json(), status_code=response.status_code)
    except httpx.HTTPStatusError as exc:
        return ApiResult(
            ok=False,
            data=exc.response.json() if exc.response.content else None,
            status_code=exc.response.status_code,
            error=f"API returned HTTP {exc.response.status_code}.",
        )
    except httpx.HTTPError as exc:
        return ApiResult(ok=False, error=_error_message(exc))


def post_json(api_base_url: str, path: str, payload: Any) -> ApiResult:
    """POST JSON to the API without caching."""
    try:
        response = httpx.post(
            f"{api_base_url}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return ApiResult(ok=True, data=response.json(), status_code=response.status_code)
    except httpx.HTTPStatusError as exc:
        return ApiResult(
            ok=False,
            data=exc.response.json() if exc.response.content else None,
            status_code=exc.response.status_code,
            error=f"API returned HTTP {exc.response.status_code}.",
        )
    except httpx.HTTPError as exc:
        return ApiResult(ok=False, error=_error_message(exc))
