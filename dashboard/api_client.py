from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 5.0


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


def _error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "API недоступен. Проверьте, что FastAPI backend запущен."
    if isinstance(exc, httpx.TimeoutException):
        return "API не ответил вовремя. Проверьте backend logs и доступность сети."
    return str(exc)


def _safe_response_payload(response: httpx.Response) -> Any | None:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        text = response.text.strip()
        return {"message": text[:500]} if text else None


@st.cache_data(ttl=60)
def get_json(api_base_url: str, path: str) -> ApiResult:
    """GET JSON from the API with short Streamlit caching."""
    try:
        response = httpx.get(
            f"{api_base_url}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            return ApiResult(
                ok=False,
                status_code=response.status_code,
                error="API вернул невалидный JSON.",
                data=_safe_response_payload(response),
            )
        return ApiResult(ok=True, data=data, status_code=response.status_code)
    except httpx.HTTPStatusError as exc:
        return ApiResult(
            ok=False,
            data=_safe_response_payload(exc.response),
            status_code=exc.response.status_code,
            error=f"API вернул HTTP {exc.response.status_code}.",
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
        try:
            data = response.json()
        except ValueError:
            return ApiResult(
                ok=False,
                status_code=response.status_code,
                error="API вернул невалидный JSON.",
                data=_safe_response_payload(response),
            )
        return ApiResult(ok=True, data=data, status_code=response.status_code)
    except httpx.HTTPStatusError as exc:
        return ApiResult(
            ok=False,
            data=_safe_response_payload(exc.response),
            status_code=exc.response.status_code,
            error=f"API вернул HTTP {exc.response.status_code}.",
        )
    except httpx.HTTPError as exc:
        return ApiResult(ok=False, error=_error_message(exc))
