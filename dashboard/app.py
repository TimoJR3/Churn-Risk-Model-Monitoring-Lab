from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.api_client import (
    DEFAULT_API_BASE_URL,
    ApiResult,
    format_probability,
    get_json,
    normalize_api_base_url,
    post_json,
    risk_band_display,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "predict_batch_sample.json"


def load_batch_sample() -> list[dict[str, Any]]:
    """Load the checked-in batch prediction sample."""
    try:
        return json.loads(BATCH_SAMPLE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []


def show_api_error(result: ApiResult) -> None:
    """Render a readable API error in Streamlit."""
    st.error(result.error or "API request failed.")
    if result.data:
        st.json(result.data)
    st.code(
        "uvicorn app.api.main:app --reload\n"
        "streamlit run dashboard/app.py",
        language="bash",
    )


def render_metric_row(summary: dict[str, Any]) -> None:
    """Render prediction summary metrics."""
    total, average, high_risk = st.columns(3)
    total.metric("Total predictions", summary.get("total_predictions", 0))
    average.metric(
        "Average probability",
        format_probability(summary.get("average_probability")),
    )
    high_risk.metric(
        "High-risk share",
        format_probability(summary.get("high_risk_share")),
    )


def render_header(api_base_url: str) -> tuple[ApiResult, ApiResult]:
    """Render project header with API and model metadata."""
    health = get_json(api_base_url, "/health")
    metadata = get_json(api_base_url, "/model/metadata")

    st.title("Churn Risk & Model Monitoring Lab")

    status_label = "online" if health.ok else "offline"
    status_delta = None if health.ok else "check backend"
    health_col, model_col, threshold_col = st.columns(3)
    health_col.metric("API health", status_label, delta=status_delta)

    if metadata.ok and isinstance(metadata.data, dict):
        model_col.metric("Best model", metadata.data.get("best_model") or "n/a")
        threshold_col.metric(
            "Threshold",
            metadata.data.get("threshold", "n/a"),
        )
    else:
        model_col.metric("Best model", "n/a")
        threshold_col.metric("Threshold", "n/a")

    return health, metadata


def build_prediction_payload() -> dict[str, Any]:
    """Render sample feature inputs and return a prediction payload."""
    left, right = st.columns(2)

    with left:
        user_id = st.text_input("User ID", value="1001")
        signup_date = st.date_input("Signup date", value=date(2025, 9, 1))
        country = st.selectbox(
            "Country",
            ["US", "DE", "FR", "BR", "IN", "PL", "NL", "ES"],
            index=0,
        )
        plan_type = st.selectbox(
            "Plan type",
            ["basic", "standard", "premium"],
            index=1,
        )
        monthly_fee = st.number_input(
            "Monthly fee",
            min_value=0.01,
            value=19.99,
            step=1.0,
        )
        days_active_last_30 = st.number_input(
            "Days active last 30",
            min_value=0,
            max_value=30,
            value=12,
            step=1,
        )

    with right:
        sessions_last_30 = st.number_input(
            "Sessions last 30",
            min_value=0,
            value=30,
            step=1,
        )
        support_tickets_last_30 = st.number_input(
            "Support tickets last 30",
            min_value=0,
            value=1,
            step=1,
        )
        payments_failed_last_90 = st.number_input(
            "Payments failed last 90",
            min_value=0,
            value=0,
            step=1,
        )
        avg_session_duration = st.number_input(
            "Average session duration",
            min_value=0.0,
            value=24.5,
            step=1.0,
        )
        feature_usage_score = st.number_input(
            "Feature usage score",
            min_value=0.0,
            max_value=100.0,
            value=61.0,
            step=1.0,
        )
        last_login_days_ago = st.number_input(
            "Last login days ago",
            min_value=0,
            value=4,
            step=1,
        )

    return {
        "user_id": user_id or None,
        "signup_date": signup_date.isoformat(),
        "country": country,
        "plan_type": plan_type,
        "monthly_fee": monthly_fee,
        "days_active_last_30": days_active_last_30,
        "sessions_last_30": sessions_last_30,
        "support_tickets_last_30": support_tickets_last_30,
        "payments_failed_last_90": payments_failed_last_90,
        "avg_session_duration": avg_session_duration,
        "feature_usage_score": feature_usage_score,
        "last_login_days_ago": last_login_days_ago,
    }


def render_predict_tab(api_base_url: str) -> None:
    """Render single prediction workflow."""
    with st.form("predict-form"):
        payload = build_prediction_payload()
        submitted = st.form_submit_button("Predict churn")

    if not submitted:
        return

    result = post_json(api_base_url, "/predict", payload)
    if not result.ok:
        show_api_error(result)
        return

    response = result.data or {}
    probability, prediction, band = st.columns(3)
    probability.metric(
        "Probability",
        format_probability(response.get("churn_probability")),
    )
    prediction.metric("Prediction", response.get("churn_prediction", "n/a"))
    band.metric("Risk band", risk_band_display(response.get("risk_band")))
    st.caption(response.get("explanation", ""))


def render_batch_tab(api_base_url: str) -> None:
    """Render batch prediction demo from sample JSON."""
    sample = load_batch_sample()
    st.caption(str(BATCH_SAMPLE_PATH))
    st.json(sample)

    if not st.button("Send batch to /predict/batch"):
        return

    result = post_json(api_base_url, "/predict/batch", sample)
    if not result.ok:
        show_api_error(result)
        return

    payload = result.data or {}
    items = payload.get("items", [])
    st.metric("Rows returned", payload.get("row_count", len(items)))
    st.dataframe(pd.DataFrame(items), use_container_width=True)


def render_monitoring_tab(api_base_url: str) -> None:
    """Render recent prediction logs and summary metrics."""
    summary = get_json(api_base_url, "/monitoring/summary")
    recent = get_json(api_base_url, "/predictions/recent?limit=20")

    if summary.ok and isinstance(summary.data, dict):
        render_metric_row(summary.data)
        st.subheader("Risk band counts")
        st.dataframe(
            pd.DataFrame(
                [
                    {"risk_band": key, "count": value}
                    for key, value in summary.data.get(
                        "risk_band_counts",
                        {},
                    ).items()
                ]
            ),
            use_container_width=True,
        )
    else:
        show_api_error(summary)

    st.subheader("Recent predictions")
    if recent.ok and isinstance(recent.data, dict):
        st.dataframe(
            pd.DataFrame(recent.data.get("items", [])),
            use_container_width=True,
        )
    else:
        st.warning("Recent predictions are unavailable.")
        if recent.data:
            st.json(recent.data)


def render_model_tab(metadata: ApiResult) -> None:
    """Render model metadata and validation metrics."""
    if not metadata.ok or not isinstance(metadata.data, dict):
        show_api_error(metadata)
        return

    st.subheader("Model metadata")
    st.json(
        {
            "best_model": metadata.data.get("best_model"),
            "threshold": metadata.data.get("threshold"),
            "generated_at": metadata.data.get("generated_at"),
            "artifacts": metadata.data.get("artifacts"),
        }
    )

    st.subheader("Validation metrics")
    validation_metrics = metadata.data.get("validation_metrics", {})
    if isinstance(validation_metrics, dict) and validation_metrics:
        st.dataframe(
            pd.DataFrame(
                [
                    {"metric": key, "value": value}
                    for key, value in validation_metrics.items()
                ]
            ),
            use_container_width=True,
        )
    else:
        st.info("Validation metrics are not available yet.")


def main() -> None:
    st.set_page_config(
        page_title="Churn Monitoring Lab",
        page_icon=":chart_with_downwards_trend:",
        layout="wide",
    )

    default_api_url = normalize_api_base_url(
        os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    )
    with st.sidebar:
        api_base_url = normalize_api_base_url(
            st.text_input("API_BASE_URL", value=default_api_url)
        )
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()

    health, metadata = render_header(api_base_url)
    if not health.ok:
        st.warning(
            "Backend is not reachable. Start the API and refresh the dashboard."
        )
        st.code(
            "uvicorn app.api.main:app --reload\n"
            "streamlit run dashboard/app.py",
            language="bash",
        )

    predict_tab, batch_tab, monitoring_tab, model_tab = st.tabs(
        ["Predict", "Batch demo", "Monitoring", "Model"]
    )
    with predict_tab:
        render_predict_tab(api_base_url)
    with batch_tab:
        render_batch_tab(api_base_url)
    with monitoring_tab:
        render_monitoring_tab(api_base_url)
    with model_tab:
        render_model_tab(metadata)


if __name__ == "__main__":
    main()
