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
    risk_band_css_class,
    risk_band_display,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "predict_batch_sample.json"


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #e8fbff;
            --muted: #8aa0a8;
            --panel: rgba(10, 18, 30, 0.74);
            --line: rgba(125, 245, 255, 0.22);
            --cyan: #42f6ff;
            --green: #7dffbd;
            --amber: #ffd166;
            --red: #ff5c8a;
        }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 12% 12%, rgba(66,246,255,0.18), transparent 28%),
                radial-gradient(circle at 78% 8%, rgba(255,92,138,0.16), transparent 26%),
                linear-gradient(135deg, #04070e 0%, #0a1420 46%, #111827 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(4,7,14,0.98), rgba(9,18,31,0.96));
            border-right: 1px solid var(--line);
        }
        .block-container {
            padding-top: 1.2rem;
            max-width: 1420px;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(16,28,46,0.92), rgba(7,13,24,0.82));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 0 28px rgba(66,246,255,0.08);
        }
        .command-deck {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(120deg, rgba(66,246,255,0.16), transparent 34%),
                linear-gradient(315deg, rgba(255,92,138,0.12), transparent 32%),
                rgba(7, 14, 25, 0.82);
            box-shadow: 0 0 46px rgba(66,246,255,0.10);
        }
        .command-title {
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: 0;
            margin: 0;
        }
        .command-subtitle {
            color: var(--muted);
            font-size: 0.96rem;
            margin-top: 0.6rem;
        }
        .signal-strip {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.9rem;
        }
        .signal-pill, .risk-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.34rem 0.72rem;
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.04);
        }
        .signal-online { color: var(--green); border-color: rgba(125,255,189,0.45); }
        .signal-offline { color: var(--red); border-color: rgba(255,92,138,0.45); }
        .risk-low { color: var(--green); border-color: rgba(125,255,189,0.45); }
        .risk-medium { color: var(--amber); border-color: rgba(255,209,102,0.45); }
        .risk-high { color: var(--red); border-color: rgba(255,92,138,0.48); }
        .risk-unknown { color: var(--muted); }
        .panel {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            background: var(--panel);
        }
        .section-kicker {
            color: var(--cyan);
            font-weight: 780;
            font-size: 0.78rem;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .probability-readout {
            font-size: 4rem;
            line-height: 1;
            font-weight: 850;
            letter-spacing: 0;
            margin: 0.2rem 0 0.45rem;
            color: var(--cyan);
            text-shadow: 0 0 24px rgba(66,246,255,0.28);
        }
        .small-muted { color: var(--muted); font-size: 0.88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_batch_sample() -> list[dict[str, Any]]:
    try:
        return json.loads(BATCH_SAMPLE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def show_api_error(result: ApiResult) -> None:
    st.error(result.error or "API request failed.")
    if result.data:
        st.json(result.data)
    st.info("Start the backend, then refresh this dashboard.")
    st.code(
        "uvicorn app.api.main:app --reload\n"
        "streamlit run dashboard/app.py",
        language="bash",
    )


def render_header(api_base_url: str) -> tuple[ApiResult, ApiResult]:
    health = get_json(api_base_url, "/health")
    metadata = get_json(api_base_url, "/model/metadata")
    online = health.ok
    model_name = "n/a"
    threshold = "n/a"

    if metadata.ok and isinstance(metadata.data, dict):
        model_name = str(metadata.data.get("best_model") or "n/a")
        threshold = str(metadata.data.get("threshold", "n/a"))

    signal_class = "signal-online" if online else "signal-offline"
    signal_text = "API ONLINE" if online else "API OFFLINE"

    st.markdown(
        f"""
        <section class="command-deck">
            <p class="section-kicker">Retention command surface</p>
            <h1 class="command-title">Churn Risk & Model Monitoring Lab</h1>
            <p class="command-subtitle">
                Live prediction cockpit for churn inference, batch scoring, model quality,
                and drift signals.
            </p>
            <div class="signal-strip">
                <span class="signal-pill {signal_class}">{signal_text}</span>
                <span class="signal-pill">MODEL: {model_name}</span>
                <span class="signal-pill">THRESHOLD: {threshold}</span>
                <span class="signal-pill">API: {api_base_url}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    return health, metadata


def render_metric_row(summary: dict[str, Any]) -> None:
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


def build_prediction_payload() -> dict[str, Any]:
    left, right = st.columns(2)

    with left:
        st.markdown('<p class="section-kicker">Identity</p>', unsafe_allow_html=True)
        user_id = st.text_input("User ID", value="1001")
        signup_date = st.date_input("Signup date", value=date(2025, 9, 1))
        country = st.selectbox(
            "Country",
            ["US", "DE", "FR", "BR", "IN", "PL", "NL", "ES"],
            index=0,
        )
        plan_type = st.segmented_control(
            "Plan type",
            ["basic", "standard", "premium"],
            default="standard",
        )
        monthly_fee = st.number_input(
            "Monthly fee",
            min_value=0.01,
            value=19.99,
            step=1.0,
        )
        days_active_last_30 = st.slider("Days active last 30", 0, 30, 12)

    with right:
        st.markdown('<p class="section-kicker">Behavior</p>', unsafe_allow_html=True)
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
        avg_session_duration = st.slider(
            "Average session duration",
            min_value=0.0,
            max_value=120.0,
            value=24.5,
            step=0.5,
        )
        feature_usage_score = st.slider("Feature usage score", 0.0, 100.0, 61.0)
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


def render_prediction_result(response: dict[str, Any]) -> None:
    probability_text = format_probability(response.get("churn_probability"))
    risk_band = str(response.get("risk_band") or "")
    risk_class = risk_band_css_class(risk_band)
    risk_label = risk_band_display(risk_band)

    st.markdown(
        f"""
        <div class="panel">
            <p class="section-kicker">Prediction signal</p>
            <p class="probability-readout">{probability_text}</p>
            <span class="risk-pill {risk_class}">{risk_label}</span>
            <p class="small-muted">
                prediction={response.get("churn_prediction", "n/a")}
                threshold={response.get("threshold", "n/a")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    explanation = response.get("explanation")
    if explanation:
        st.caption(str(explanation))


def render_predict_tab(api_base_url: str) -> None:
    with st.form("predict-form"):
        payload = build_prediction_payload()
        submitted = st.form_submit_button("Predict churn")

    if not submitted:
        st.markdown(
            '<div class="panel"><p class="small-muted">'
            "Tune the user profile and run a live API prediction."
            "</p></div>",
            unsafe_allow_html=True,
        )
        return

    result = post_json(api_base_url, "/predict", payload)
    if not result.ok:
        show_api_error(result)
        return

    render_prediction_result(result.data or {})


def render_batch_tab(api_base_url: str) -> None:
    sample = load_batch_sample()
    st.markdown('<p class="section-kicker">Sample batch payload</p>', unsafe_allow_html=True)
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
    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)


def render_monitoring_tab(api_base_url: str) -> None:
    summary = get_json(api_base_url, "/monitoring/summary")
    recent = get_json(api_base_url, "/predictions/recent?limit=20")

    if summary.ok and isinstance(summary.data, dict):
        render_metric_row(summary.data)
        counts = summary.data.get("risk_band_counts", {})
        if isinstance(counts, dict):
            st.bar_chart(pd.Series(counts, name="count"))
    else:
        show_api_error(summary)

    st.markdown('<p class="section-kicker">Recent prediction logs</p>', unsafe_allow_html=True)
    if recent.ok and isinstance(recent.data, dict):
        st.dataframe(
            pd.DataFrame(recent.data.get("items", [])),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Recent predictions are unavailable.")
        if recent.data:
            st.json(recent.data)


def render_model_tab(metadata: ApiResult) -> None:
    if not metadata.ok or not isinstance(metadata.data, dict):
        show_api_error(metadata)
        return

    model_col, generated_col = st.columns(2)
    model_col.metric("Best model", metadata.data.get("best_model") or "n/a")
    generated_col.metric("Generated at", metadata.data.get("generated_at") or "n/a")

    st.markdown('<p class="section-kicker">Validation metrics</p>', unsafe_allow_html=True)
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
            hide_index=True,
        )
    else:
        st.info("Validation metrics are not available yet.")

    st.markdown('<p class="section-kicker">Artifacts</p>', unsafe_allow_html=True)
    st.json(metadata.data.get("artifacts", {}))


def main() -> None:
    st.set_page_config(
        page_title="Churn Monitoring Lab",
        page_icon="chart_with_downwards_trend",
        layout="wide",
    )
    inject_theme()

    default_api_url = normalize_api_base_url(
        os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    )
    with st.sidebar:
        st.markdown("### Control plane")
        api_base_url = normalize_api_base_url(
            st.text_input("API_BASE_URL", value=default_api_url)
        )
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("GET requests cache for 60 seconds. POST requests stay live.")

    health, metadata = render_header(api_base_url)
    if not health.ok:
        show_api_error(health)

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
