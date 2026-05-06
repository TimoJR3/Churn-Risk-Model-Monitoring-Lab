from __future__ import annotations

import json
import os
from datetime import date
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.api_client import (
    DEFAULT_API_BASE_URL,
    ApiResult,
    get_json,
    normalize_api_base_url,
    post_json,
)
from dashboard.helpers import (
    format_probability,
    prediction_label,
    prepare_batch_results_table,
    prepare_recent_predictions_table,
    prepare_risk_band_counts_table,
    risk_band_css_class,
    risk_band_display,
    risk_band_interpretation,
    summarize_batch_results,
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
                radial-gradient(
                    circle at 12% 12%,
                    rgba(66,246,255,0.18),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 78% 8%,
                    rgba(255,92,138,0.16),
                    transparent 26%
                ),
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
            background: linear-gradient(
                145deg,
                rgba(16,28,46,0.92),
                rgba(7,13,24,0.82)
            );
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
            margin-bottom: 1rem;
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


def load_batch_sample() -> tuple[list[dict[str, Any]], str | None]:
    try:
        payload = json.loads(BATCH_SAMPLE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], (
            "Sample JSON не найден. Нужен файл "
            "`data/sample/predict_batch_sample.json`."
        )
    except JSONDecodeError:
        return [], (
            "Файл `data/sample/predict_batch_sample.json` содержит "
            "невалидный JSON."
        )

    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        return [], "Sample JSON должен быть списком объектов клиентов."

    return payload, None


def show_startup_commands() -> None:
    st.code(
        "docker compose up --build\n\n"
        "# или локально без PostgreSQL:\n"
        "$env:SAVE_PREDICTIONS=\"false\"\n"
        "uvicorn app.api.main:app --reload\n"
        "streamlit run dashboard/app.py",
        language="powershell",
    )


def show_api_error(
    result: ApiResult,
    title: str = "Не удалось получить данные",
) -> None:
    st.error(f"{title}. {result.error or 'Проверьте backend API.'}")

    detail = result.data.get("detail") if isinstance(result.data, dict) else None
    error_code = detail.get("error") if isinstance(detail, dict) else None
    if error_code == "prediction_log_unavailable":
        st.warning(
            "Логи прогнозов сейчас недоступны. Проверьте PostgreSQL или "
            "запустите API с `SAVE_PREDICTIONS=false`, если нужна только "
            "демонстрация inference."
        )
    elif error_code == "model_artifacts_unavailable":
        st.warning(
            "Artifacts модели не найдены. Сначала выполните обучение: "
            "`python -m app.ml.training --source csv --n-splits 3`."
        )
    else:
        st.info("Запустите backend и обновите страницу dashboard.")

    show_startup_commands()
    if result.data:
        with st.expander("Показать технические детали"):
            st.json(result.data)


def render_header(api_base_url: str) -> tuple[ApiResult, ApiResult]:
    health = get_json(api_base_url, "/health")
    metadata = get_json(api_base_url, "/model/metadata")
    online = health.ok
    model_name = "н/д"
    threshold = "н/д"

    if metadata.ok and isinstance(metadata.data, dict):
        model_name = str(metadata.data.get("best_model") or "н/д")
        threshold = str(metadata.data.get("threshold", "н/д"))

    signal_class = "signal-online" if online else "signal-offline"
    signal_text = "API онлайн" if online else "API недоступен"

    st.markdown(
        f"""
        <section class="command-deck">
            <p class="section-kicker">Мониторинг оттока</p>
            <h1 class="command-title">Лаборатория мониторинга churn-модели</h1>
            <p class="command-subtitle">
                Демо-сервис для прогноза оттока, batch scoring,
                мониторинга качества модели и drift-сигналов.
            </p>
            <div class="signal-strip">
                <span class="signal-pill {signal_class}">{signal_text}</span>
                <span class="signal-pill">Модель: {model_name}</span>
                <span class="signal-pill">Порог: {threshold}</span>
                <span class="signal-pill">Backend: {api_base_url}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not online:
        st.warning(
            "Dashboard работает, но backend API сейчас недоступен. "
            "Данные появятся после запуска FastAPI."
        )
        show_startup_commands()

    return health, metadata


def render_metric_row(summary: dict[str, Any]) -> None:
    total, average, high_risk, updated = st.columns(4)
    total.metric("Всего прогнозов", summary.get("total_predictions", 0))
    average.metric(
        "Средняя вероятность оттока",
        format_probability(summary.get("average_probability")),
    )
    high_risk.metric(
        "Доля высокого риска",
        format_probability(summary.get("high_risk_share")),
    )
    updated.metric("Последнее обновление", "сейчас")


def build_prediction_payload() -> dict[str, Any]:
    st.subheader("Профиль клиента")
    left, right = st.columns(2)

    with left:
        user_id = st.text_input("ID клиента", value="1001")
        signup_date = st.date_input("Дата регистрации", value=date(2025, 9, 1))
        country = st.selectbox(
            "Страна",
            ["US", "DE", "FR", "BR", "IN", "PL", "NL", "ES"],
            index=0,
        )
        plan_type = st.segmented_control(
            "Тариф",
            ["basic", "standard", "premium"],
            default="standard",
        )
        monthly_fee = st.number_input(
            "Ежемесячный платёж",
            min_value=0.01,
            value=19.99,
            step=1.0,
        )
        days_active_last_30 = st.slider("Активных дней за 30 дней", 0, 30, 12)

    with right:
        st.subheader("Активность")
        sessions_last_30 = st.number_input(
            "Сессий за 30 дней",
            min_value=0,
            value=30,
            step=1,
        )
        support_tickets_last_30 = st.number_input(
            "Обращений в поддержку за 30 дней",
            min_value=0,
            value=1,
            step=1,
        )
        payments_failed_last_90 = st.number_input(
            "Неуспешных платежей за 90 дней",
            min_value=0,
            value=0,
            step=1,
        )
        avg_session_duration = st.slider(
            "Средняя длительность сессии",
            min_value=0.0,
            max_value=120.0,
            value=24.5,
            step=0.5,
        )
        feature_usage_score = st.slider(
            "Индекс использования функций",
            0.0,
            100.0,
            61.0,
        )
        last_login_days_ago = st.number_input(
            "Дней с последнего входа",
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
            <p class="section-kicker">Результат inference</p>
            <p class="probability-readout">{probability_text}</p>
            <span class="risk-pill {risk_class}">Уровень риска: {risk_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability, prediction, risk, threshold = st.columns(4)
    probability.metric("Вероятность оттока", probability_text)
    prediction.metric(
        "Класс прогноза",
        prediction_label(response.get("churn_prediction")),
    )
    risk.metric("Уровень риска", risk_label)
    threshold.metric(
        "Порог классификации",
        format_probability(response.get("threshold")),
    )

    st.info(f"Интерпретация: {risk_band_interpretation(risk_band)}")
    with st.expander("Показать технический ответ API"):
        st.json(response)


def render_predict_tab(api_base_url: str) -> None:
    st.header("Прогноз оттока")
    st.caption(
        "Заполните тестовый профиль клиента и отправьте его в FastAPI endpoint "
        "`/predict`."
    )

    with st.form("predict-form"):
        payload = build_prediction_payload()
        submitted = st.form_submit_button("Рассчитать риск оттока")

    if not submitted:
        st.info(
            "Измените признаки клиента или используйте значения по умолчанию, "
            "затем нажмите «Рассчитать риск оттока»."
        )
        return

    result = post_json(api_base_url, "/predict", payload)
    if not result.ok:
        show_api_error(result, "Прогноз не выполнен")
        return

    render_prediction_result(result.data or {})


def render_batch_tab(api_base_url: str) -> None:
    st.header("Пакетный прогноз оттока")
    st.caption(
        "Запустите scoring для нескольких тестовых клиентов и сравните "
        "распределение риска."
    )

    sample, sample_error = load_batch_sample()
    if sample_error:
        st.error(sample_error)
        st.info(
            "Создайте `data/sample/predict_batch_sample.json` со списком "
            "объектов клиентов для endpoint `/predict/batch`."
        )
        return

    st.success(f"Пример запроса загружен: {len(sample)} клиента.")
    with st.expander("Показать sample JSON"):
        st.json(sample)

    if not st.button("Запустить пакетный прогноз", type="primary"):
        return

    result = post_json(api_base_url, "/predict/batch", sample)
    if not result.ok:
        show_api_error(result, "Пакетный прогноз не выполнен")
        return

    payload = result.data or {}
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        st.warning("API вернул пустой список результатов.")
        with st.expander("Показать технический ответ API"):
            st.json(payload)
        return

    summary = summarize_batch_results(items)
    row_count, average, high_count, high_share = st.columns(4)
    row_count.metric("Клиентов обработано", payload.get("row_count", len(items)))
    average.metric(
        "Средняя вероятность оттока",
        format_probability(summary["average_probability"]),
    )
    high_count.metric("Клиентов с высоким риском", summary["high_risk_count"])
    high_share.metric(
        "Доля высокого риска",
        format_probability(summary["high_risk_share"]),
    )

    st.dataframe(
        prepare_batch_results_table(items, sample),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Показать технический ответ API"):
        st.json(payload)


def render_monitoring_tab(api_base_url: str) -> None:
    st.header("Мониторинг предсказаний")
    st.caption(
        "Здесь отображается сводка по последним логам прогнозов и "
        "распределение риска."
    )

    summary = get_json(api_base_url, "/monitoring/summary")
    recent = get_json(api_base_url, "/predictions/recent?limit=20")

    if summary.ok and isinstance(summary.data, dict):
        render_metric_row(summary.data)
        counts = summary.data.get("risk_band_counts", {})
        if isinstance(counts, dict):
            counts_table = prepare_risk_band_counts_table(counts)
            left, right = st.columns([1, 2])
            left.dataframe(counts_table, use_container_width=True, hide_index=True)
            right.bar_chart(counts_table.set_index("Уровень риска"))
    else:
        show_api_error(summary, "Сводка мониторинга недоступна")

    st.subheader("Последние прогнозы")
    if not recent.ok or not isinstance(recent.data, dict):
        show_api_error(recent, "Логи прогнозов недоступны")
        return

    items = recent.data.get("items", [])
    if not items:
        st.info(
            "Пока нет сохранённых прогнозов. Перейдите во вкладку «Прогноз» "
            "и рассчитайте риск для тестового клиента."
        )
        return

    st.dataframe(
        prepare_recent_predictions_table(items),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Показать технические детали логов прогнозов"):
        st.json(items)


def _metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "н/д"


def render_model_tab(metadata: ApiResult) -> None:
    st.header("Информация о модели")
    if not metadata.ok or not isinstance(metadata.data, dict):
        show_api_error(metadata, "Metadata модели недоступна")
        return

    data = metadata.data
    validation_metrics = data.get("validation_metrics", {})
    artifacts = data.get("artifacts", {})

    model_col, threshold_col, version_col = st.columns(3)
    model_col.metric("Тип модели", data.get("best_model") or "н/д")
    threshold_col.metric(
        "Порог классификации",
        format_probability(data.get("threshold")),
    )
    version_col.metric("Версия / имя artifacts", data.get("best_model") or "н/д")

    st.subheader("Метрики валидации")
    if isinstance(validation_metrics, dict) and validation_metrics:
        roc_auc, precision, recall, f1 = st.columns(4)
        roc_auc.metric("ROC AUC", _metric_value(validation_metrics, "roc_auc"))
        precision.metric("Precision", _metric_value(validation_metrics, "precision"))
        recall.metric("Recall", _metric_value(validation_metrics, "recall"))
        f1.metric("F1", _metric_value(validation_metrics, "f1"))
    else:
        st.info(
            "Метрики не найдены. Проверьте `artifacts/metrics.json` "
            "или переобучите модель."
        )

    st.subheader("Статус artifacts")
    if isinstance(artifacts, dict) and artifacts:
        artifact_rows = [
            {
                "Файл artifacts": name,
                "Статус": "найден" if details.get("exists") else "не найден",
                "Путь": details.get("path", "н/д"),
            }
            for name, details in artifacts.items()
            if isinstance(details, dict)
        ]
        st.dataframe(
            pd.DataFrame(artifact_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Статус artifacts недоступен.")

    st.info(
        "Как читать результат: вероятность оттока — это оценка риска по "
        "текущим признакам клиента. Она помогает приоритизировать действия, "
        "но не является бизнес-решением сама по себе."
    )
    with st.expander("Показать полный ответ /model/metadata"):
        st.json(data)


def main() -> None:
    st.set_page_config(
        page_title="Лаборатория churn-мониторинга",
        layout="wide",
    )
    inject_theme()

    default_api_url = normalize_api_base_url(
        os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    )
    with st.sidebar:
        st.markdown("### Настройки")
        api_base_url = normalize_api_base_url(
            st.text_input("Backend API URL", value=default_api_url)
        )
        if st.button("Обновить данные", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("GET-запросы кешируются на 60 секунд. POST-запросы не кешируются.")

    _health, metadata = render_header(api_base_url)

    predict_tab, batch_tab, monitoring_tab, model_tab = st.tabs(
        ["Прогноз", "Пакетный прогноз", "Мониторинг", "Модель"]
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
