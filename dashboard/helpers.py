from __future__ import annotations

from typing import Any

import pandas as pd

RISK_BAND_LABELS = {
    "low": "низкий",
    "medium": "средний",
    "high": "высокий",
}
RISK_BAND_CLASSES = {
    "low": "risk-low",
    "medium": "risk-medium",
    "high": "risk-high",
}
RISK_BAND_ORDER = ("low", "medium", "high")


def format_probability(value: float | int | None) -> str:
    """Format a probability as a dashboard-friendly percentage."""
    if value is None:
        return "н/д"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "н/д"


def risk_band_display(value: str | None) -> str:
    """Return a Russian risk band label."""
    if value is None:
        return "неизвестно"
    return RISK_BAND_LABELS.get(value.lower(), "неизвестно")


def risk_band_css_class(value: str | None) -> str:
    """Return a stable CSS class for a risk band."""
    if value is None:
        return "risk-unknown"
    return RISK_BAND_CLASSES.get(value.lower(), "risk-unknown")


def risk_band_interpretation(value: str | None) -> str:
    """Return a short business-readable interpretation for a risk band."""
    match (value or "").lower():
        case "high":
            return (
                "Клиент демонстрирует признаки возможного оттока. "
                "Стоит проверить недавнюю активность, платежные проблемы "
                "и историю обращений в поддержку."
            )
        case "medium":
            return (
                "Есть умеренные признаки риска. Клиента стоит наблюдать "
                "и проверить, не ухудшается ли его продуктовая активность."
            )
        case "low":
            return (
                "Существенных признаков риска не выявлено. Клиент выглядит "
                "стабильным по текущим входным признакам."
            )
    return "Интерпретация недоступна: API вернул неизвестный уровень риска."


def prediction_label(value: int | str | None) -> str:
    """Return a Russian label for a binary churn prediction."""
    if value in (1, "1", True):
        return "отток"
    if value in (0, "0", False):
        return "не отток"
    return "н/д"


def prepare_batch_results_table(
    items: list[dict[str, Any]],
    source_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a recruiter-friendly batch scoring table with Russian columns."""
    rows = []
    source_rows = source_rows or []
    for index, item in enumerate(items):
        source = source_rows[index] if index < len(source_rows) else {}
        rows.append(
            {
                "ID клиента": source.get("user_id", "н/д"),
                "Вероятность оттока": format_probability(
                    item.get("churn_probability")
                ),
                "Прогноз": prediction_label(item.get("churn_prediction")),
                "Уровень риска": risk_band_display(
                    item.get("risk_band")
                    if isinstance(item.get("risk_band"), str)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_batch_results(items: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize batch prediction response items for metric cards."""
    probabilities = [
        float(item["churn_probability"])
        for item in items
        if item.get("churn_probability") is not None
    ]
    high_risk_count = sum(
        1 for item in items if str(item.get("risk_band", "")).lower() == "high"
    )
    row_count = len(items)
    return {
        "row_count": row_count,
        "average_probability": (
            sum(probabilities) / len(probabilities) if probabilities else 0.0
        ),
        "high_risk_count": high_risk_count,
        "high_risk_share": high_risk_count / row_count if row_count else 0.0,
    }


def prepare_risk_band_counts_table(counts: dict[str, int]) -> pd.DataFrame:
    """Build a stable risk-band count table with Russian labels."""
    rows = [
        {
            "Уровень риска": risk_band_display(key),
            "Количество": int(counts.get(key, 0)),
        }
        for key in RISK_BAND_ORDER
    ]
    for key, value in counts.items():
        if key not in RISK_BAND_ORDER:
            rows.append(
                {
                    "Уровень риска": risk_band_display(key),
                    "Количество": int(value),
                }
            )
    return pd.DataFrame(rows)


def prepare_recent_predictions_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a privacy-safe recent predictions table without raw input features."""
    rows = []
    for item in items:
        rows.append(
            {
                "Дата": item.get("created_at", "н/д"),
                "ID hash": item.get("user_id_hash") or "н/д",
                "Вероятность оттока": format_probability(
                    item.get("churn_probability")
                ),
                "Прогноз": prediction_label(item.get("churn_prediction")),
                "Уровень риска": risk_band_display(
                    item.get("risk_band")
                    if isinstance(item.get("risk_band"), str)
                    else None
                ),
                "Порог": format_probability(item.get("threshold")),
                "Модель": item.get("model_version") or "н/д",
            }
        )
    return pd.DataFrame(rows)
