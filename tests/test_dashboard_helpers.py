from dashboard.api_client import DEFAULT_API_BASE_URL, normalize_api_base_url
from dashboard.helpers import (
    format_probability,
    prepare_batch_results_table,
    risk_band_display,
    risk_band_interpretation,
    summarize_batch_results,
)


def test_format_probability_handles_values_and_missing_data() -> None:
    assert format_probability(0.582) == "58.2%"
    assert format_probability(1) == "100.0%"
    assert format_probability(None) == "н/д"


def test_risk_band_display_returns_readable_labels() -> None:
    assert risk_band_display("low") == "низкий"
    assert risk_band_display("medium") == "средний"
    assert risk_band_display("high") == "высокий"
    assert risk_band_display("unexpected") == "неизвестно"
    assert risk_band_display(None) == "неизвестно"


def test_risk_band_interpretation_is_human_readable() -> None:
    assert "Существенных признаков риска" in risk_band_interpretation("low")
    assert "умеренные признаки" in risk_band_interpretation("medium")
    assert "признаки возможного оттока" in risk_band_interpretation("high")


def test_prepare_batch_results_table_uses_russian_columns() -> None:
    table = prepare_batch_results_table(
        [
            {
                "churn_probability": 0.73,
                "churn_prediction": 1,
                "risk_band": "high",
            }
        ],
        [{"user_id": 1001}],
    )

    assert list(table.columns) == [
        "ID клиента",
        "Вероятность оттока",
        "Прогноз",
        "Уровень риска",
    ]
    assert table.iloc[0].to_dict() == {
        "ID клиента": 1001,
        "Вероятность оттока": "73.0%",
        "Прогноз": "отток",
        "Уровень риска": "высокий",
    }


def test_summarize_batch_results_handles_empty_items() -> None:
    assert summarize_batch_results([]) == {
        "row_count": 0,
        "average_probability": 0.0,
        "high_risk_count": 0,
        "high_risk_share": 0.0,
    }


def test_normalize_api_base_url_uses_default_and_strips_slash() -> None:
    assert normalize_api_base_url(None) == DEFAULT_API_BASE_URL
    assert normalize_api_base_url("  ") == DEFAULT_API_BASE_URL
    assert normalize_api_base_url("http://api:8000/") == "http://api:8000"
