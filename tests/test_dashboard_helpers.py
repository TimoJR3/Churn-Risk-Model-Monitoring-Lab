from dashboard.api_client import (
    DEFAULT_API_BASE_URL,
    format_probability,
    normalize_api_base_url,
    risk_band_display,
)


def test_format_probability_handles_values_and_missing_data() -> None:
    assert format_probability(0.582) == "58.2%"
    assert format_probability(1) == "100.0%"
    assert format_probability(None) == "n/a"


def test_risk_band_display_returns_readable_labels() -> None:
    assert risk_band_display("low") == "LOW RISK"
    assert risk_band_display("medium") == "MEDIUM RISK"
    assert risk_band_display("high") == "HIGH RISK"
    assert risk_band_display("unexpected") == "UNKNOWN RISK"
    assert risk_band_display(None) == "UNKNOWN RISK"


def test_normalize_api_base_url_uses_default_and_strips_slash() -> None:
    assert normalize_api_base_url(None) == DEFAULT_API_BASE_URL
    assert normalize_api_base_url("  ") == DEFAULT_API_BASE_URL
    assert normalize_api_base_url("http://api:8000/") == "http://api:8000"
