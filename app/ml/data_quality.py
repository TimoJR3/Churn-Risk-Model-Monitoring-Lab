from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "user_id",
    "signup_date",
    "country",
    "plan_type",
    "monthly_fee",
    "days_active_last_30",
    "sessions_last_30",
    "support_tickets_last_30",
    "payments_failed_last_90",
    "avg_session_duration",
    "feature_usage_score",
    "last_login_days_ago",
    "churn",
}

ALLOWED_PLAN_TYPES = {"basic", "standard", "premium"}
ALLOWED_COUNTRIES = {"US", "DE", "FR", "BR", "IN", "PL", "NL", "ES"}


def validate_churn_dataset(data: pd.DataFrame) -> list[str]:
    errors = []

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        errors.append(f"Missing columns: {sorted(missing_columns)}")
        return errors

    if data["user_id"].duplicated().any():
        errors.append("user_id must be unique.")

    if data["user_id"].isna().any():
        errors.append("user_id must not contain missing values.")

    if not set(data["plan_type"].dropna()).issubset(ALLOWED_PLAN_TYPES):
        errors.append("plan_type contains unexpected values.")

    if not set(data["country"].dropna()).issubset(ALLOWED_COUNTRIES):
        errors.append("country contains unexpected values.")

    if data["monthly_fee"].dropna().le(0).any():
        errors.append("monthly_fee must be positive.")

    if data["days_active_last_30"].dropna().lt(0).any():
        errors.append("days_active_last_30 must be non-negative.")

    if data["days_active_last_30"].dropna().gt(30).any():
        errors.append("days_active_last_30 must be less than or equal to 30.")

    if data["sessions_last_30"].dropna().lt(0).any():
        errors.append("sessions_last_30 must be non-negative.")

    if data["payments_failed_last_90"].dropna().lt(0).any():
        errors.append("payments_failed_last_90 must be non-negative.")

    if data["feature_usage_score"].dropna().lt(0).any():
        errors.append("feature_usage_score must be non-negative.")

    if data["feature_usage_score"].dropna().gt(100).any():
        errors.append("feature_usage_score must be less than or equal to 100.")

    churn_values = set(data["churn"].dropna().unique())
    if not churn_values.issubset({0, 1, False, True}):
        errors.append("churn must be binary.")

    churn_rate = data["churn"].mean()
    if churn_rate < 0.05 or churn_rate > 0.60:
        errors.append("churn rate is outside the expected range.")

    return errors


def assert_churn_dataset_is_valid(data: pd.DataFrame) -> None:
    errors = validate_churn_dataset(data)
    if errors:
        raise ValueError("Data quality checks failed: " + "; ".join(errors))
