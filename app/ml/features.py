from __future__ import annotations

import numpy as np
import pandas as pd


SNAPSHOT_DATE = "2026-04-24"


def add_feature_engineering(
    data: pd.DataFrame,
    snapshot_date: str = SNAPSHOT_DATE,
) -> pd.DataFrame:
    result = data.copy()
    snapshot = pd.to_datetime(snapshot_date)
    signup_date = pd.to_datetime(result["signup_date"])

    active_ratio = result["days_active_last_30"] / 30
    session_ratio = np.minimum(result["sessions_last_30"] / 90, 1)
    usage_median = result["feature_usage_score"].median()
    usage_ratio = result["feature_usage_score"].fillna(usage_median) / 100

    result["activity_score"] = (
        0.45 * active_ratio
        + 0.35 * session_ratio
        + 0.20 * usage_ratio
    ) * 100

    result["payment_risk_score"] = np.minimum(
        result["payments_failed_last_90"] / 3,
        1,
    ) * 100

    result["engagement_level"] = pd.cut(
        result["activity_score"],
        bins=[-np.inf, 35, 70, np.inf],
        labels=["low", "medium", "high"],
    ).astype("object")

    result["days_since_signup"] = (snapshot - signup_date).dt.days

    result["usage_per_session"] = (
        result["feature_usage_score"].fillna(usage_median)
        / result["sessions_last_30"].replace(0, np.nan)
    )

    tickets = result["support_tickets_last_30"].fillna(0)
    result["support_intensity"] = (
        tickets / (result["sessions_last_30"] + 1)
    )

    return result
