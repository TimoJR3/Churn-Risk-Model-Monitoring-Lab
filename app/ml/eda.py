from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.ml.data_loader import load_churn_data
from app.ml.features import add_feature_engineering


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = DOCS_DIR / "eda_report.md"
TARGET_COLUMN = "churn"


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    table = data.copy()
    table = table.reset_index()
    table.columns = [str(column) for column in table.columns]

    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = []

    for _, row in table.iterrows():
        values = [str(value) for value in row.to_list()]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def detect_outliers_iqr(data: pd.DataFrame) -> pd.Series:
    numeric = data.select_dtypes(include="number").drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )
    counts = {}

    for column in numeric.columns:
        q1 = numeric[column].quantile(0.25)
        q3 = numeric[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        counts[column] = numeric[column].lt(lower).sum()
        counts[column] += numeric[column].gt(upper).sum()

    return pd.Series(counts).sort_values(ascending=False)


def build_eda_summary(data: pd.DataFrame) -> str:
    engineered = add_feature_engineering(data)
    rows, columns = data.shape
    churn_rate = data[TARGET_COLUMN].mean()
    missing = data.isna().sum()
    outliers = detect_outliers_iqr(data)

    numeric_data = data.select_dtypes(include="number")
    numeric_summary = numeric_data.describe().T[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    categorical_summary = data[["country", "plan_type"]].describe().T

    feature_columns = [
        "days_active_last_30",
        "sessions_last_30",
        "support_tickets_last_30",
        "payments_failed_last_90",
        "avg_session_duration",
        "feature_usage_score",
        "last_login_days_ago",
        "activity_score",
        "payment_risk_score",
        "days_since_signup",
        "usage_per_session",
        "support_intensity",
    ]
    churn_relation = engineered.groupby(TARGET_COLUMN)[feature_columns].mean()
    churn_relation = churn_relation.T.round(2)

    plan_churn = (
        data.groupby("plan_type")[TARGET_COLUMN]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
        .round(3)
    )

    country_churn = (
        data.groupby("country")[TARGET_COLUMN]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
        .round(3)
    )

    report = f"""# EDA Report

## Dataset Overview

- Rows: `{rows}`
- Columns: `{columns}`
- Target: `churn`
- Churn rate: `{churn_rate:.3f}`

## Missing Values

{dataframe_to_markdown(missing.to_frame("missing_count"))}

Missing values appear in behavioral features. This is realistic for
event data: some session, support, or usage events can be absent.
The preprocessing pipeline handles them with median or most frequent
imputation.

## Outliers

{dataframe_to_markdown(outliers.to_frame("outlier_count"))}

Outliers are expected mostly in `sessions_last_30` and
`avg_session_duration`. They represent power users, bots, or noisy
event tracking. They are not removed at this stage, so the future
baseline model can see realistic edge cases.

## Numeric Distributions

{dataframe_to_markdown(numeric_summary.round(2))}

## Categorical Distributions

{dataframe_to_markdown(categorical_summary)}

## Feature Relationship With Churn

{dataframe_to_markdown(churn_relation)}

Churned users are less active on average, use the product less,
log in less recently, and have more failed payments. This matches
subscription-product logic: low engagement and payment friction often
come before churn.

## Churn By Plan

{dataframe_to_markdown(plan_churn)}

## Churn By Country

{dataframe_to_markdown(country_churn)}

## Leakage Check

Potential leakage features:

- `churn` must not be used as a feature because it is the target.
- `user_id` has no business signal and can let a model memorize rows.
- Features created after the churn date must not be added to training.
- `model_predictions`, `model_metrics`, and `drift_metrics` must not
  be used for training because they are created after model inference.

The preprocessing step drops `user_id` and `signup_date`. Instead of
the raw signup date, it uses the engineered `days_since_signup` feature.
"""
    return report


def save_eda_report(source: str = "csv") -> Path:
    data = load_churn_data(source=source)
    report = build_eda_summary(data)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA report.")
    parser.add_argument("--source", choices=["csv", "postgres"], default="csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = save_eda_report(source=args.source)
    print(f"EDA report: {path}")


if __name__ == "__main__":
    main()
