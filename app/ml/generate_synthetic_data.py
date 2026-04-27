from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.data_quality import assert_churn_dataset_is_valid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

COUNTRIES = np.array(["US", "DE", "FR", "BR", "IN", "PL", "NL", "ES"])
COUNTRY_PROBS = np.array([0.24, 0.14, 0.12, 0.10, 0.16, 0.08, 0.08, 0.08])
PLAN_TYPES = np.array(["basic", "standard", "premium"])
PLAN_PROBS = np.array([0.48, 0.36, 0.16])
PLAN_FEES = {
    "basic": 9.99,
    "standard": 19.99,
    "premium": 39.99,
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def add_missing_values(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = data.copy()
    columns = [
        "avg_session_duration",
        "feature_usage_score",
        "support_tickets_last_30",
    ]

    for column in columns:
        mask = rng.random(len(result)) < 0.04
        result.loc[mask, column] = np.nan

    return result


def add_outliers(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = data.copy()
    outlier_count = max(1, int(len(result) * 0.01))
    outlier_ids = rng.choice(result.index, size=outlier_count, replace=False)

    result.loc[outlier_ids, "sessions_last_30"] = rng.integers(
        120,
        260,
        size=outlier_count,
    )
    result.loc[outlier_ids, "avg_session_duration"] = rng.uniform(
        180,
        420,
        size=outlier_count,
    ).round(2)

    return result


def generate_synthetic_churn_data(
    n_users: int = 2000,
    seed: int = 42,
    snapshot_date: str = "2026-04-24",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    user_ids = np.arange(1, n_users + 1)
    signup_offsets = rng.integers(30, 900, size=n_users)
    snapshot = pd.to_datetime(snapshot_date)
    signup_dates = snapshot - pd.to_timedelta(signup_offsets, unit="D")

    countries = rng.choice(COUNTRIES, size=n_users, p=COUNTRY_PROBS)
    plan_types = rng.choice(PLAN_TYPES, size=n_users, p=PLAN_PROBS)
    monthly_fee = np.array([PLAN_FEES[plan] for plan in plan_types])
    monthly_fee = monthly_fee + rng.normal(0, 1.5, size=n_users)
    monthly_fee = np.clip(monthly_fee, 4.99, None).round(2)

    engagement = rng.beta(2.2, 2.8, size=n_users)
    days_active = rng.binomial(30, engagement)
    sessions = rng.poisson(days_active * rng.uniform(1.2, 3.2, size=n_users))
    tickets = rng.poisson(0.35 + (1 - engagement) * 1.2)
    failed_payments = rng.poisson(0.12 + (1 - engagement) * 0.55)
    failed_payments = np.clip(failed_payments, 0, 6)

    avg_duration = rng.normal(18 + engagement * 35, 8, size=n_users)
    avg_duration = np.clip(avg_duration, 1, None).round(2)

    usage_score = 100 * (
        0.50 * engagement
        + 0.30 * (days_active / 30)
        + 0.20 * np.minimum(sessions / 90, 1)
    )
    usage_score = np.clip(usage_score + rng.normal(0, 6, n_users), 0, 100)
    usage_score = usage_score.round(2)

    last_login_lambda = np.clip(18 - days_active * 0.45, 1, 35)
    last_login_days = rng.poisson(last_login_lambda)
    last_login_days = np.clip(last_login_days, 0, 90)

    logit = (
        -1.10
        - 0.09 * days_active
        - 0.018 * sessions
        - 0.030 * usage_score
        + 0.42 * failed_payments
        + 0.18 * tickets
        + 0.055 * last_login_days
        + np.where(plan_types == "basic", 0.22, 0.0)
        + np.where(plan_types == "premium", -0.18, 0.0)
    )
    churn_probability = sigmoid(logit)
    churn = rng.binomial(1, churn_probability)

    data = pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_date": signup_dates.date,
            "country": countries,
            "plan_type": plan_types,
            "monthly_fee": monthly_fee,
            "days_active_last_30": days_active,
            "sessions_last_30": sessions,
            "support_tickets_last_30": tickets,
            "payments_failed_last_90": failed_payments,
            "avg_session_duration": avg_duration,
            "feature_usage_score": usage_score,
            "last_login_days_ago": last_login_days,
            "churn": churn,
        }
    )

    data = add_outliers(data, rng)
    data = add_missing_values(data, rng)
    assert_churn_dataset_is_valid(data)

    return data


def save_dataset(data: pd.DataFrame) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DATA_DIR / "synthetic_churn_dataset.csv"
    users_path = PROCESSED_DATA_DIR / "users.csv"
    features_path = PROCESSED_DATA_DIR / "user_features.csv"

    data.to_csv(raw_path, index=False)

    users = data[
        [
            "user_id",
            "signup_date",
            "country",
            "plan_type",
            "monthly_fee",
            "churn",
        ]
    ]
    users.to_csv(users_path, index=False)

    features = data[
        [
            "user_id",
            "days_active_last_30",
            "sessions_last_30",
            "support_tickets_last_30",
            "payments_failed_last_90",
            "avg_session_duration",
            "feature_usage_score",
            "last_login_days_ago",
        ]
    ].copy()
    features.insert(1, "snapshot_date", "2026-04-24")
    features.to_csv(features_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic churn dataset."
    )
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-date", type=str, default="2026-04-24")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = generate_synthetic_churn_data(
        n_users=args.n_users,
        seed=args.seed,
        snapshot_date=args.snapshot_date,
    )
    save_dataset(data)

    churn_rate = data["churn"].mean()
    print(f"Generated rows: {len(data)}")
    print(f"Churn rate: {churn_rate:.3f}")
    print(f"Raw dataset: {RAW_DATA_DIR / 'synthetic_churn_dataset.csv'}")
    print(f"Users seed: {PROCESSED_DATA_DIR / 'users.csv'}")
    print(f"Features seed: {PROCESSED_DATA_DIR / 'user_features.csv'}")


if __name__ == "__main__":
    main()
