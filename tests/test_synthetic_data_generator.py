import pandas as pd

from app.ml.data_quality import validate_churn_dataset
from app.ml.generate_synthetic_data import generate_synthetic_churn_data


def test_generator_is_reproducible_with_seed() -> None:
    first = generate_synthetic_churn_data(n_users=100, seed=123)
    second = generate_synthetic_churn_data(n_users=100, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_generator_creates_expected_columns() -> None:
    data = generate_synthetic_churn_data(n_users=100, seed=42)

    assert validate_churn_dataset(data) == []
    assert len(data) == 100


def test_churn_depends_on_activity_and_usage() -> None:
    data = generate_synthetic_churn_data(n_users=3000, seed=42)

    churned = data[data["churn"] == 1]
    retained = data[data["churn"] == 0]

    assert churned["days_active_last_30"].mean() < (
        retained["days_active_last_30"].mean()
    )
    assert churned["feature_usage_score"].mean() < (
        retained["feature_usage_score"].mean()
    )
    assert churned["payments_failed_last_90"].mean() > (
        retained["payments_failed_last_90"].mean()
    )
