import numpy as np

from app.ml.features import add_feature_engineering
from app.ml.generate_synthetic_data import generate_synthetic_churn_data
from app.ml.preprocessing import (
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    make_train_validation_split,
    transform_to_dataframe,
)


def test_feature_engineering_adds_expected_columns() -> None:
    data = generate_synthetic_churn_data(n_users=100, seed=42)
    result = add_feature_engineering(data)

    expected_columns = {
        "activity_score",
        "payment_risk_score",
        "engagement_level",
        "days_since_signup",
        "usage_per_session",
        "support_intensity",
    }

    assert expected_columns.issubset(result.columns)
    assert result["activity_score"].between(0, 100).all()
    assert result["payment_risk_score"].between(0, 100).all()


def test_train_validation_split_is_stratified() -> None:
    data = generate_synthetic_churn_data(n_users=500, seed=42)
    train_x, valid_x, train_y, valid_y = make_train_validation_split(data)

    assert len(train_x) == 400
    assert len(valid_x) == 100
    assert abs(train_y.mean() - valid_y.mean()) < 0.05


def test_preprocessing_pipeline_removes_missing_values() -> None:
    data = generate_synthetic_churn_data(n_users=300, seed=42)
    data = add_feature_engineering(data)
    train_x, valid_x, _, _ = make_train_validation_split(data)

    train_x = train_x.drop(columns=["user_id", "signup_date"])
    valid_x = valid_x.drop(columns=["user_id", "signup_date"])

    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(train_x)

    transformed = transform_to_dataframe(preprocessor, valid_x)

    assert TARGET_COLUMN not in transformed.columns
    assert not np.isnan(transformed.to_numpy()).any()
