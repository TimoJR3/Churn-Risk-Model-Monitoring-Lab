import pandas as pd

from app.ml.generate_synthetic_data import generate_synthetic_churn_data
from app.ml.features import add_feature_engineering
from app.ml.preprocessing import DROP_COLUMNS, make_train_validation_split
from app.ml.training import (
    build_candidate_models,
    get_feature_importance,
    run_cross_validation,
    select_best_model,
)


def test_build_candidate_models_contains_required_models() -> None:
    models = build_candidate_models(random_state=42)

    assert set(models) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }


def test_cross_validation_returns_expected_metrics() -> None:
    data = generate_synthetic_churn_data(n_users=250, seed=42)
    data = add_feature_engineering(data)
    train_x, _, train_y, _ = make_train_validation_split(data)
    train_x = train_x.drop(columns=DROP_COLUMNS)
    models = build_candidate_models(random_state=42)
    models = {"logistic_regression": models["logistic_regression"]}

    results = run_cross_validation(
        train_x=train_x,
        train_y=train_y,
        models=models,
        n_splits=3,
        random_state=42,
    )

    assert "logistic_regression" in results
    assert 0 <= results["logistic_regression"]["cv_roc_auc_mean"] <= 1
    assert 0 <= results["logistic_regression"]["cv_f1_mean"] <= 1


def test_select_best_model_uses_roc_auc_then_f1() -> None:
    results = {
        "model_a": {"cv_roc_auc_mean": 0.80, "cv_f1_mean": 0.50},
        "model_b": {"cv_roc_auc_mean": 0.80, "cv_f1_mean": 0.60},
    }

    assert select_best_model(results) == "model_b"


def test_feature_importance_handles_unsupported_model() -> None:
    class DummyModel:
        pass

    result = get_feature_importance(
        model=DummyModel(),
        feature_names=["feature_a", "feature_b"],
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "importance"]
    assert result.empty
