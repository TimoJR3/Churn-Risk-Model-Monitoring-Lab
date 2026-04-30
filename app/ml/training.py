from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.ml.data_loader import load_churn_data
from app.ml.features import add_feature_engineering
from app.ml.preprocessing import (
    DROP_COLUMNS,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    make_train_validation_split,
    transform_to_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"
TRAINED_MODEL_PATH = ARTIFACTS_DIR / "trained_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
MODEL_CARD_PATH = DOCS_DIR / "model_card.md"

SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


def build_candidate_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=10,
            n_estimators=200,
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=150,
            random_state=random_state,
        ),
    }


def run_cross_validation(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    models: dict[str, object],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    results = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                ("model", model),
            ]
        )
        scores = cross_validate(
            pipeline,
            train_x,
            train_y,
            cv=cv,
            scoring=SCORING,
            n_jobs=None,
        )
        results[name] = {
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "cv_f1_std": float(scores["test_f1"].std()),
            "cv_precision_mean": float(scores["test_precision"].mean()),
            "cv_precision_std": float(scores["test_precision"].std()),
            "cv_recall_mean": float(scores["test_recall"].mean()),
            "cv_recall_std": float(scores["test_recall"].std()),
        }

    return results


def select_best_model(cv_results: dict[str, dict[str, float]]) -> str:
    return max(
        cv_results,
        key=lambda name: (
            cv_results[name]["cv_roc_auc_mean"],
            cv_results[name]["cv_f1_mean"],
        ),
    )


def evaluate_model(
    model: object,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
) -> dict[str, object]:
    probabilities = model.predict_proba(valid_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(valid_y, probabilities)),
        "f1": float(f1_score(valid_y, predictions)),
        "precision": float(
            precision_score(valid_y, predictions, zero_division=0)
        ),
        "recall": float(recall_score(valid_y, predictions)),
        "confusion_matrix": confusion_matrix(
            valid_y,
            predictions,
        ).tolist(),
        "classification_report": classification_report(
            valid_y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def get_feature_importance(
    model: object,
    feature_names: list[str],
) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_[0]
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )
        .assign(abs_importance=lambda data: data["importance"].abs())
        .sort_values("abs_importance", ascending=False)
        .drop(columns=["abs_importance"])
    )


def save_model_card(metrics: dict[str, object]) -> None:
    best_model = metrics["best_model"]
    validation = metrics["validation_metrics"]

    text = f"""# Model Card

## Назначение модели

Baseline-модель оценивает вероятность churn для пользователя
subscription-продукта. На этом этапе модель используется только как
offline artifact: API inference еще не реализован.

## Данные

Источник данных: synthetic churn dataset.

Target: `churn`.

Используются поведенческие, платежные и продуктовые признаки:
активные дни, сессии, failed payments, обращения в поддержку,
feature usage score, давность последнего логина и engineered features.

## Кандидаты

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

Модели сравнивались через StratifiedKFold cross-validation.
Главная метрика выбора: ROC-AUC. F1 использовался как tie-breaker.

## Лучшая модель

Best model: `{best_model}`

Validation metrics:

- ROC-AUC: `{validation["roc_auc"]:.4f}`
- F1: `{validation["f1"]:.4f}`
- Precision: `{validation["precision"]:.4f}`
- Recall: `{validation["recall"]:.4f}`

Confusion matrix:

```text
{validation["confusion_matrix"]}
```

## Почему не accuracy

Churn обычно несбалансирован: ушедших пользователей меньше, чем
оставшихся. Модель, которая почти всегда предсказывает "no churn",
может получить высокую accuracy, но будет бесполезна для retention.
Поэтому важнее смотреть ROC-AUC, recall, precision и F1.

## Ограничения

- Данные synthetic, поэтому результаты нельзя считать бизнес-бенчмарком.
- Threshold зафиксирован на `0.5` и пока не оптимизирован под стоимость
  ошибок.
- Нет проверки temporal split, потому что датасет является snapshot.
- Drift monitoring и online inference будут добавлены на следующих этапах.
"""
    MODEL_CARD_PATH.write_text(text, encoding="utf-8")


def train_baseline_models(
    source: str = "csv",
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
) -> dict[str, object]:
    data = load_churn_data(source=source)
    data = add_feature_engineering(data)
    train_x, valid_x, train_y, valid_y = make_train_validation_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    train_x = train_x.drop(columns=DROP_COLUMNS)
    valid_x = valid_x.drop(columns=DROP_COLUMNS)

    models = build_candidate_models(random_state=random_state)
    cv_results = run_cross_validation(
        train_x=train_x,
        train_y=train_y,
        models=models,
        n_splits=n_splits,
        random_state=random_state,
    )
    best_model_name = select_best_model(cv_results)

    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(train_x)
    train_prepared = transform_to_dataframe(preprocessor, train_x)
    valid_prepared = transform_to_dataframe(preprocessor, valid_x)

    model = clone(models[best_model_name])
    model.fit(train_prepared, train_y)
    validation_metrics = evaluate_model(model, valid_prepared, valid_y)

    feature_importance = get_feature_importance(
        model=model,
        feature_names=list(train_prepared.columns),
    )

    metrics = {
        "best_model": best_model_name,
        "selection_metric": "cv_roc_auc_mean",
        "cv_results": cv_results,
        "validation_metrics": validation_metrics,
        "test_size": test_size,
        "random_state": random_state,
        "n_splits": n_splits,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TRAINED_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    save_model_card(metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline churn models."
    )
    parser.add_argument("--source", choices=["csv", "postgres"], default="csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_baseline_models(
        source=args.source,
        test_size=args.test_size,
        random_state=args.random_state,
        n_splits=args.n_splits,
    )
    validation = metrics["validation_metrics"]

    print(f"Best model: {metrics['best_model']}")
    print(f"Validation ROC-AUC: {validation['roc_auc']:.4f}")
    print(f"Validation F1: {validation['f1']:.4f}")
    print(f"Saved model: {TRAINED_MODEL_PATH}")
    print(f"Saved preprocessor: {PREPROCESSOR_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
