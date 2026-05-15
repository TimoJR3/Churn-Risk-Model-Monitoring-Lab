# Model Card: churn-модель

## Назначение модели

Модель оценивает вероятность оттока пользователя подписочного продукта на
синтетическом датасете. В проекте она используется как демонстрация полного
DS-процесса: данные, preprocessing, обучение, validation, inference API и
model monitoring.

Скор модели следует трактовать как сигнал риска для анализа и приоритизации
retention-действий. Это не автоматическое бизнес-решение.

## Целевая переменная

- `churn`
- Тип задачи: binary classification
- `1`: пользователь ушёл
- `0`: пользователь не ушёл

## Входные признаки

Raw input fields:

- `signup_date`
- `country`
- `plan_type`
- `monthly_fee`
- `days_active_last_30`
- `sessions_last_30`
- `support_tickets_last_30`
- `payments_failed_last_90`
- `avg_session_duration`
- `feature_usage_score`
- `last_login_days_ago`

`user_id` может приходить в API payload для демонстрационного логирования, но
не используется как model feature.

Engineered features:

- `activity_score`
- `payment_risk_score`
- `engagement_level`
- `days_since_signup`
- `usage_per_session`
- `support_intensity`

## Подход к моделированию

Кандидаты:

- Logistic Regression с `class_weight="balanced"`
- Random Forest с `class_weight="balanced"`
- HistGradientBoostingClassifier

Preprocessing:

- median imputation для числовых признаков;
- standard scaling для числовых признаков;
- most-frequent imputation для категориальных признаков;
- one-hot encoding для категориальных признаков;
- `user_id` и `signup_date` удаляются перед обучением.

## Метод валидации

Используется:

- stratified train/validation split;
- default `test_size=0.2`;
- `random_state=42` для воспроизводимости;
- `StratifiedKFold` cross-validation на train-части;
- ROC-AUC как основная метрика выбора модели;
- F1 как дополнительный сигнал.

## Метрики качества

Текущие сохранённые метрики из `artifacts/metrics.json`:

- best model: `random_forest`
- ROC-AUC: `0.8408`
- F1: `0.5072`
- precision: `0.3846`
- recall: `0.7447`
- confusion matrix: `[[297, 56], [12, 35]]`

Эти метрики получены на synthetic dataset и не являются бизнес-бенчмарком.

## Inference

Артефакты:

- `artifacts/trained_model.pkl`
- `artifacts/preprocessor.pkl`
- `artifacts/metrics.json`

FastAPI endpoints загружают artifacts через cache и не переобучают модель во
время request.

Default threshold: `0.5`, настраивается через `PREDICTION_THRESHOLD`.

Risk bands:

- `low`: probability `< 0.35`
- `medium`: `0.35 <= probability < 0.65`
- `high`: probability `>= 0.65`

## Сигналы мониторинга

Реализованы:

- количество prediction logs;
- средняя вероятность оттока;
- доля high-risk прогнозов;
- распределение `risk_band`;
- PSI drift check для числового признака;
- quality metrics по labels и scores, если labels доступны.

PSI thresholds:

- `stable`: PSI `< 0.1`
- `warning`: `0.1 <= PSI < 0.25`
- `drift`: PSI `>= 0.25`

## Ограничения

- Данные синтетические.
- Нет реального пользовательского трафика.
- Нет temporal split.
- Threshold не оптимизирован под business cost.
- Нет probability calibration.
- Monitoring request-driven и демонстрационный.
- Нет model registry, scheduled retraining и alert routing.

## Риски неправильного использования

- Churn score нельзя использовать как единственное основание для решения по
  клиенту.
- Retention-кампании на основе скоринга нужно проверять на качество,
  уместность и fairness.
- Синтетические данные не отражают реальные поведенческие, рыночные и
  демографические сдвиги.
- При переносе на реальные данные нельзя логировать raw персональные данные;
  признаки нужно проверять на privacy и fairness risks.

## Возможные улучшения

- Probability calibration.
- Cost-aware threshold optimization.
- Temporal validation / out-of-time holdout.
- Monitoring по top features из feature importance.
- Scheduled monitoring reports.
- Experiment tracking.
- Model registry metadata.
- Delayed-label quality tracking.
