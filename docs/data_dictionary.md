# Data Dictionary

Документ описывает synthetic dataset для задачи churn prediction.
Сценарий: subscription-продукт или онлайн-сервис.

## Target

`churn` показывает, ушел ли пользователь из продукта.

- `1`: пользователь ушел;
- `0`: пользователь остался.

В synthetic data `churn` не является случайным шумом. Вероятность churn
растет при низкой активности, большом числе failed payments, частых
обращениях в поддержку и долгом отсутствии логина. Вероятность churn
снижается при высоком `feature_usage_score`, большем числе активных дней
и более дорогом плане.

## Поля

| Поле | Тип | Описание |
| --- | --- | --- |
| `user_id` | integer | Уникальный идентификатор пользователя. |
| `signup_date` | date | Дата регистрации пользователя. |
| `country` | string | Страна пользователя. |
| `plan_type` | string | Тариф: `basic`, `standard`, `premium`. |
| `monthly_fee` | float | Ежемесячная стоимость подписки. |
| `days_active_last_30` | integer | Сколько дней пользователь был активен за последние 30 дней. |
| `sessions_last_30` | integer | Количество сессий за последние 30 дней. |
| `support_tickets_last_30` | integer | Количество обращений в поддержку за последние 30 дней. |
| `payments_failed_last_90` | integer | Количество неуспешных платежей за последние 90 дней. |
| `avg_session_duration` | float | Средняя длительность сессии в минутах. |
| `feature_usage_score` | float | Индекс использования ключевых функций от 0 до 100. |
| `last_login_days_ago` | integer | Сколько дней прошло с последнего логина. |
| `churn` | integer | Целевая переменная: `1` - ушел, `0` - остался. |

## Пропуски

Часть значений намеренно содержит пропуски:

- `avg_session_duration`;
- `feature_usage_score`;
- `support_tickets_last_30`.

Это нужно, чтобы следующие этапы проекта могли показать preprocessing:
imputation, validation и обработку неполных пользовательских данных.

## Выбросы

Около 1% пользователей получают аномально высокие значения:

- `sessions_last_30`;
- `avg_session_duration`.

Такие выбросы имитируют power users, ботов или ошибочные события
трекинга. Это полезно для будущего EDA и мониторинга данных.

## PostgreSQL Tables

### `users`

Хранит стабильные атрибуты пользователя:

- `user_id`;
- `signup_date`;
- `country`;
- `plan_type`;
- `monthly_fee`;
- `churn`.

### `user_features`

Хранит feature snapshot для пользователя на конкретную дату:

- `user_id`;
- `snapshot_date`;
- activity features;
- payment features;
- support and usage features.

### `model_predictions`

Будет хранить предсказания модели:

- `user_id`;
- `model_version`;
- `churn_probability`;
- `predicted_churn`;
- `prediction_date`.

### `model_metrics`

Будет хранить метрики качества модели:

- ROC-AUC;
- F1;
- precision;
- recall.

### `drift_metrics`

Будет хранить результаты drift checks по признакам:

- `feature_name`;
- `drift_score`;
- `drift_detected`.
