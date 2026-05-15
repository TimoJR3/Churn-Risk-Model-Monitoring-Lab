# Словарь данных

Документ описывает synthetic dataset для задачи churn prediction.
Сценарий: подписочный продукт или онлайн-сервис.

## Целевая переменная

`churn` показывает, ушёл ли пользователь из продукта.

- `1`: пользователь ушёл;
- `0`: пользователь остался.

В synthetic data `churn` зависит от активности, платежных проблем, обращений в
поддержку, давности логина и тарифа. Это сделано, чтобы модель училась на
понятных бизнес-сигналах, а не на случайном шуме.

## Raw поля

| Поле | Тип | Описание |
| --- | --- | --- |
| `user_id` | integer | Уникальный идентификатор пользователя. Не используется как model feature. |
| `signup_date` | date | Дата регистрации пользователя. |
| `country` | string | Страна пользователя. |
| `plan_type` | string | Тариф: `basic`, `standard`, `premium`. |
| `monthly_fee` | float | Ежемесячная стоимость подписки. |
| `days_active_last_30` | integer | Активных дней за последние 30 дней. |
| `sessions_last_30` | integer | Количество сессий за последние 30 дней. |
| `support_tickets_last_30` | integer | Обращения в поддержку за последние 30 дней. |
| `payments_failed_last_90` | integer | Неуспешные платежи за последние 90 дней. |
| `avg_session_duration` | float | Средняя длительность сессии в минутах. |
| `feature_usage_score` | float | Индекс использования ключевых функций от 0 до 100. |
| `last_login_days_ago` | integer | Дней с последнего логина. |
| `churn` | integer | Target: `1` — ушёл, `0` — остался. |

## Пропуски

Часть значений намеренно содержит пропуски:

- `avg_session_duration`;
- `feature_usage_score`;
- `support_tickets_last_30`.

Это позволяет показать preprocessing: imputation, validation и работу с
неполными пользовательскими данными.

## Выбросы

Около 1% пользователей получают аномально высокие значения:

- `sessions_last_30`;
- `avg_session_duration`.

Такие значения имитируют power users, bots или шум event tracking.

## Engineered features

Эти признаки создаются в `app.ml.features` и используются моделью после
preprocessing.

| Поле | Тип | Описание |
| --- | --- | --- |
| `activity_score` | float | Композитный скор активности на основе активных дней, сессий и usage score. |
| `payment_risk_score` | float | Нормализованный риск проблем с оплатой. |
| `engagement_level` | string | Категория вовлечённости: `low`, `medium`, `high`. |
| `days_since_signup` | integer | Дней между регистрацией и snapshot date. |
| `usage_per_session` | float | Usage score на одну сессию. |
| `support_intensity` | float | Доля обращений в поддержку относительно числа сессий. |

## PostgreSQL tables

### `users`

Seed-таблица со стабильными атрибутами пользователя:

- `user_id`;
- `signup_date`;
- `country`;
- `plan_type`;
- `monthly_fee`;
- `churn`.

### `user_features`

Seed-таблица с feature snapshot:

- `user_id`;
- `snapshot_date`;
- activity features;
- payment features;
- support and usage features.

### `prediction_logs`

Таблица логирования inference:

- `request_id`;
- `user_id_hash`;
- `churn_probability`;
- `churn_prediction`;
- `risk_band`;
- `threshold`;
- `model_version`;
- `input_features`;
- `created_at`.

Raw `user_id` не сохраняется в prediction logs.

## Leakage check

- `churn` нельзя использовать как feature, это target.
- `user_id` не содержит бизнес-сигнала и может привести к memorization.
- Признаки, созданные после даты churn, нельзя добавлять в training data.
- Prediction logs и monitoring metrics не должны использоваться для обучения,
  потому что они появляются после inference.

Preprocessing удаляет `user_id` и `signup_date`; вместо raw даты используется
engineered feature `days_since_signup`.
