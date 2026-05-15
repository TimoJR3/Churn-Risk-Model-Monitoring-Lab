# EDA-отчёт

Документ фиксирует базовый EDA для synthetic churn dataset.

## Обзор датасета

- Rows: `2000`
- Columns: `13`
- Target: `churn`
- Churn rate: `0.117`

## Пропуски

Пропуски намеренно есть в поведенческих признаках:

- `support_tickets_last_30`: `86`
- `avg_session_duration`: `85`
- `feature_usage_score`: `76`

Это похоже на event data: часть пользовательских событий или агрегатов может
отсутствовать. Preprocessing обрабатывает такие значения через imputation.

## Выбросы

Больше всего выбросов в:

- `monthly_fee`;
- `sessions_last_30`;
- `payments_failed_last_90`;
- `avg_session_duration`;
- `last_login_days_ago`.

Выбросы не удаляются на EDA-этапе, чтобы baseline model видела более реаличные
edge cases.

## Числовые распределения

Ключевые средние значения:

| Признак | Mean |
| --- | ---: |
| `monthly_fee` | `18.13` |
| `days_active_last_30` | `13.03` |
| `sessions_last_30` | `30.11` |
| `support_tickets_last_30` | `1.04` |
| `payments_failed_last_90` | `0.43` |
| `avg_session_duration` | `35.69` |
| `feature_usage_score` | `41.02` |
| `last_login_days_ago` | `12.19` |
| `churn` | `0.12` |

## Связь признаков с churn

Пользователи с churn в среднем:

- менее активны;
- имеют меньше сессий;
- реже используют функции продукта;
- дольше не заходили в продукт;
- имеют больше failed payments;
- чаще обращаются в поддержку.

Это соответствует логике subscription-продукта: низкая вовлечённость и
платёжное трение часто предшествуют churn.

## Churn по тарифам

| `plan_type` | Churn rate |
| --- | ---: |
| `basic` | `0.134` |
| `standard` | `0.100` |
| `premium` | `0.099` |

В synthetic data базовый тариф имеет чуть более высокий churn rate.

## Churn по странам

Churn rate отличается по `country`, но эти различия синтетические и не должны
интерпретироваться как реальные рыночные выводы.

## Leakage check

Потенциальные leakage risks:

- `churn` нельзя использовать как feature;
- `user_id` может привести к memorization;
- post-inference таблицы нельзя использовать для обучения;
- признаки после даты churn нельзя добавлять в training data.

Текущий preprocessing удаляет `user_id` и `signup_date`, а вместо даты
регистрации использует `days_since_signup`.
