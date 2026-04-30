# Model Card

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

Best model: `random_forest`

Validation metrics:

- ROC-AUC: `0.8408`
- F1: `0.5072`
- Precision: `0.3846`
- Recall: `0.7447`

Confusion matrix:

```text
[[297, 56], [12, 35]]
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
