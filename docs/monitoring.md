# Мониторинг модели

Monitoring layer в проекте специально сделан небольшим и тестируемым. Чистые
функции находятся в `app/monitoring`, API endpoints — в `app/api/routers`.

Это демо мониторинга модели, а не промышленная monitoring platform.

## Prediction summary

`GET /monitoring/summary` читает последние prediction logs и возвращает:

- `total_predictions`;
- `average_probability`;
- `high_risk_share`;
- `risk_band_counts`.

Этот endpoint показывает, сколько пользователей было просчитано, какая средняя
вероятность оттока и как распределены уровни риска.

## PSI drift

`POST /monitoring/drift` считает Population Stability Index для одного
числового признака.

PSI сравнивает expected distribution и actual distribution:

- маленький PSI означает, что распределения похожи;
- большой PSI указывает на возможный feature drift;
- PSI одномерный и требует контекста.

Пороги:

| PSI | Статус |
| --- | --- |
| `< 0.1` | `stable` |
| `0.1 <= PSI < 0.25` | `warning` |
| `>= 0.25` | `drift` |

Реализация обрабатывает `NaN`, константные массивы и нулевые bin counts через
малый epsilon.

## Quality metrics

`POST /monitoring/quality` принимает labels и prediction scores:

- ROC-AUC;
- precision;
- recall;
- F1;
- sample count;
- positive count.

Если `y_true` содержит один класс, endpoint возвращает `roc_auc: null`, а не
падает с traceback.

## Ограничения

- Monitoring запускается request-driven, scheduler отсутствует.
- PSI считается по одному признаку и не ловит все multivariate shifts.
- Нет alert routing.
- Нет model registry.
- Данные синтетические, thresholds демонстрационные.
