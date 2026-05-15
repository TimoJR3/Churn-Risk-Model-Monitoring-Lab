# Interview Notes

## Питч на 60 секунд

Это воспроизводимый DS-проект по churn prediction. Я генерирую синтетический
датасет подписочного продукта, делаю feature engineering и preprocessing,
сравниваю несколько baseline моделей через stratified cross-validation,
сохраняю model artifacts и использую их в FastAPI inference API. Проект
поддерживает одиночный прогноз, batch scoring, логирование прогнозов в
PostgreSQL без сохранения raw `user_id`, а также демо model monitoring:
summary по prediction logs, quality metrics и PSI drift checks. Результаты
можно посмотреть через Streamlit dashboard.

## Как объяснить churn prediction

Churn prediction — это binary classification: нужно оценить, уйдёт ли
пользователь из продукта. Для бизнеса это полезно, потому что риск можно
использовать для приоритизации retention-действий.

Ключевые мысли:

- churn часто несбалансирован, поэтому accuracy может вводить в заблуждение;
- recall важен, если дорого пропустить пользователя с риском оттока;
- precision важен, если retention-действия дорогие или навязчивые;
- threshold нужно выбирать с учётом стоимости false positives и false
  negatives;
- вероятность оттока — это risk signal, а не самостоятельное бизнес-решение.

## Как объяснить PSI

PSI, Population Stability Index, сравнивает распределение признака на expected
sample и actual sample.

Интуиция:

1. Значения разбиваются на buckets.
2. Для каждого bucket считается доля наблюдений в expected и actual.
3. Разница долей агрегируется через log ratio.
4. Чем выше PSI, тем сильнее сдвиг распределения.

Пороги в проекте:

- `< 0.1`: stable;
- `0.1 <= PSI < 0.25`: warning;
- `>= 0.25`: drift.

Ограничение: PSI одномерный. Он показывает data drift по одному признаку, но не
доказывает, что качество модели ухудшилось.

## Вопросы, которые может задать интервьюер

1. Почему ROC-AUC, а не accuracy?
   Churn может быть несбалансирован. ROC-AUC показывает ranking quality по
   разным thresholds, а accuracy может скрывать плохой recall.

2. Почему нужен stratified split?
   Чтобы сохранить долю churn/no churn в train, validation и CV folds.

3. Как выбрать threshold?
   Нужно оценить business cost false positives и false negatives, затем
   подобрать threshold на validation data. Следующий шаг — calibration.

4. Почему модель не переобучается в API?
   Inference API должен использовать уже сохранённые artifacts. Training и
   serving разделены, чтобы request был быстрым и воспроизводимым.

5. Как устроена приватность логов?
   Raw `user_id` удаляется из input features, в таблицу пишется только
   SHA-256 hash.

6. Как понять, что модель начала деградировать?
   Смотреть score distribution, PSI по важным признакам, data quality checks и
   delayed-label metrics: ROC-AUC, precision, recall, F1.

7. Почему данные синтетические?
   Для portfolio-проекта это безопасный способ показать DS pipeline без
   реальных персональных данных и коммерческих датасетов.

## Ограничения, которые стоит честно признать

- Датасет синтетический.
- Нет реального production traffic.
- Нет temporal split.
- Threshold фиксирован на `0.5` и не оптимизирован под cost.
- Нет probability calibration.
- PSI реализован как simple per-feature demo.
- Нет scheduler, alerting, model registry и automated retraining.

## Как улучшить проект дальше

- Добавить calibration.
- Оптимизировать threshold через cost matrix.
- Добавить temporal validation.
- Считать PSI по top features из feature importance.
- Добавить scheduled monitoring report.
- Добавить experiment tracking.
- Добавить model registry metadata.
- Добавить delayed-label feedback loop для оценки качества после inference.
