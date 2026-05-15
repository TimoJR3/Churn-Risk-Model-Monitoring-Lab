# Demo script

Короткий сценарий презентации проекта на DS-собеседовании или перед
техническим ревьюером.

## 1. Объяснить бизнес-задачу

Проект решает churn prediction: нужно оценить риск оттока пользователя
подписочного продукта. Бизнес-смысл — заранее находить клиентов с повышенным
риском и приоритизировать retention-действия.

## 2. Показать архитектуру

Открыть [architecture.md](architecture.md) и показать поток:

```text
synthetic data -> preprocessing -> training -> artifacts
-> FastAPI inference -> PostgreSQL logs -> monitoring -> dashboard
```

## 3. Запустить ML workflow

```bash
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
```

Показать artifacts:

- `artifacts/trained_model.pkl`
- `artifacts/preprocessor.pkl`
- `artifacts/metrics.json`
- `artifacts/feature_importance.csv`
- `docs/model_card.md`

## 4. Запустить API

Для локального запуска без PostgreSQL:

```bash
export SAVE_PREDICTIONS=false
uvicorn app.api.main:app --reload
```

Проверить:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/metadata
```

## 5. Показать inference

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_sample.json
```

Пояснить поля:

- `churn_probability`;
- `churn_prediction`;
- `risk_band`;
- `threshold`;
- `model_version`.

## 6. Показать batch scoring

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_batch_sample.json
```

Пояснить, что batch endpoint нужен для скоринга нескольких пользователей одним
запросом.

## 7. Показать logging и monitoring

Если проект запущен через Docker Compose, prediction logs сохраняются в
PostgreSQL.

```bash
curl http://localhost:8000/monitoring/summary
curl "http://localhost:8000/predictions/recent?limit=20"
```

Отдельно объяснить privacy boundary: raw `user_id` не сохраняется, только hash.

## 8. Объяснить PSI

PSI сравнивает expected и actual distribution одного числового признака.

- `< 0.1`: stable;
- `0.1..0.25`: warning;
- `>= 0.25`: drift.

Важно сказать, что PSI — это data drift signal, а не доказательство падения
качества модели.

## 9. Показать dashboard

```bash
streamlit run dashboard/app.py
```

Открыть <http://localhost:8501> и показать:

- `Прогноз`;
- `Пакетный прогноз`;
- `Мониторинг`;
- `Модель`.

## 10. Показать инженерное качество

```bash
python -m compileall app dashboard tests
python -m ruff check .
python -m pytest -q
docker compose config
```

Подчеркнуть, что проект покрывает не только модель, но и inference, schemas,
monitoring functions, API endpoints и dashboard helpers.
