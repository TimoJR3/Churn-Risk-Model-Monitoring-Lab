# API-примеры

Примеры предполагают, что API запущен на `http://localhost:8000`.

## Health check

```bash
curl http://localhost:8000/health
```

## Одиночный прогноз

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_sample.json
```

Форма ответа:

```json
{
  "churn_probability": 0.42,
  "churn_prediction": 0,
  "risk_band": "medium",
  "threshold": 0.5,
  "model_version": "random_forest",
  "model_artifact_name": "trained_model.pkl",
  "explanation": "Risk band is medium; threshold 0.50 maps probabilities at or above threshold to churn."
}
```

## Batch scoring

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_batch_sample.json
```

## Metadata модели

```bash
curl http://localhost:8000/model/metadata
```

## Последние prediction logs

```bash
curl "http://localhost:8000/predictions/recent?limit=20"
```

Ответ не содержит raw `user_id`; если идентификатор был передан, возвращается
`user_id_hash`.

## Monitoring summary

```bash
curl http://localhost:8000/monitoring/summary
```

## PSI drift check

```bash
curl -X POST http://localhost:8000/monitoring/drift \
  -H "Content-Type: application/json" \
  -d "{\"expected\":[1,2,3,4,5],\"actual\":[2,3,4,5,6],\"buckets\":5}"
```

## Quality metrics

```bash
curl -X POST http://localhost:8000/monitoring/quality \
  -H "Content-Type: application/json" \
  -d "{\"y_true\":[0,0,1,1],\"y_score\":[0.1,0.4,0.6,0.9],\"threshold\":0.5}"
```
