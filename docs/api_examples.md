# API Examples

Assume the API runs at `http://localhost:8000`.

## Health

```bash
curl http://localhost:8000/health
```

## Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_sample.json
```

Response shape:

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

## Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_batch_sample.json
```

## Model Metadata

```bash
curl http://localhost:8000/model/metadata
```

## Recent Prediction Logs

```bash
curl "http://localhost:8000/predictions/recent?limit=20"
```

The response does not include raw `user_id`; it returns `user_id_hash` when an
identifier was provided.

## Monitoring Summary

```bash
curl http://localhost:8000/monitoring/summary
```

## Drift Check

```bash
curl -X POST http://localhost:8000/monitoring/drift \
  -H "Content-Type: application/json" \
  -d "{\"expected\":[1,2,3,4,5],\"actual\":[2,3,4,5,6],\"buckets\":5}"
```

## Quality Metrics

```bash
curl -X POST http://localhost:8000/monitoring/quality \
  -H "Content-Type: application/json" \
  -d "{\"y_true\":[0,0,1,1],\"y_score\":[0.1,0.4,0.6,0.9],\"threshold\":0.5}"
```
