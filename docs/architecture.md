# Architecture

Stage 1 creates a simple layered scaffold. The goal is to keep the project
easy to understand while leaving clear places for future ML service code.

```text
                  +----------------------+
                  | Streamlit dashboard  |
                  +----------+-----------+
                             |
                             v
+--------------+  +----------v-----------+
| API clients  +->+ FastAPI application  |
+--------------+  +----------+-----------+
                             |
                             v
                  +----------+-----------+
                  | Service layer        |
                  +----------+-----------+
                             |
          +------------------+------------------+
          |                  |                  |
   +------v-----+     +------v-----+     +------v-----+
   | ML layer   |     | DB layer   |     | Monitoring |
   +------+-----+     +------+-----+     +------+-----+
          |                  |                  |
   +------v-----+     +------v-----+     +------v-----+
   | Artifacts  |     | PostgreSQL |     | Reports    |
   +------------+     +------------+     +------------+
```

## Layers

- `app/api`: HTTP endpoints and API wiring.
- `app/core`: configuration and shared application settings.
- `app/db`: database connections and repository code.
- `app/models`: database table models.
- `app/schemas`: request and response schemas.
- `app/services`: business logic used by the API.
- `app/ml`: training, preprocessing, and inference code.
- `app/monitoring`: model quality and drift checks.
- `dashboard`: Streamlit UI for project metrics.
- `data`: raw and processed datasets.
- `artifacts`: trained models and generated ML files.
- `tests`: automated tests.
- `docs`: project documentation.
