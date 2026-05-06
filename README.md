# Churn Risk & Model Monitoring Lab

Портфолио-проект в формате production-like ML-сервиса: прогноз оттока
пользователей, API-инференс, логирование предсказаний, мониторинг модели и
Streamlit dashboard.

Проект показывает полный путь:

```text
synthetic data -> preprocessing -> training -> artifacts -> FastAPI inference
-> PostgreSQL logs -> monitoring -> Streamlit dashboard
```

## Бизнес-задача

Подписочный продукт хочет заранее находить пользователей с повышенным риском
оттока. Если риск известен до ухода клиента, бизнес может запустить retention:
персональную коммуникацию, помощь поддержки, обучение продукту, скидку или
ручную работу аккаунт-менеджера.

Модель оценивает вероятность churn по активности в продукте, платежным
проблемам, обращениям в поддержку, тарифу и давности последнего входа.

## Что демонстрирует проект

Проект ориентирован на junior Data Scientist / ML Engineer роль и показывает:

- воспроизводимую генерацию synthetic dataset;
- feature engineering и preprocessing вне notebook;
- baseline training и сохранение model artifacts;
- FastAPI inference с Pydantic-схемами;
- batch prediction;
- PostgreSQL prediction logs;
- privacy-aware logging: raw `user_id` не сохраняется;
- model metadata, PSI drift и quality monitoring endpoints;
- Streamlit dashboard поверх API, полностью русифицированный для демо на РФ-рынке;
- pytest, Ruff, Docker Compose и GitHub Actions CI.

## Архитектура

```text
data/raw synthetic CSV
    -> app.ml.preprocessing
    -> data/processed train/validation CSV
    -> app.ml.training
    -> artifacts/trained_model.pkl + preprocessor.pkl + metrics.json
    -> FastAPI /predict and /predict/batch
    -> PostgreSQL prediction_logs
    -> monitoring endpoints
    -> Streamlit dashboard
```

Документация:

- [Architecture](docs/architecture.md)
- [API examples](docs/api_examples.md)
- [Monitoring](docs/monitoring.md)
- [Demo script](docs/demo_script.md)
- [Data dictionary](docs/data_dictionary.md)
- [Model card](docs/model_card.md)

## Быстрый старт

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

Подготовить данные и модель:

```powershell
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
```

Если запускаете API локально без PostgreSQL, отключите сохранение логов:

```powershell
$env:SAVE_PREDICTIONS="false"
uvicorn app.api.main:app --reload
```

Dashboard:

```powershell
streamlit run dashboard/app.py
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Подготовить данные и модель:

```bash
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
```

Если запускаете API локально без PostgreSQL:

```bash
export SAVE_PREDICTIONS=false
uvicorn app.api.main:app --reload
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

### Docker Compose

Docker Compose поднимает API, dashboard и PostgreSQL. В этом режиме
`SAVE_PREDICTIONS=true` и prediction logs пишутся в базу.

```bash
cp .env.example .env
docker compose up --build
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Сервисы:

- API: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>
- PostgreSQL: `localhost:5432`

## Как проверить demo локально

Самый короткий сценарий для просмотра portfolio demo:

```bash
docker compose up --build -d
```

1. Откройте <http://localhost:8501>.
2. Убедитесь, что в верхней панели отображается `API онлайн`.
3. На вкладке `Прогноз` нажмите `Рассчитать риск оттока`.
4. Откройте `Пакетный прогноз` и нажмите `Запустить пакетный прогноз`.
5. Откройте `Мониторинг` и проверьте summary по prediction logs.
6. Откройте `Модель` и проверьте metadata и validation metrics.

Dashboard ориентирован на русскоязычного ревьюера: вкладки, кнопки,
пояснения, ошибки, empty states и таблицы отображаются на русском языке.

## Команды

Команды соответствуют `Makefile`.

| Задача | Make target | Команда |
| --- | --- | --- |
| Установить зависимости | `make install` | `pip install -r requirements.txt` |
| Сгенерировать данные | `make generate-data` | `python -m app.ml.generate_synthetic_data` |
| EDA-отчет | `make eda-report` | `python -m app.ml.eda` |
| Preprocessing | `make prepare-data` | `python -m app.ml.preprocessing` |
| Обучить модель | `make train` | `python -m app.ml.training` |
| Загрузить seed в БД | `make seed-db` | `python -m app.db.load_seed_data` |
| Запустить API | `make run-api` | `uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000` |
| Запустить dashboard | `make run-dashboard` | `streamlit run dashboard/app.py --server.port 8501` |
| Запустить тесты | `make test` | `pytest -q` |
| Docker up | `make docker-up` | `docker compose up --build` |
| Docker down | `make docker-down` | `docker compose down` |

Частые команды:

```bash
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
uvicorn app.api.main:app --reload
streamlit run dashboard/app.py
python -m pytest -q
docker compose up --build
```

## API endpoints

| Метод | Endpoint | Назначение |
| --- | --- | --- |
| `GET` | `/health` | Проверка состояния API |
| `POST` | `/predict` | Prediction для одного пользователя |
| `POST` | `/predict/batch` | Batch prediction |
| `GET` | `/model/metadata` | Метрики модели и статус artifacts |
| `GET` | `/predictions/recent` | Последние prediction logs без raw `user_id` |
| `GET` | `/monitoring/summary` | Summary по prediction logs |
| `POST` | `/monitoring/drift` | PSI drift check |
| `POST` | `/monitoring/quality` | ROC-AUC, precision, recall, F1 |

## Curl-примеры

Health:

```bash
curl http://localhost:8000/health
```

Single prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_sample.json
```

Batch prediction:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_batch_sample.json
```

Metadata:

```bash
curl http://localhost:8000/model/metadata
```

Recent logs:

```bash
curl "http://localhost:8000/predictions/recent?limit=20"
```

Drift:

```bash
curl -X POST http://localhost:8000/monitoring/drift \
  -H "Content-Type: application/json" \
  -d "{\"expected\":[1,2,3,4,5],\"actual\":[2,3,4,5,6],\"buckets\":5}"
```

Quality:

```bash
curl -X POST http://localhost:8000/monitoring/quality \
  -H "Content-Type: application/json" \
  -d "{\"y_true\":[0,0,1,1],\"y_score\":[0.1,0.4,0.6,0.9],\"threshold\":0.5}"
```

Больше примеров: [docs/api_examples.md](docs/api_examples.md).

## Тестирование и CI

Локальные проверки:

```bash
python -m compileall app dashboard tests
python -m ruff check .
python -m pytest -q
docker compose config
docker build -t churn-lab:test .
```

GitHub Actions запускает compile, Ruff, pytest, `docker compose config` и
Docker build на push/pull request в `main`. На tag `v*` публикуется Docker
image в GitHub Container Registry.

## Model monitoring

Monitoring layer включает:

- summary по prediction logs;
- PSI для numeric feature drift;
- quality metrics по labels и prediction scores.

PSI показывает, насколько распределение фактических значений отличается от
ожидаемого:

- `stable`: PSI `< 0.1`;
- `warning`: `0.1 <= PSI < 0.25`;
- `drift`: PSI `>= 0.25`.

Ограничения PSI в этом проекте: это univariate signal, результат зависит от
bucket-стратегии, нет scheduler jobs и alerting.

Подробнее: [docs/monitoring.md](docs/monitoring.md).

## Security / Privacy

- Секреты не хранятся в репозитории.
- `.env` игнорируется git.
- Prediction logs не сохраняют raw `user_id`; сохраняется только SHA-256 hash.
- Данные synthetic, без реальных персональных данных.
- Sample payloads вымышленные.

## Ограничения

- Synthetic dataset, поэтому метрики не являются бизнес-бенчмарком.
- Нет production SLA, autoscaling и incident process.
- Threshold `0.5` не оптимизирован под стоимость ошибок.
- Drift demo упрощен и запускается через API request.
- Dashboard не является production-grade authenticated console.

## Roadmap

- Probability calibration.
- Cost-aware threshold optimization.
- Scheduled monitoring.
- Evidently или custom HTML/PDF reports.
- Deployment с managed database и environment-specific secrets.
