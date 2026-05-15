# Churn Risk & Model Monitoring Lab

Демонстрационный DS-проект по прогнозированию оттока пользователей: генерация
синтетических данных, обучение ML-модели, проверка качества, inference API,
batch scoring, логирование прогнозов в PostgreSQL и демо model monitoring с PSI
drift checks.

Проект сделан как воспроизводимый portfolio MVP для роли стажёра / Junior Data
Scientist. Это не промышленная система и не имитация реального трафика.

```text
синтетические данные -> preprocessing -> training -> artifacts
-> FastAPI inference / batch scoring -> PostgreSQL prediction logs
-> monitoring endpoints -> Streamlit dashboard
```

## Бизнес-задача

Подписочный продукт хочет заранее находить пользователей с повышенным риском
оттока. Если риск известен до ухода клиента, аналитик или retention-команда
может приоритизировать коммуникации, поддержку или продуктовые действия.

Модель оценивает вероятность churn по активности пользователя, платежным
сигналам, обращениям в поддержку, тарифу и давности последнего входа.

Важно: результат модели является оценкой риска, а не автоматическим бизнес-
решением.

## Данные и признаки

Датасет синтетический и генерируется командой:

```bash
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
```

Источник данных: `data/raw/synthetic_churn_dataset.csv`.

Целевая переменная:

- `churn`: `1`, если пользователь ушёл, и `0`, если не ушёл.

Профильные признаки:

- `user_id` — идентификатор для демо, не используется как model feature;
- `signup_date` — дата регистрации;
- `country` — страна;
- `plan_type` — тариф;
- `monthly_fee` — ежемесячный платёж.

Поведенческие признаки:

- `days_active_last_30` — активных дней за 30 дней;
- `sessions_last_30` — сессий за 30 дней;
- `avg_session_duration` — средняя длительность сессии;
- `feature_usage_score` — индекс использования функций;
- `last_login_days_ago` — дней с последнего входа.

Сигналы трения:

- `support_tickets_last_30` — обращения в поддержку;
- `payments_failed_last_90` — неуспешные платежи.

Feature engineering добавляет:

- `activity_score`;
- `payment_risk_score`;
- `engagement_level`;
- `days_since_signup`;
- `usage_per_session`;
- `support_intensity`.

В генераторе есть пропуски и небольшая доля выбросов, чтобы preprocessing был
похож на реальную DS-задачу, а не на идеально чистую таблицу.

## Подход к моделированию

Код обучения находится в `app/ml/training.py`.

Сравниваются базовые scikit-learn модели:

- Logistic Regression с `class_weight="balanced"`;
- Random Forest с `class_weight="balanced"`;
- HistGradientBoostingClassifier.

Preprocessing:

- числовые признаки: median imputation + standard scaling;
- категориальные признаки: most-frequent imputation + one-hot encoding;
- `user_id` и `signup_date` удаляются перед обучением.

Обучение сохраняет:

- `artifacts/trained_model.pkl`;
- `artifacts/preprocessor.pkl`;
- `artifacts/metrics.json`;
- `artifacts/feature_importance.csv`;
- `docs/model_card.md`.

## Валидация

Используется stratified train/validation split:

- `test_size=0.2`;
- `random_state=42`;
- стратификация по `churn`.

Выбор модели выполняется через `StratifiedKFold` cross-validation на train-
части. Основная метрика выбора — ROC-AUC, дополнительный ориентир — F1.

Почему не только accuracy: в churn-задачах классы часто несбалансированы, и
модель, которая почти всегда предсказывает `no churn`, может иметь высокую
accuracy, но плохо находить пользователей с риском оттока.

## Метрики качества

В `artifacts/metrics.json` сохраняются:

- ROC-AUC;
- F1;
- precision;
- recall;
- confusion matrix;
- classification report;
- результаты cross-validation.

Текущие сохранённые метрики для synthetic dataset:

- best model: `random_forest`;
- ROC-AUC: `0.8408`;
- F1: `0.5072`;
- precision: `0.3846`;
- recall: `0.7447`;
- confusion matrix: `[[297, 56], [12, 35]]`.

Эти значения подтверждены сохранённым `artifacts/metrics.json`, но не являются
бизнес-бенчмарком, потому что данные синтетические.

## API для прогноза

FastAPI приложение находится в `app/api/main.py`.

Основные endpoints:

- `GET /health`;
- `POST /predict`;
- `POST /predict/batch`;
- `GET /model/metadata`;
- `GET /predictions/recent`;
- `GET /monitoring/summary`;
- `POST /monitoring/drift`;
- `POST /monitoring/quality`.

`POST /predict` принимает признаки одного пользователя, применяет сохранённый
preprocessor и model artifact, затем возвращает:

- `churn_probability`;
- `churn_prediction`;
- `risk_band`: `low`, `medium`, `high`;
- `threshold`;
- `model_version`;
- `model_artifact_name`;
- `explanation`.

Если artifacts отсутствуют, API возвращает контролируемый HTTP 503 с командой
для обучения модели. Модель не переобучается во время request.

Пример:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_sample.json
```

## Пакетный скоринг

`POST /predict/batch` принимает список пользователей и возвращает:

- `row_count`;
- `items` со структурой ответа как у `/predict`.

Ограничение размера batch задаётся через `MAX_BATCH_SIZE`, значение по
умолчанию — `100`.

Пример:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_batch_sample.json
```

## Логирование прогнозов

При `SAVE_PREDICTIONS=true` endpoint `/predict` сохраняет prediction log в
PostgreSQL.

Сохраняется:

- `request_id`;
- `user_id_hash`, а не raw `user_id`;
- `churn_probability`;
- `churn_prediction`;
- `risk_band`;
- `threshold`;
- `model_version`;
- sanitized input features;
- `created_at`.

Последние прогнозы доступны через:

```bash
curl "http://localhost:8000/predictions/recent?limit=20"
```

Для локального запуска API без PostgreSQL можно отключить сохранение:

```bash
export SAVE_PREDICTIONS=false
```

или в PowerShell:

```powershell
$env:SAVE_PREDICTIONS="false"
```

## Мониторинг модели

Monitoring layer находится в `app/monitoring`.

Реализовано:

- summary по prediction logs;
- распределение risk bands;
- средняя вероятность оттока;
- доля high-risk прогнозов;
- PSI drift check для числового признака;
- quality metrics по `y_true` и `y_score`, если labels доступны.

Endpoint:

```bash
curl http://localhost:8000/monitoring/summary
```

## PSI drift

PSI, Population Stability Index, сравнивает expected distribution и actual
distribution одного числового признака. Если распределение сильно изменилось,
это может быть сигналом data drift.

Пороговые значения в проекте:

- `stable`: PSI `< 0.1`;
- `warning`: `0.1 <= PSI < 0.25`;
- `drift`: PSI `>= 0.25`.

Пример:

```bash
curl -X POST http://localhost:8000/monitoring/drift \
  -H "Content-Type: application/json" \
  -d "{\"expected\":[1,2,3,4,5],\"actual\":[2,3,4,5,6],\"buckets\":5}"
```

Ограничение: PSI — одномерный сигнал. Он не доказывает падение качества модели
и не ловит все виды multivariate drift.

## Dashboard

Streamlit dashboard полностью на русском языке и находится в `dashboard/app.py`.

Вкладки:

- `Прогноз` — одиночный inference;
- `Пакетный прогноз` — batch scoring;
- `Мониторинг` — prediction logs и risk distribution;
- `Модель` — metadata, metrics и artifacts.

Скриншоты:

![Одиночный прогноз риска оттока](docs/images/dashboard-predict.jpg)

![Пакетный прогноз оттока](docs/images/dashboard-batch.jpg)

![Мониторинг последних предсказаний](docs/images/dashboard-monitoring.jpg)

![Метаданные и метрики модели](docs/images/dashboard-model.jpg)

## Запуск

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env

python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
```

API без PostgreSQL logging:

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

python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
```

API без PostgreSQL logging:

```bash
export SAVE_PREDICTIONS=false
uvicorn app.api.main:app --reload
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

### Docker Compose

Docker Compose поднимает API, dashboard и PostgreSQL. В этом режиме prediction
logging включён.

```bash
cp .env.example .env
docker compose up --build
```

Сервисы:

- API: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>
- PostgreSQL: `localhost:5432`

## Как проверить demo локально

```bash
docker compose up --build -d
```

1. Открыть <http://localhost:8501>.
2. Проверить, что header показывает `API онлайн`.
3. На вкладке `Прогноз` нажать `Рассчитать риск оттока`.
4. На вкладке `Пакетный прогноз` нажать `Запустить пакетный прогноз`.
5. На вкладке `Мониторинг` проверить summary.
6. На вкладке `Модель` проверить metadata и validation metrics.

## Команды

| Задача | Make target | Команда |
| --- | --- | --- |
| Установить зависимости | `make install` | `pip install -r requirements.txt` |
| Сгенерировать данные | `make generate-data` | `python -m app.ml.generate_synthetic_data` |
| EDA-отчёт | `make eda-report` | `python -m app.ml.eda` |
| Preprocessing | `make prepare-data` | `python -m app.ml.preprocessing` |
| Обучить модель | `make train` | `python -m app.ml.training` |
| Загрузить seed в БД | `make seed-db` | `python -m app.db.load_seed_data` |
| Запустить API | `make run-api` | `uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000` |
| Запустить dashboard | `make run-dashboard` | `streamlit run dashboard/app.py --server.port 8501` |
| Запустить тесты | `make test` | `pytest -q` |
| Docker up | `make docker-up` | `docker compose up --build` |
| Docker down | `make docker-down` | `docker compose down` |

## Проверки и CI

```bash
python -m compileall app dashboard tests
python -m ruff check .
python -m pytest -q
docker compose config
docker build -t churn-lab:test .
```

GitHub Actions запускает compile, Ruff, pytest, Docker Compose config и Docker
build на push / pull request в `main`.

## Что проект демонстрирует для DS-роли

- Умение сформулировать churn prediction как binary classification.
- Воспроизводимую генерацию synthetic dataset.
- Feature engineering и preprocessing вне notebook.
- Stratified split и StratifiedKFold validation.
- Сравнение baseline моделей.
- Метрики ROC-AUC, F1, precision, recall и confusion matrix.
- Сохранение artifacts и повторное использование в inference.
- Typed API schemas для online и batch scoring.
- Privacy-aware logging без raw `user_id`.
- Простую реализацию PSI drift check.
- Dashboard для презентации результата техническому и нетехническому ревьюеру.

## Ограничения

- Данные синтетические.
- Нет реального production traffic.
- Нет production SLA или промышленного мониторинга.
- Threshold `0.5` не оптимизирован под стоимость ошибок.
- Нет probability calibration.
- Нет temporal validation split, потому что датасет является snapshot.
- PSI monitoring упрощён и запускается request-driven.
- Нет scheduler, alerting, model registry и automated retraining.
- Dashboard — локальный demo UI, не защищённая операционная консоль.

## Документация

- [Architecture](docs/architecture.md)
- [API examples](docs/api_examples.md)
- [Monitoring](docs/monitoring.md)
- [Model card](docs/model_card.md)
- [Interview notes](docs/interview_notes.md)
- [Demo script](docs/demo_script.md)
- [Data dictionary](docs/data_dictionary.md)

## GitHub-подача

Описание репозитория:

```text
Демонстрационный DS-проект по прогнозированию оттока пользователей, inference API, batch scoring и мониторингу PSI drift.
```

Темы:

```text
data-science, churn-prediction, machine-learning, model-monitoring,
psi-drift, fastapi, postgresql, streamlit, scikit-learn
```

## Что можно улучшить дальше

- Probability calibration.
- Cost-aware threshold optimization.
- Temporal validation или out-of-time holdout.
- Scheduled monitoring report.
- Отслеживание качества по delayed labels.
- Experiment tracking.
- Простая model registry metadata.
