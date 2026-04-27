# Churn Risk & Model Monitoring Lab

Pet-project для портфолио junior Data Scientist / ML Engineer.

Проект показывает не только обучение ML-модели, а полный путь к
production-like ML-сервису: API, база данных, dashboard, тесты,
Docker Compose и документация.

## Бизнес-задача

Компания хочет заранее находить пользователей с высоким риском оттока.
Если такой риск известен до ухода клиента, бизнес может предложить
персональную коммуникацию, скидку или другой retention-сценарий.

Цель проекта - построить сервис, который в следующих этапах будет:

- загружать данные о пользователях;
- обучать baseline-модель churn prediction;
- отдавать прогноз через API;
- сохранять предсказания в PostgreSQL;
- отслеживать качество модели и drift;
- показывать состояние модели в Streamlit dashboard.

## Что уже сделано в Stage 1

На текущем этапе создан качественный каркас проекта:

- FastAPI сервис с endpoint `GET /health`;
- Streamlit dashboard-заготовка;
- PostgreSQL в `docker-compose.yml`;
- базовые pytest-тесты;
- конфигурация через `.env`;
- Dockerfile для API и dashboard;
- структура папок под ML, monitoring, DB, schemas и services;
- документация архитектуры в `docs/architecture.md`.

Обучение модели, drift detection, сохранение предсказаний и аналитика
dashboard пока намеренно не реализованы. Это будет добавляться по этапам,
чтобы проект оставался понятным и проверяемым.

## Стек

- Python 3.11
- FastAPI
- Streamlit
- pandas, NumPy, SciPy
- scikit-learn
- PostgreSQL
- SQLAlchemy
- Docker Compose
- pytest

## Быстрый старт

### Локальный запуск

Создать виртуальное окружение и установить зависимости:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Запустить API:

```powershell
uvicorn app.api.main:app --reload
```

Проверить:

- API health: http://localhost:8000/health
- Swagger docs: http://localhost:8000/docs

Запустить dashboard:

```powershell
streamlit run dashboard/app.py
```

Dashboard будет доступен здесь:

- http://localhost:8501

Запустить тесты:

```powershell
pytest -q
```

### Запуск через Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Сервисы:

- API: http://localhost:8000
- Dashboard: http://localhost:8501
- PostgreSQL: localhost:5432

## Структура проекта

```text
app/api          - HTTP endpoints и FastAPI wiring
app/core         - настройки приложения и env-конфигурация
app/db           - будущий слой работы с PostgreSQL
app/models       - будущие ORM-модели
app/schemas      - будущие Pydantic-схемы запросов и ответов
app/services     - бизнес-логика между API, ML и DB
app/ml           - будущие preprocessing, training и inference
app/monitoring   - будущие проверки качества модели и drift
dashboard        - Streamlit dashboard
data/raw         - исходные данные
data/processed   - подготовленные данные
artifacts        - будущие модели и ML-артефакты
tests            - automated tests
docs             - документация проекта
```

## Что этот проект демонстрирует

Этот проект сделан как портфолио-кейс, который показывает навыки,
важные для реальной DS/ML-инженерной работы:

- умение превращать ML-идею в сервис, а не только в notebook;
- понимание слоистой архитектуры приложения;
- базовую работу с API через FastAPI;
- подготовку к хранению данных и предсказаний в PostgreSQL;
- использование Docker Compose для локальной инфраструктуры;
- привычку покрывать поведение тестами;
- умение документировать проект для команды и ревьюера.

## Roadmap

Следующие этапы:

1. Добавить synthetic dataset, data model и seed-данные. `Done`
2. Сделать EDA и baseline churn-модель.
3. Добавить training pipeline и сохранение model artifact.
4. Реализовать prediction endpoint.
5. Сохранять prediction logs в PostgreSQL.
6. Добавить model quality checks и drift monitoring.
7. Расширить Streamlit dashboard метриками модели.
8. Добавить GitHub Actions для тестов.

## Генерация synthetic data

Сгенерировать воспроизводимый датасет:

```powershell
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
```

Команда создает:

- `data/raw/synthetic_churn_dataset.csv`;
- `data/processed/users.csv`;
- `data/processed/user_features.csv`.

Загрузить seed-данные в PostgreSQL после запуска Docker Compose:

```powershell
python -m app.db.load_seed_data
```
