from __future__ import annotations

import importlib


def test_api_main_imports_without_side_effects() -> None:
    module = importlib.import_module("app.api.main")

    assert hasattr(module, "app")


def test_training_module_imports() -> None:
    module = importlib.import_module("app.ml.training")

    assert hasattr(module, "train_baseline_models")


def test_dashboard_app_imports_without_running_streamlit_server() -> None:
    module = importlib.import_module("dashboard.app")

    assert module is not None
