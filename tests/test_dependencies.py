"""Testes do carregamento e injeção do preditor na API (app/dependencies.py)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
import yaml
from app.dependencies import _load_deploy_config, get_predictor

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.inference.predictor import FraudPredictor
from src.models.factory import ModelName
from src.models.persistence import save_model
from src.training.trainer import build_training_pipeline, prepare_xy, train_model


@pytest.fixture(autouse=True)
def _clear_predictor_cache() -> Iterator[None]:
    """Garante que o cache do preditor (``lru_cache``) não vaze entre testes."""
    get_predictor.cache_clear()
    yield
    get_predictor.cache_clear()


def test_load_deploy_config_returns_empty_dict_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem ``configs/deploy.yaml``, retorna um dicionário vazio."""
    fake_paths = ProjectPaths(root=tmp_path, configs=tmp_path / "configs")
    monkeypatch.setattr("app.dependencies.get_paths", lambda: fake_paths)
    assert _load_deploy_config() == {}


def test_load_deploy_config_reads_api_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lê a seção ``api`` do YAML de deploy."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "deploy.yaml").write_text(
        yaml.safe_dump({"api": {"decision_threshold": 0.7, "model_path": "models/m.joblib"}}),
        encoding="utf-8",
    )
    fake_paths = ProjectPaths(root=tmp_path, configs=configs_dir)
    monkeypatch.setattr("app.dependencies.get_paths", lambda: fake_paths)
    config = _load_deploy_config()
    assert config["decision_threshold"] == 0.7


def test_get_predictor_loads_and_caches_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_transactions: pl.DataFrame,
    settings: Settings,
) -> None:
    """``get_predictor`` carrega o modelo do disco e reaproveita a instância em cache."""
    x, y = prepare_xy(synthetic_transactions)
    assert y is not None
    pipeline = build_training_pipeline(ModelName.LOGISTIC_REGRESSION, settings, y)
    pipeline = train_model(pipeline, x, y)

    model_path = tmp_path / "models" / "fraud_model.joblib"
    save_model(pipeline, model_path)

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "deploy.yaml").write_text(
        yaml.safe_dump(
            {"api": {"decision_threshold": 0.65, "model_path": "models/fraud_model.joblib"}}
        ),
        encoding="utf-8",
    )

    fake_paths = ProjectPaths(root=tmp_path, configs=configs_dir)
    monkeypatch.setattr("app.dependencies.get_paths", lambda: fake_paths)

    predictor = get_predictor()
    assert isinstance(predictor, FraudPredictor)
    assert predictor.threshold == 0.65
    assert get_predictor() is predictor
