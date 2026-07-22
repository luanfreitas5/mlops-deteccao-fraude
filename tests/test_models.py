"""Testes da fábrica de modelos e da persistência."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import ModelParams
from src.exceptions.model import ModelNotFoundError
from src.models.factory import ModelName, create_estimator
from src.models.persistence import load_model, save_model


def test_create_logistic_regression() -> None:
    """A fábrica cria uma Regressão Logística com random_state fixo."""
    est = create_estimator(ModelName.LOGISTIC_REGRESSION, ModelParams(), seed=42)
    assert est.__class__.__name__ == "LogisticRegression"
    assert est.random_state == 42


def test_create_xgboost_applies_scale_pos_weight() -> None:
    """O XGBoost recebe o scale_pos_weight informado."""
    est = create_estimator(ModelName.XGBOOST, ModelParams(), seed=42, scale_pos_weight=99.0)
    assert est.__class__.__name__ == "XGBClassifier"
    assert est.get_params()["scale_pos_weight"] == 99.0


def test_save_and_load_model_roundtrip(tmp_path: Path) -> None:
    """Salvar e carregar um objeto preserva o conteúdo e grava metadados."""
    obj = {"pesos": [1, 2, 3]}
    path = save_model(obj, tmp_path / "model.joblib", metadata={"threshold": 0.3})
    assert path.with_suffix(".json").exists()
    reloaded = load_model(path)
    assert reloaded == obj


def test_load_missing_model_raises(tmp_path: Path) -> None:
    """Carregar um modelo inexistente levanta ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "ausente.joblib")
