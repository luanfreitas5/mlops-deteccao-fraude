"""Testes de preparação de dados, montagem de pipeline e treino."""

from __future__ import annotations

import polars as pl

from src.config.settings import Settings
from src.features.engineering import get_model_features
from src.models.factory import ModelName
from src.training.cross_validation import cross_validate_pr_auc
from src.training.trainer import build_training_pipeline, prepare_xy, train_model


def test_prepare_xy_shapes(synthetic_transactions: pl.DataFrame) -> None:
    """prepare_xy retorna X com as colunas do modelo e y alinhado."""
    x, y = prepare_xy(synthetic_transactions)
    assert list(x.columns) == get_model_features()
    assert y is not None
    assert len(y) == synthetic_transactions.height


def test_prepare_xy_without_target(synthetic_transactions: pl.DataFrame) -> None:
    """Sem alvo, prepare_xy retorna y como None."""
    features_only = synthetic_transactions.drop("Class")
    x, y = prepare_xy(features_only, with_target=False)
    assert y is None
    assert x.height if hasattr(x, "height") else len(x)


def test_build_and_train_pipeline_predicts(
    synthetic_transactions: pl.DataFrame, settings: Settings
) -> None:
    """O pipeline treina e gera probabilidades no intervalo [0, 1]."""
    x, y = prepare_xy(synthetic_transactions)
    assert y is not None
    pipeline = build_training_pipeline(ModelName.XGBOOST, settings, y)
    pipeline = train_model(pipeline, x, y)
    proba = pipeline.predict_proba(x)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_cross_validate_returns_positive_pr_auc(
    synthetic_transactions: pl.DataFrame, settings: Settings
) -> None:
    """Com sinal aprendível, a PR-AUC média da CV é claramente positiva."""
    x, y = prepare_xy(synthetic_transactions)
    assert y is not None
    pipeline = build_training_pipeline(ModelName.LOGISTIC_REGRESSION, settings, y)
    result = cross_validate_pr_auc(pipeline, x, y, n_folds=3, seed=42)
    assert result.mean > 0.3
    assert result.ci_95 >= 0.0
