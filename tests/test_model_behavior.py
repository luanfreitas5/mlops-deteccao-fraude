"""Testes comportamentais de ML e de regressão de métrica.

Verificam expectativas direcionais e um piso de desempenho no dado sintético com
sinal aprendível — falham se o modelo regredir de forma inaceitável.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.config.settings import Settings
from src.evaluation.classification import compute_metrics
from src.models.factory import ModelName
from src.training.trainer import build_training_pipeline, prepare_xy, train_model

# Piso mínimo de PR-AUC no dado sintético (sinal forte injetado nas fraudes).
MIN_PR_AUC = 0.80


@pytest.mark.ml
def test_pr_auc_does_not_regress(synthetic_transactions: pl.DataFrame, settings: Settings) -> None:
    """Regressão de métrica: a PR-AUC não pode cair abaixo do mínimo acordado."""
    x, y = prepare_xy(synthetic_transactions)
    assert y is not None
    pipeline = build_training_pipeline(ModelName.XGBOOST, settings, y)
    pipeline = train_model(pipeline, x, y)
    y_prob = pipeline.predict_proba(x)[:, 1]
    metrics = compute_metrics(y, y_prob, threshold=0.5)
    assert metrics["average_precision"] >= MIN_PR_AUC, (
        f"PR-AUC regrediu: {metrics['average_precision']:.4f}"
    )


@pytest.mark.ml
def test_stronger_fraud_signal_increases_probability(
    synthetic_transactions: pl.DataFrame, settings: Settings
) -> None:
    """Direcional: reforçar o sinal de fraude não reduz a probabilidade média prevista."""
    x, y = prepare_xy(synthetic_transactions)
    assert y is not None
    pipeline = build_training_pipeline(ModelName.XGBOOST, settings, y)
    pipeline = train_model(pipeline, x, y)

    baseline_prob = pipeline.predict_proba(x)[:, 1]
    # Amplifica as componentes que carregam o sinal de fraude.
    x_stronger = x.copy()
    for col in ("V1", "V2", "V3", "V4"):
        x_stronger[col] = x_stronger[col] + 3.0
    stronger_prob = pipeline.predict_proba(x_stronger)[:, 1]

    assert np.mean(stronger_prob) >= np.mean(baseline_prob)
