"""Testes do preditor de inferência."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config.settings import Settings
from src.inference.predictor import FraudPredictor
from src.models.factory import ModelName
from src.training.trainer import build_training_pipeline, prepare_xy, train_model


def _fit_predictor(df: pl.DataFrame, settings: Settings, threshold: float) -> FraudPredictor:
    """Treina um pipeline e devolve um preditor pronto."""
    x, y = prepare_xy(df)
    assert y is not None
    pipeline = build_training_pipeline(ModelName.XGBOOST, settings, y)
    pipeline = train_model(pipeline, x, y)
    return FraudPredictor(pipeline=pipeline, threshold=threshold)


def test_predict_proba_range(synthetic_transactions: pl.DataFrame, settings: Settings) -> None:
    """As probabilidades ficam em [0, 1] e há uma por transação."""
    predictor = _fit_predictor(synthetic_transactions, settings, 0.5)
    proba = predictor.predict_proba(synthetic_transactions.drop("Class"))
    assert proba.shape[0] == synthetic_transactions.height
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_respects_threshold(
    synthetic_transactions: pl.DataFrame, settings: Settings
) -> None:
    """Threshold maior nunca produz mais alertas de fraude que um menor."""
    predictor = _fit_predictor(synthetic_transactions, settings, 0.5)
    features = synthetic_transactions.drop("Class")

    predictor.threshold = 0.2
    alerts_low = predictor.predict(features).sum()
    predictor.threshold = 0.8
    alerts_high = predictor.predict(features).sum()
    assert alerts_high <= alerts_low
