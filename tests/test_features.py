"""Testes de engenharia de features."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.features.engineering import (
    ENGINEERED_COLUMNS,
    engineer_features,
    get_model_features,
)


def test_engineer_features_adds_expected_columns(
    synthetic_transactions: pl.DataFrame,
) -> None:
    """As colunas derivadas são adicionadas."""
    out = engineer_features(synthetic_transactions)
    for col in ENGINEERED_COLUMNS:
        assert col in out.columns


def test_hour_is_within_day_range(synthetic_transactions: pl.DataFrame) -> None:
    """A hora derivada fica no intervalo [0, 24)."""
    out = engineer_features(synthetic_transactions)
    hours = out["Hour"].to_numpy()
    assert np.all(hours >= 0) and np.all(hours < 24)


def test_amount_log_matches_log1p(synthetic_transactions: pl.DataFrame) -> None:
    """Amount_log corresponde a log1p(Amount)."""
    out = engineer_features(synthetic_transactions)
    expected = np.log1p(synthetic_transactions["Amount"].to_numpy())
    np.testing.assert_allclose(out["Amount_log"].to_numpy(), expected, rtol=1e-6)


def test_get_model_features_excludes_raw_time() -> None:
    """A lista de features do modelo não inclui Time bruto."""
    features = get_model_features()
    assert "Time" not in features
    assert "V1" in features
