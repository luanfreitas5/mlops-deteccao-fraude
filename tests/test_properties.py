"""Testes property-based (hypothesis) para funções de transformação e métricas."""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from src.constants.columns import PCA_COLUMNS
from src.evaluation.threshold import find_cost_optimal_threshold
from src.features.engineering import ENGINEERED_COLUMNS, engineer_features


@given(
    amounts=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
def test_engineer_features_never_produces_nan_on_valid_input(
    amounts: list[float],
) -> None:
    """Invariante: entradas válidas nunca geram NaN nas features derivadas."""
    n = len(amounts)
    data = {"Time": [float(i) for i in range(n)], "Amount": amounts}
    for col in PCA_COLUMNS:
        data[col] = [0.0] * n
    out = engineer_features(pl.DataFrame(data))
    for col in ENGINEERED_COLUMNS:
        assert not np.any(np.isnan(out[col].to_numpy()))


@given(
    threshold_seed=st.integers(min_value=0, max_value=10_000),
    cost_fn=st.floats(min_value=1, max_value=1000),
)
@settings(max_examples=25, deadline=None)
def test_cost_optimal_threshold_in_unit_interval(threshold_seed: int, cost_fn: float) -> None:
    """Invariante: o threshold ótimo por custo está sempre em [0, 1]."""
    rng = np.random.default_rng(threshold_seed)
    y = rng.integers(0, 2, size=50)
    if len(np.unique(y)) < 2:
        y[0], y[1] = 0, 1
    p = rng.uniform(0, 1, size=50)
    threshold, _ = find_cost_optimal_threshold(y, p, cost_fn, 1.0)
    assert 0.0 <= threshold <= 1.0
