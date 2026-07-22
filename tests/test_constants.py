"""Testes dos nomes canônicos de métricas."""

from __future__ import annotations

from src.constants.metrics import (
    AVERAGE_PRECISION,
    BRIER,
    F1,
    PRECISION,
    PRIMARY_METRIC,
    RECALL,
    REPORTED_METRICS,
    ROC_AUC,
)


def test_primary_metric_is_average_precision() -> None:
    """A métrica principal do projeto é a PR-AUC (average_precision)."""
    assert PRIMARY_METRIC == AVERAGE_PRECISION == "average_precision"


def test_reported_metrics_contains_all_canonical_names() -> None:
    """A lista de métricas reportadas contém todos os nomes canônicos, em ordem estável."""
    assert REPORTED_METRICS == [
        AVERAGE_PRECISION,
        ROC_AUC,
        RECALL,
        PRECISION,
        F1,
        BRIER,
    ]
