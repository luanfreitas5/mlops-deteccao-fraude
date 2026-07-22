"""Testes de métricas, otimização de threshold e avaliador."""

from __future__ import annotations

import numpy as np

from src.evaluation.classification import compute_metrics
from src.evaluation.evaluator import evaluate_model
from src.evaluation.threshold import (
    find_cost_optimal_threshold,
    find_threshold_for_recall,
)


def test_compute_metrics_perfect_separation() -> None:
    """Separação perfeita produz PR-AUC e ROC-AUC iguais a 1."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = compute_metrics(y, p, threshold=0.5)
    assert metrics["average_precision"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0


def test_find_threshold_for_recall_reaches_target() -> None:
    """O threshold escolhido atinge ao menos o recall-alvo."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.9])
    threshold = find_threshold_for_recall(y, p, target_recall=0.66)
    y_pred = (p >= threshold).astype(int)
    recall = y_pred[y == 1].mean()
    assert recall >= 0.66


def test_cost_optimal_threshold_prefers_recall_when_fn_expensive() -> None:
    """Com FN muito caro, o threshold escolhido não deixa passar fraudes óbvias."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.9])
    threshold, cost = find_cost_optimal_threshold(y, p, 1000.0, 1.0)
    assert (p >= threshold).astype(int)[y == 1].mean() == 1.0
    assert cost >= 0.0


def test_evaluate_model_returns_ci() -> None:
    """O avaliador retorna um IC 95% válido (limite inferior <= superior)."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    p = np.clip(y * 0.6 + rng.normal(0, 0.2, size=100), 0, 1)
    result = evaluate_model(y, p, threshold=0.5, n_boot=100)
    lower, upper = result.pr_auc_ci
    assert lower <= upper
