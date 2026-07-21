"""Avaliação rigorosa de modelos de detecção de fraude.

Módulos
-------
classification
    Métricas de classificação adequadas a desbalanceamento extremo.
threshold
    Otimização do threshold de decisão (por recall-alvo ou custo de negócio).
evaluator
    Avaliação de ponta a ponta com intervalos de confiança (bootstrap).
"""

from src.evaluation.classification import compute_metrics
from src.evaluation.evaluator import EvaluationResult, evaluate_model
from src.evaluation.threshold import (
    find_cost_optimal_threshold,
    find_threshold_for_recall,
)

__all__ = [
    "EvaluationResult",
    "compute_metrics",
    "evaluate_model",
    "find_cost_optimal_threshold",
    "find_threshold_for_recall",
]
