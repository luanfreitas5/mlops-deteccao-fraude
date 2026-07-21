"""Nomes canônicos das métricas de avaliação usadas no projeto.

A métrica principal é ``average_precision`` (PR-AUC): sob desbalanceamento
extremo (~0,17% de fraudes), a ROC-AUC é otimista demais, enquanto a PR-AUC
foca no desempenho sobre a classe positiva (fraude).
"""

from __future__ import annotations

AVERAGE_PRECISION: str = "average_precision"  # PR-AUC — métrica principal
ROC_AUC: str = "roc_auc"
RECALL: str = "recall"
PRECISION: str = "precision"
F1: str = "f1"
BRIER: str = "brier_score"

PRIMARY_METRIC: str = AVERAGE_PRECISION

# Métricas reportadas em toda avaliação (ordem estável para relatórios).
REPORTED_METRICS: list[str] = [
    AVERAGE_PRECISION,
    ROC_AUC,
    RECALL,
    PRECISION,
    F1,
    BRIER,
]
