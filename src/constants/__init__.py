"""Constantes, colunas e rótulos do domínio de detecção de fraude.

Módulos
-------
columns
    Nomes de colunas do dataset Credit Card Fraud (Time, V1-V28, Amount, Class).
labels
    Rótulos e mapeamentos da tarefa de classificação binária.
metrics
    Nomes de métricas de avaliação usadas no projeto.
"""

from src.constants.columns import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    PCA_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from src.constants.labels import FRAUD, LABEL_NAMES, LEGITIMATE

__all__ = [
    "AMOUNT_COLUMN",
    "FEATURE_COLUMNS",
    "FRAUD",
    "LABEL_NAMES",
    "LEGITIMATE",
    "PCA_COLUMNS",
    "TARGET_COLUMN",
    "TIME_COLUMN",
]
