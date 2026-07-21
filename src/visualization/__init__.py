"""Utilitários de visualização.

Módulos
-------
theme
    Paleta de cores e estilo compartilhados do projeto.
plots
    Gráficos de avaliação (curva PR, ROC, matriz de confusão, importâncias).
"""

from src.visualization.plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.visualization.theme import COLORS, apply_theme

__all__ = [
    "COLORS",
    "apply_theme",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_roc_curve",
]
