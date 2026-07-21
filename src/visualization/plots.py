"""Gráficos de avaliação, salvos em ``.png`` (300 dpi) e ``.svg``.

Foco em curvas adequadas a desbalanceamento (Precision-Recall à frente da ROC) e
na matriz de confusão no threshold de operação.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.constants.labels import LABEL_NAMES
from src.visualization.theme import COLORS, apply_theme


def _save(fig: Figure, output_path: Path) -> None:
    """Salva a figura em PNG (300 dpi) e SVG, criando o diretório se preciso."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    output_path: Path,
) -> Path:
    """Plota a curva Precision-Recall e salva a figura.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Rótulos verdadeiros.
    y_prob : NDArray[np.float64]
        Probabilidades previstas da classe positiva.
    output_path : Path
        Caminho base de saída (sem extensão).

    Returns
    -------
    Path
        Caminho base das figuras salvas.
    """
    apply_theme()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=COLORS["fraude"], label=f"PR-AUC = {ap:.4f}")
    ax.set_title("Curva Precision-Recall — Detecção de Fraude")
    ax.set_xlabel("Recall (sensibilidade)")
    ax.set_ylabel("Precision (precisão)")
    ax.legend(loc="upper right")
    _save(fig, output_path)
    return output_path


def plot_roc_curve(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    output_path: Path,
) -> Path:
    """Plota a curva ROC e salva a figura.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Rótulos verdadeiros.
    y_prob : NDArray[np.float64]
        Probabilidades previstas da classe positiva.
    output_path : Path
        Caminho base de saída (sem extensão).

    Returns
    -------
    Path
        Caminho base das figuras salvas.
    """
    apply_theme()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS["legitima"], label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["neutro"])
    ax.set_title("Curva ROC — Detecção de Fraude")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.legend(loc="lower right")
    _save(fig, output_path)
    return output_path


def plot_confusion_matrix(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    output_path: Path,
) -> Path:
    """Plota a matriz de confusão no threshold de operação e salva a figura.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Rótulos verdadeiros.
    y_pred : NDArray[np.int_]
        Rótulos previstos (já limiarizados).
    output_path : Path
        Caminho base de saída (sem extensão).

    Returns
    -------
    Path
        Caminho base das figuras salvas.
    """
    apply_theme()
    display_labels = [LABEL_NAMES[0], LABEL_NAMES[1]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=display_labels,
        cmap="Reds",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Matriz de Confusão — Threshold de Operação")
    _save(fig, output_path)
    return output_path
