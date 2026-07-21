"""Métricas de classificação para dados fortemente desbalanceados.

A métrica principal é a PR-AUC (``average_precision``). Também reportamos ROC-AUC
(comparabilidade), recall/precision/F1 no threshold escolhido e o Brier score
(qualidade da calibração das probabilidades).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula o conjunto padrão de métricas de classificação.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Rótulos verdadeiros (0 = legítima, 1 = fraude).
    y_prob : NDArray[np.float64]
        Probabilidades previstas para a classe positiva (fraude).
    threshold : float, optional
        Threshold de decisão para as métricas dependentes de corte, by default 0.5.

    Returns
    -------
    dict[str, float]
        Dicionário com ``average_precision`` (PR-AUC), ``roc_auc``, ``recall``,
        ``precision``, ``f1`` e ``brier_score``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1])
    >>> p = np.array([0.1, 0.4, 0.35, 0.8])
    >>> metrics = compute_metrics(y, p, threshold=0.5)
    >>> "average_precision" in metrics
    True
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        # zero_division=0 é válido em runtime (retorna 0 sem warning); o stub do
        # sklearn tipa o parâmetro como str, por isso a supressão pontual.
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
