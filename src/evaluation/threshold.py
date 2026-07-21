"""Otimização do threshold de decisão.

Em fraude, o threshold de 0,5 raramente é o correto: o custo de um falso negativo
(fraude não detectada) é muito maior que o de um falso positivo (revisão de uma
transação legítima). Oferecemos duas estratégias:

- por **recall-alvo**: menor threshold que garante um recall mínimo, maximizando
  a precisão;
- por **custo de negócio**: threshold que minimiza o custo total esperado.
"""

from __future__ import annotations

import operator

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve


def find_threshold_for_recall(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    target_recall: float,
) -> float:
    """Encontra o threshold que atinge um recall mínimo com a maior precisão.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Rótulos verdadeiros.
    y_prob : NDArray[np.float64]
        Probabilidades previstas da classe positiva.
    target_recall : float
        Recall mínimo desejado (0 a 1).

    Returns
    -------
    float
        Threshold de decisão que satisfaz o recall-alvo. Se nenhum ponto atingir
        o alvo, retorna o menor threshold disponível (recall máximo).

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1])
    >>> p = np.array([0.1, 0.4, 0.35, 0.8])
    >>> 0.0 <= find_threshold_for_recall(y, p, 0.5) <= 1.0
    True
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # `thresholds` tem tamanho n-1 em relação a precision/recall.
    candidates = [
        (thresholds[i], precision[i]) for i in range(len(thresholds)) if recall[i] >= target_recall
    ]
    if not candidates:
        return float(thresholds.min())
    # Maior precisão entre os que satisfazem o recall-alvo.
    best_threshold, _ = max(candidates, key=operator.itemgetter(1))
    return float(best_threshold)


def find_cost_optimal_threshold(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    cost_false_negative: float,
    cost_false_positive: float,
    n_steps: int = 200,
) -> tuple[float, float]:
    """Encontra o threshold que minimiza o custo total esperado.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Rótulos verdadeiros.
    y_prob : NDArray[np.float64]
        Probabilidades previstas da classe positiva.
    cost_false_negative : float
        Custo de um falso negativo (fraude não detectada).
    cost_false_positive : float
        Custo de um falso positivo (transação legítima bloqueada/revisada).
    n_steps : int, optional
        Número de thresholds testados na varredura [0, 1], by default 200.

    Returns
    -------
    tuple[float, float]
        Par ``(melhor_threshold, custo_total_no_melhor_threshold)``.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1])
    >>> p = np.array([0.1, 0.4, 0.35, 0.8])
    >>> thr, cost = find_cost_optimal_threshold(y, p, 100.0, 1.0)
    >>> 0.0 <= thr <= 1.0
    True
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    best_threshold = 0.5
    best_cost = float("inf")
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        false_negatives = int(np.sum((y_pred == 0) & (y_true == 1)))
        false_positives = int(np.sum((y_pred == 1) & (y_true == 0)))
        total_cost = false_negatives * cost_false_negative + false_positives * cost_false_positive
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(threshold)
    return best_threshold, best_cost
