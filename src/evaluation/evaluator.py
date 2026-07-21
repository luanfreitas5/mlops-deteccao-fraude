"""Avaliação de ponta a ponta com quantificação de incerteza.

Não reportamos um único ponto de métrica: além do valor pontual da PR-AUC,
estimamos um intervalo de confiança por bootstrap. Isso torna a comparação entre
modelos honesta e defensável.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from src.evaluation.classification import compute_metrics


@dataclass(frozen=True)
class EvaluationResult:
    """Resultado consolidado de uma avaliação.

    Attributes
    ----------
    metrics : dict[str, float]
        Métricas pontuais no threshold avaliado.
    threshold : float
        Threshold de decisão utilizado.
    pr_auc_ci : tuple[float, float]
        Intervalo de confiança (95%) da PR-AUC estimado por bootstrap.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    pr_auc_ci: tuple[float, float] = (0.0, 0.0)


def _bootstrap_pr_auc_ci(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estima o IC 95% da PR-AUC por reamostragem bootstrap.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Rótulos verdadeiros.
    y_prob : NDArray[np.float64]
        Probabilidades previstas.
    n_boot : int, optional
        Número de reamostragens, by default 1000.
    seed : int, optional
        Semente do gerador aleatório, by default 42.

    Returns
    -------
    tuple[float, float]
        Percentis 2.5% e 97.5% da distribuição bootstrap da PR-AUC.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # amostra sem ambas as classes: PR-AUC indefinida
        scores.append(float(average_precision_score(y_true[idx], y_prob[idx])))
    if not scores:
        return (0.0, 0.0)
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return (float(lower), float(upper))


def evaluate_model(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    threshold: float = 0.5,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> EvaluationResult:
    """Avalia previsões e retorna métricas com IC bootstrap da PR-AUC.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Rótulos verdadeiros.
    y_prob : NDArray[np.float64]
        Probabilidades previstas da classe positiva.
    threshold : float, optional
        Threshold de decisão, by default 0.5.
    n_boot : int, optional
        Número de reamostragens bootstrap, by default 1000.
    seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    EvaluationResult
        Métricas pontuais, threshold e IC 95% da PR-AUC.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> p = np.array([0.1, 0.2, 0.8, 0.6, 0.3, 0.9])
    >>> result = evaluate_model(y, p, threshold=0.5, n_boot=50)
    >>> "average_precision" in result.metrics
    True
    """
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    ci = _bootstrap_pr_auc_ci(y_true, y_prob, n_boot=n_boot, seed=seed)
    return EvaluationResult(metrics=metrics, threshold=threshold, pr_auc_ci=ci)
