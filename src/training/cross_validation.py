"""Validação cruzada estratificada com intervalo de confiança da PR-AUC.

Uma única pontuação de CV não basta: reportamos média ± IC 95% (via desvio-padrão
entre folds), permitindo julgar se a diferença entre modelos é relevante.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from src.config.logging import get_logger
from src.evaluation.classification import average_precision_score
from src.models.base import Estimator

logger = get_logger(__name__)


@dataclass(frozen=True)
class CVResult:
    """Resultado da validação cruzada.

    Attributes
    ----------
    mean : float
        PR-AUC média entre os folds.
    std : float
        Desvio-padrão da PR-AUC entre os folds.
    ci_95 : float
        Meia-amplitude do IC 95% (1.96 * std / sqrt(k)).
    scores : list[float]
        PR-AUC de cada fold.
    """

    mean: float
    std: float
    ci_95: float
    scores: list[float]


def cross_validate_pr_auc(
    pipeline: BaseEstimator,
    x: pd.DataFrame,
    y: NDArray[np.int_],
    n_folds: int = 5,
    seed: int = 42,
) -> CVResult:
    """Executa validação cruzada estratificada medindo a PR-AUC.

    Parameters
    ----------
    pipeline : sklearn.base.BaseEstimator
        Pipeline scikit-learn/imblearn não treinado (será clonado por fold).
    x : pandas.DataFrame
        Matriz de features.
    y : NDArray[np.int_]
        Alvo binário.
    n_folds : int, optional
        Número de folds, by default 5.
    seed : int, optional
        Semente do ``StratifiedKFold``, by default 42.

    Returns
    -------
    CVResult
        Média, desvio, IC 95% e a PR-AUC de cada fold.

    Examples
    --------
    >>> result = cross_validate_pr_auc(pipe, X, y, n_folds=5)  # doctest: +SKIP
    >>> result.mean > 0  # doctest: +SKIP
    True
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        model = cast(Estimator, clone(pipeline))
        model.fit(x.iloc[train_idx], y[train_idx])
        y_prob = model.predict_proba(x.iloc[val_idx])[:, 1]
        score = float(average_precision_score(y[val_idx], y_prob))
        scores.append(score)
        logger.info("Fold %d/%d — PR-AUC=%.4f", fold, n_folds, score)

    arr = np.array(scores)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci_95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    logger.info("PR-AUC CV: %.4f ± %.4f (IC 95%%)", mean, ci_95)
    return CVResult(mean=mean, std=std, ci_95=ci_95, scores=scores)
