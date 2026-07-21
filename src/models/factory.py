"""Fábrica de estimadores (padrão Factory).

Centraliza a criação dos modelos suportados a partir dos hiperparâmetros
validados, sempre fixando ``random_state`` para reprodutibilidade.
"""

from __future__ import annotations

from enum import Enum

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config.settings import ModelParams
from src.exceptions.base import FraudDetectionError


class ModelName(str, Enum):
    """Modelos suportados pelo projeto."""

    LOGISTIC_REGRESSION = "logistic_regression"
    XGBOOST = "xgboost"


def create_estimator(
    name: ModelName | str,
    params: ModelParams,
    *,
    seed: int = 42,
    scale_pos_weight: float | None = None,
) -> ClassifierMixin:
    """Cria um estimador scikit-learn compatível a partir do nome e dos parâmetros.

    Parameters
    ----------
    name : ModelName or str
        Nome do modelo a instanciar.
    params : ModelParams
        Hiperparâmetros validados de todos os modelos.
    seed : int, optional
        Semente para ``random_state``, by default 42.
    scale_pos_weight : float, optional
        Peso da classe positiva para o XGBoost (n_neg / n_pos). Ignorado pelo
        baseline. Use ``None`` para não aplicar.

    Returns
    -------
    ClassifierMixin
        Estimador não treinado.

    Raises
    ------
    FraudDetectionError
        Se o nome do modelo for desconhecido.

    Examples
    --------
    >>> from src.config.settings import ModelParams
    >>> est = create_estimator("logistic_regression", ModelParams())
    >>> est.__class__.__name__
    'LogisticRegression'
    """
    name = ModelName(name)

    if name is ModelName.LOGISTIC_REGRESSION:
        lr = params.logistic_regression
        return LogisticRegression(
            C=lr.C,
            max_iter=lr.max_iter,
            class_weight=lr.class_weight,
            solver=lr.solver,
            random_state=seed,
            n_jobs=-1,
        )

    if name is ModelName.XGBOOST:
        xgb = params.xgboost
        return XGBClassifier(
            n_estimators=xgb.n_estimators,
            max_depth=xgb.max_depth,
            learning_rate=xgb.learning_rate,
            subsample=xgb.subsample,
            colsample_bytree=xgb.colsample_bytree,
            min_child_weight=xgb.min_child_weight,
            reg_lambda=xgb.reg_lambda,
            reg_alpha=xgb.reg_alpha,
            tree_method=xgb.tree_method,
            eval_metric=xgb.eval_metric,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
        )

    raise FraudDetectionError(f"Modelo desconhecido: {name}")  # pragma: no cover
