"""Montagem e ajuste do pipeline de treino.

O pipeline combina, na ordem: pré-processador (escalonamento), tratamento
opcional de desbalanceamento (SMOTE) e o estimador. Usamos ``imblearn.Pipeline``
para que o SMOTE seja aplicado somente no ajuste (nunca na validação/serving),
evitando vazamento. O ``scale_pos_weight`` do XGBoost é derivado dos dados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from numpy.typing import NDArray

from src.config.logging import get_logger
from src.config.settings import Settings
from src.constants.columns import TARGET_COLUMN
from src.features.engineering import engineer_features, get_model_features
from src.models.factory import ModelName, create_estimator
from src.preprocessing.pipeline import build_preprocessor

logger = get_logger(__name__)


def prepare_xy(
    df: pl.DataFrame, *, with_target: bool = True
) -> tuple[pd.DataFrame, NDArray[np.int_] | None]:
    """Aplica a engenharia de features e separa X (pandas) e y (numpy).

    Parameters
    ----------
    df : pl.DataFrame
        Transações (com ou sem a coluna alvo).
    with_target : bool, optional
        Se ``True`` (padrão), extrai também o alvo ``Class``.

    Returns
    -------
    tuple[pandas.DataFrame, numpy.ndarray or None]
        Matriz de features na ordem de ``get_model_features`` e o vetor alvo
        (``None`` quando ``with_target`` é ``False``).

    Examples
    --------
    >>> X, y = prepare_xy(df)  # doctest: +SKIP
    >>> list(X.columns) == get_model_features()  # doctest: +SKIP
    True
    """
    engineered = engineer_features(df)
    features = get_model_features()
    x = engineered.select(features).to_pandas()
    y: NDArray[np.int_] | None = None
    if with_target:
        y = engineered[TARGET_COLUMN].to_numpy()
    return x, y


def _compute_scale_pos_weight(y: NDArray[np.int_]) -> float:
    """Calcula ``n_negativos / n_positivos`` para o XGBoost."""
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def build_training_pipeline(
    model_name: ModelName | str,
    settings: Settings,
    y_train: NDArray[np.int_],
) -> ImbPipeline:
    """Monta o pipeline de treino conforme o modelo e a estratégia de desbalanceamento.

    Parameters
    ----------
    model_name : ModelName or str
        Modelo a treinar (baseline ou XGBoost).
    settings : Settings
        Configuração validada do projeto.
    y_train : NDArray[np.int_]
        Alvo de treino, usado para derivar ``scale_pos_weight``.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline não treinado com pré-processador, (opcional) SMOTE e modelo.

    Examples
    --------
    >>> pipe = build_training_pipeline("xgboost", settings, y_train)  # doctest: +SKIP
    """
    model_name = ModelName(model_name)
    strategy = settings.imbalance.strategy

    scale_pos_weight: float | None = None
    if strategy == "scale_pos_weight" and model_name is ModelName.XGBOOST:
        scale_pos_weight = _compute_scale_pos_weight(y_train)
        logger.info("scale_pos_weight derivado dos dados: %.2f", scale_pos_weight)

    estimator = create_estimator(
        model_name,
        settings.model,
        seed=settings.random_seed,
        scale_pos_weight=scale_pos_weight,
    )

    steps: list[tuple[str, object]] = [("preprocessor", build_preprocessor())]
    if strategy == "smote":
        steps.append(
            (
                "smote",
                SMOTE(
                    # imblearn aceita float (razão minoria/maioria) em runtime;
                    # o stub restringe a str, por isso a supressão pontual.
                    sampling_strategy=settings.imbalance.smote_sampling_strategy,  # pyright: ignore[reportArgumentType]
                    random_state=settings.random_seed,
                ),
            )
        )
    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)


def train_model(
    pipeline: ImbPipeline,
    x_train: pd.DataFrame,
    y_train: NDArray[np.int_],
) -> ImbPipeline:
    """Treina o pipeline com os dados de treino.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline
        Pipeline não treinado.
    x_train : pandas.DataFrame
        Matriz de features de treino.
    y_train : NDArray[np.int_]
        Alvo de treino.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline treinado.

    Examples
    --------
    >>> fitted = train_model(pipe, X_train, y_train)  # doctest: +SKIP
    """
    logger.info("Treinando pipeline com %d amostras", len(x_train))
    pipeline.fit(x_train, y_train)
    logger.info("Treino concluído")
    return pipeline
