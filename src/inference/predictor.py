"""Predição de fraude a partir do pipeline treinado.

Reaplica exatamente a engenharia de features e a seleção de colunas usadas no
treino (via :func:`prepare_xy`), evitando *train-serve skew*. O threshold de
decisão é injetado no construtor e deve refletir o valor otimizado na avaliação.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.config.logging import get_logger
from src.models.base import Estimator
from src.models.persistence import load_model
from src.training.trainer import prepare_xy

logger = get_logger(__name__)


class FraudPredictor:
    """Encapsula um pipeline treinado para predição de fraude.

    Parameters
    ----------
    pipeline : Estimator
        Pipeline scikit-learn/imblearn treinado (pré-processador + modelo).
    threshold : float
        Threshold de decisão para classificar como fraude (>= threshold).
    """

    def __init__(self, pipeline: Estimator, threshold: float = 0.5) -> None:
        self._pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def from_path(cls, model_path: Path, threshold: float = 0.5) -> FraudPredictor:
        """Cria um preditor carregando o pipeline serializado do disco.

        Parameters
        ----------
        model_path : Path
            Caminho do artefato ``.joblib``.
        threshold : float, optional
            Threshold de decisão, by default 0.5.

        Returns
        -------
        FraudPredictor
            Preditor pronto para uso.
        """
        pipeline = load_model(model_path)
        return cls(pipeline=pipeline, threshold=threshold)

    def predict_proba(self, df: pl.DataFrame) -> NDArray[np.float64]:
        """Retorna a probabilidade de fraude para cada transação.

        Parameters
        ----------
        df : pl.DataFrame
            Transações de entrada (com as colunas brutas do dataset).

        Returns
        -------
        NDArray[np.float64]
            Probabilidades da classe positiva (fraude), uma por linha.

        Examples
        --------
        >>> proba = predictor.predict_proba(df)  # doctest: +SKIP
        """
        x, _ = prepare_xy(df, with_target=False)
        return self._pipeline.predict_proba(x)[:, 1]

    def predict(self, df: pl.DataFrame) -> NDArray[np.int_]:
        """Retorna a decisão binária (fraude/legítima) por transação.

        Parameters
        ----------
        df : pl.DataFrame
            Transações de entrada.

        Returns
        -------
        NDArray[np.int_]
            Vetor binário: 1 = fraude, 0 = legítima (aplicando o threshold).
        """
        proba = self.predict_proba(df)
        return (proba >= self.threshold).astype(int)
