"""Divisão dos dados em treino, validação e teste.

Suporta duas estratégias:

- ``stratified``: mantém a proporção de fraudes em cada partição (padrão).
- ``temporal``: ordena por ``Time`` e separa o passado (treino) do futuro
  (teste), espelhando o cenário real de produção e evitando vazamento temporal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from src.config.logging import get_logger
from src.config.settings import SplitConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """Resultado da divisão dos dados.

    Attributes
    ----------
    train, val, test : pl.DataFrame
        Partições de treino, validação e teste (incluem a coluna alvo).
    """

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def _stratified_split(df: pl.DataFrame, config: SplitConfig, seed: int) -> DataSplit:
    """Divide de forma estratificada pela coluna alvo."""
    target = df[config.target_column].to_numpy()
    idx = pl.arange(0, df.height, eager=True).to_numpy()

    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=config.test_size,
        stratify=target,
        random_state=seed,
    )
    val_ratio = config.val_size / (1.0 - config.test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_ratio,
        stratify=target[idx_train_val],
        random_state=seed,
    )
    return DataSplit(
        train=df[np.asarray(idx_train).tolist()],
        val=df[np.asarray(idx_val).tolist()],
        test=df[np.asarray(idx_test).tolist()],
    )


def _temporal_split(df: pl.DataFrame, config: SplitConfig) -> DataSplit:
    """Divide temporalmente por ``Time`` (passado -> treino, futuro -> teste)."""
    df_sorted = df.sort(config.time_column)
    n = df_sorted.height
    n_test = int(n * config.test_size)
    n_val = int(n * config.val_size)
    n_train = n - n_val - n_test
    return DataSplit(
        train=df_sorted.slice(0, n_train),
        val=df_sorted.slice(n_train, n_val),
        test=df_sorted.slice(n_train + n_val, n_test),
    )


def split_data(df: pl.DataFrame, config: SplitConfig, seed: int = 42) -> DataSplit:
    """Divide o DataFrame em treino/validação/teste conforme a estratégia.

    Parameters
    ----------
    df : pl.DataFrame
        Transações completas (com a coluna alvo).
    config : SplitConfig
        Configuração de divisão (estratégia, tamanhos, colunas).
    seed : int, optional
        Semente para a divisão estratificada, by default 42.

    Returns
    -------
    DataSplit
        Partições de treino, validação e teste.

    Raises
    ------
    ValueError
        Se a estratégia de divisão for desconhecida.

    Examples
    --------
    >>> split = split_data(df, SplitConfig(), seed=42)  # doctest: +SKIP
    >>> split.train.height > split.test.height  # doctest: +SKIP
    True
    """
    if config.strategy == "stratified":
        split = _stratified_split(df, config, seed)
    elif config.strategy == "temporal":
        split = _temporal_split(df, config)
    else:  # pragma: no cover - protegido por validação Pydantic
        raise ValueError(f"Estratégia de divisão desconhecida: {config.strategy}")

    logger.info(
        "Divisão '%s': treino=%d, val=%d, teste=%d",
        config.strategy,
        split.train.height,
        split.val.height,
        split.test.height,
    )
    return split
