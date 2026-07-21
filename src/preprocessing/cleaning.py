"""Limpeza de dados brutos de transações.

O dataset Credit Card Fraud contém transações duplicadas exatas que inflam
artificialmente algumas classes; removê-las evita vazamento entre treino e teste
quando linhas idênticas caem em partições diferentes.
"""

from __future__ import annotations

import polars as pl

from src.config.logging import get_logger

logger = get_logger(__name__)


def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Remove linhas totalmente duplicadas do DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de transações.

    Returns
    -------
    pl.DataFrame
        DataFrame sem linhas duplicadas.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"a": [1, 1], "b": [2, 2]})
    >>> drop_duplicates(df).height
    1
    """
    before = df.height
    deduped = df.unique(keep="first", maintain_order=True)
    removed = before - deduped.height
    if removed:
        logger.info("Removidas %d linhas duplicadas", removed)
    return deduped
