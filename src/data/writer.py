"""Escrita de DataFrames em Parquet (armazenamento colunar via PyArrow)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config.logging import get_logger

logger = get_logger(__name__)


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Escreve um DataFrame em Parquet, criando o diretório pai se necessário.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame a persistir.
    path : Path
        Caminho de destino do arquivo Parquet.

    Returns
    -------
    Path
        O caminho escrito.

    Examples
    --------
    >>> write_parquet(df, Path("data/processed/train.parquet"))  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Parquet escrito em %s (%d linhas)", path, df.height)
    return path
