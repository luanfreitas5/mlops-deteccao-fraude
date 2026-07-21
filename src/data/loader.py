"""Carregamento do dataset bruto de transações com Polars.

Lê o CSV original, valida o contrato de dados (pandera) e retorna um
``pl.DataFrame`` pronto para as etapas seguintes. Nunca modifica o arquivo bruto.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config.logging import get_logger
from src.config.paths import get_paths
from src.constants.columns import ALL_COLUMNS
from src.exceptions.data import RawDataNotFoundError
from src.schemas.dataset import validate_transactions

logger = get_logger(__name__)


def load_raw_transactions(
    path: Path | None = None,
    *,
    validate: bool = True,
) -> pl.DataFrame:
    """Carrega as transações brutas do CSV e opcionalmente valida o contrato.

    Parameters
    ----------
    path : Path, optional
        Caminho do CSV bruto. Por padrão usa ``data/raw/creditcard.csv``.
    validate : bool, optional
        Se ``True`` (padrão), valida o DataFrame contra o contrato de dados.

    Returns
    -------
    pl.DataFrame
        Transações carregadas com as colunas esperadas.

    Raises
    ------
    RawDataNotFoundError
        Se o arquivo bruto não existir.
    DataValidationError
        Se ``validate`` for ``True`` e os dados violarem o contrato.

    Examples
    --------
    >>> df = load_raw_transactions()  # doctest: +SKIP
    >>> "Class" in df.columns  # doctest: +SKIP
    True
    """
    path = path or get_paths().data_raw
    if not path.exists():
        raise RawDataNotFoundError(
            f"Dados brutos não encontrados em {path}. "
            "Baixe o dataset Credit Card Fraud (ULB) para data/raw/."
        )

    logger.info("Carregando transações brutas de %s", path)
    df = pl.read_csv(path, schema_overrides={"Time": pl.Float64})

    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise RawDataNotFoundError(f"Colunas ausentes no arquivo bruto: {sorted(missing)}")

    df = df.select(ALL_COLUMNS)
    logger.info("Transações carregadas: %d linhas, %d colunas", df.height, df.width)

    if validate:
        df = validate_transactions(df)
    return df
