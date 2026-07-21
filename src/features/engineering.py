"""Engenharia de features para detecção de fraude.

As colunas V1-V28 já são componentes de PCA e não requerem transformação. As
features derivadas focam em ``Amount`` (fortemente assimétrico) e ``Time``
(convertido em hora do dia), sem introduzir vazamento — cada transformação
depende apenas da própria linha.
"""

from __future__ import annotations

import polars as pl

from src.constants.columns import AMOUNT_COLUMN, PCA_COLUMNS, TIME_COLUMN

# Features criadas por este módulo.
ENGINEERED_COLUMNS: list[str] = ["Amount_log", "Hour"]

_SECONDS_PER_HOUR = 3600
_HOURS_PER_DAY = 24


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Adiciona features derivadas ao DataFrame de transações.

    Cria:

    - ``Amount_log``: ``log1p(Amount)`` para reduzir a assimetria do valor.
    - ``Hour``: hora do dia (0-23) derivada de ``Time`` (segundos).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame contendo ao menos ``Time`` e ``Amount``.

    Returns
    -------
    pl.DataFrame
        DataFrame com as colunas de :data:`ENGINEERED_COLUMNS` adicionadas.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"Time": [3600.0], "Amount": [99.0]})
    >>> out = engineer_features(df)
    >>> "Amount_log" in out.columns and "Hour" in out.columns
    True
    """
    return df.with_columns(
        (pl.col(AMOUNT_COLUMN) + 1).log().alias("Amount_log"),
        ((pl.col(TIME_COLUMN) / _SECONDS_PER_HOUR) % _HOURS_PER_DAY).alias("Hour"),
    )


def get_model_features() -> list[str]:
    """Retorna a lista ordenada de colunas usadas como entrada do modelo.

    Exclui ``Time`` bruto (substituído por ``Hour``) e mantém ``Amount`` junto de
    suas derivadas.

    Returns
    -------
    list[str]
        Nomes das features de entrada do modelo.

    Examples
    --------
    >>> feats = get_model_features()
    >>> "V1" in feats and "Time" not in feats
    True
    """
    return [*PCA_COLUMNS, AMOUNT_COLUMN, *ENGINEERED_COLUMNS]
