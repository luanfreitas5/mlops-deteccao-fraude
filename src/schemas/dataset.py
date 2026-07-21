"""Contrato de dados (pandera) para as transações de cartão de crédito.

Valida o DataFrame nas fronteiras do pipeline (raw -> processed e entrada do
modelo), falhando cedo com um erro claro em vez de propagar corrupção silenciosa.
Checa tipos, faixas, nulos e categorias aceitas do alvo.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing.polars import Series

from src.constants.labels import FRAUD, LEGITIMATE
from src.exceptions.data import DataValidationError


class TransactionsSchema(pa.DataFrameModel):
    """Contrato para o dataset Credit Card Fraud (Time, V1-V28, Amount, Class).

    Notes
    -----
    As colunas V1-V28 são componentes de PCA (podem assumir qualquer valor real),
    portanto validam-se apenas tipo e ausência de nulos. ``Time`` e ``Amount``
    são não negativos e ``Class`` só aceita 0 (legítima) ou 1 (fraude).
    """

    Time: Series[float] = pa.Field(ge=0, nullable=False)
    Amount: Series[float] = pa.Field(ge=0, nullable=False)
    Class: Series[int] = pa.Field(isin=[LEGITIMATE, FRAUD], nullable=False)

    # A classe aninhada Config é o padrão documentado do pandera; o pyright a vê
    # como sobrescrita incompatível do atributo herado de DataFrameModel.
    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = False  # V1-V28 são validadas dinamicamente (mesmo dtype numérico)
        coerce = True


def validate_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame de transações contra o contrato de dados.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de transações a validar.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, se válido.

    Raises
    ------
    DataValidationError
        Se o DataFrame violar o contrato (tipos, faixas ou nulos inválidos).

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {"Time": [0.0], "Amount": [1.0], "Class": [0], **{f"V{i}": [0.0] for i in range(1, 29)}}
    ... )
    >>> _ = validate_transactions(df)
    """
    try:
        return TransactionsSchema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataValidationError(f"Transações violaram o contrato de dados: {exc}") from exc
