"""Testes do contrato de dados (pandera) e dos schemas da API."""

from __future__ import annotations

import polars as pl
import pytest

from src.constants.columns import FEATURE_COLUMNS
from src.exceptions.data import DataValidationError
from src.schemas.dataset import validate_transactions
from src.schemas.prediction import TransactionRequest, build_feature_frame


@pytest.mark.smoke
def test_validate_transactions_accepts_valid(synthetic_transactions: pl.DataFrame) -> None:
    """Dados válidos passam pelo contrato sem erro."""
    result = validate_transactions(synthetic_transactions)
    assert result.height == synthetic_transactions.height


def test_validate_transactions_rejects_negative_amount(
    synthetic_transactions: pl.DataFrame,
) -> None:
    """Valor negativo viola o contrato e levanta DataValidationError."""
    corrupted = synthetic_transactions.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0).then(-1.0).otherwise(pl.col("Amount")).alias("Amount")
    )
    with pytest.raises(DataValidationError):
        validate_transactions(corrupted)


def test_validate_transactions_rejects_invalid_class(
    synthetic_transactions: pl.DataFrame,
) -> None:
    """Rótulo fora de {0, 1} viola o contrato."""
    corrupted = synthetic_transactions.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0).then(2).otherwise(pl.col("Class")).alias("Class")
    )
    with pytest.raises(DataValidationError):
        validate_transactions(corrupted)


def test_build_feature_frame_orders_columns() -> None:
    """O frame construído a partir da requisição respeita a ordem das features."""
    payload = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    request = TransactionRequest(**payload)
    frame = build_feature_frame(request)
    assert frame.columns == FEATURE_COLUMNS
    assert frame.height == 1
