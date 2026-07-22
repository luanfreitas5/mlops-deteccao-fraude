"""Testes de carregamento, limpeza, divisão e escrita de dados."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.config.settings import SplitConfig
from src.constants.columns import ALL_COLUMNS
from src.data.loader import load_raw_transactions
from src.data.splitter import split_data
from src.data.writer import write_parquet
from src.exceptions.data import RawDataNotFoundError
from src.preprocessing.cleaning import drop_duplicates


def test_load_raw_missing_file_raises(tmp_path: Path) -> None:
    """Arquivo inexistente levanta RawDataNotFoundError."""
    with pytest.raises(RawDataNotFoundError):
        load_raw_transactions(tmp_path / "inexistente.csv")


def test_load_raw_transactions_reads_and_validates(
    synthetic_transactions: pl.DataFrame, tmp_path: Path
) -> None:
    """Um CSV bruto válido é carregado, reordenado e validado com sucesso."""
    path = tmp_path / "creditcard.csv"
    synthetic_transactions.write_csv(path)
    df = load_raw_transactions(path)
    assert df.columns == ALL_COLUMNS
    assert df.height == synthetic_transactions.height


def test_load_raw_transactions_can_skip_validation(
    synthetic_transactions: pl.DataFrame, tmp_path: Path
) -> None:
    """Com ``validate=False``, o contrato de dados não é aplicado."""
    path = tmp_path / "creditcard.csv"
    synthetic_transactions.write_csv(path)
    df = load_raw_transactions(path, validate=False)
    assert df.height == synthetic_transactions.height


def test_load_raw_transactions_missing_columns_raises(
    synthetic_transactions: pl.DataFrame, tmp_path: Path
) -> None:
    """CSV bruto sem uma coluna esperada levanta RawDataNotFoundError."""
    path = tmp_path / "creditcard.csv"
    synthetic_transactions.drop("V1").write_csv(path)
    with pytest.raises(RawDataNotFoundError):
        load_raw_transactions(path)


def test_drop_duplicates_removes_exact_copies(
    synthetic_transactions: pl.DataFrame,
) -> None:
    """Linhas idênticas são removidas."""
    doubled = pl.concat([synthetic_transactions, synthetic_transactions])
    deduped = drop_duplicates(doubled)
    assert deduped.height == synthetic_transactions.height


def test_drop_duplicates_is_noop_without_duplicates(
    synthetic_transactions: pl.DataFrame,
) -> None:
    """Sem linhas duplicadas, o DataFrame retorna inalterado."""
    deduped = drop_duplicates(synthetic_transactions)
    assert deduped.height == synthetic_transactions.height


def test_split_data_stratified_partitions_are_disjoint(
    synthetic_transactions: pl.DataFrame,
) -> None:
    """As partições estratificadas somam o total e preservam ambas as classes."""
    config = SplitConfig(strategy="stratified", test_size=0.2, val_size=0.2)
    split = split_data(synthetic_transactions, config, seed=42)
    total = split.train.height + split.val.height + split.test.height
    assert total == synthetic_transactions.height
    for part in (split.train, split.val, split.test):
        assert part["Class"].n_unique() == 2


def test_split_data_temporal_respects_time_order(
    synthetic_transactions: pl.DataFrame,
) -> None:
    """Na divisão temporal, o treino precede o teste no tempo."""
    config = SplitConfig(strategy="temporal", test_size=0.2, val_size=0.2)
    split = split_data(synthetic_transactions, config)
    assert split.train["Time"].max() <= split.test["Time"].max()


def test_write_parquet_roundtrip(synthetic_transactions: pl.DataFrame, tmp_path: Path) -> None:
    """O Parquet escrito pode ser relido sem perda."""
    path = write_parquet(synthetic_transactions, tmp_path / "data.parquet")
    reloaded = pl.read_parquet(path)
    assert reloaded.height == synthetic_transactions.height
