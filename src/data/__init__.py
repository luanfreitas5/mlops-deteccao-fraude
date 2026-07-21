"""Ingestão, carregamento, divisão e escrita de dados.

Módulos
-------
loader
    Carrega o CSV bruto de transações com Polars e valida o contrato.
splitter
    Divide os dados em treino/validação/teste (estratificado ou temporal).
writer
    Persiste DataFrames em Parquet (armazenamento colunar).
"""

from src.data.loader import load_raw_transactions
from src.data.splitter import split_data
from src.data.writer import write_parquet

__all__ = ["load_raw_transactions", "split_data", "write_parquet"]
