"""Ingestão, carregamento, divisão e escrita de dados.

Módulos
-------
loader
    Carrega o CSV bruto de transações com Polars e valida o contrato.
splitter
    Divide os dados em treino/validação/teste (estratificado ou temporal).
writer
    Persiste DataFrames em Parquet e grava o manifesto da camada bronze.

Camadas de dados (medalhão): ``data/bronze`` (como recebido), ``data/silver``
(limpo e validado) e ``data/gold`` (partições prontas para modelagem).
"""

from src.data.loader import load_raw_transactions
from src.data.splitter import split_data
from src.data.writer import write_manifest, write_parquet

__all__ = ["load_raw_transactions", "split_data", "write_manifest", "write_parquet"]
