"""Pipeline de pré-processamento: raw -> processed.

Carrega o CSV bruto, valida o contrato, remove duplicatas, divide em
treino/validação/teste e persiste cada partição em Parquet. Retorna também o
hash do arquivo bruto para rastreabilidade.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config.logging import get_logger
from src.config.paths import get_paths
from src.config.settings import Settings
from src.data.loader import load_raw_transactions
from src.data.splitter import DataSplit, split_data
from src.data.writer import write_parquet
from src.preprocessing.cleaning import drop_duplicates
from src.utils.hashing import hash_file

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessingOutput:
    """Saída do pipeline de pré-processamento.

    Attributes
    ----------
    split : DataSplit
        Partições treino/validação/teste.
    data_hash : str
        Hash SHA-256 do arquivo bruto (linhagem).
    """

    split: DataSplit
    data_hash: str


def run_preprocessing_pipeline(settings: Settings) -> PreprocessingOutput:
    """Executa o pré-processamento completo e persiste os dados processados.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.

    Returns
    -------
    PreprocessingOutput
        Partições e hash dos dados brutos.

    Examples
    --------
    >>> output = run_preprocessing_pipeline(settings)  # doctest: +SKIP
    """
    paths = get_paths()
    paths.ensure_dirs()

    data_hash = hash_file(paths.data_raw)
    logger.info("Hash dos dados brutos: %s", data_hash)

    df = load_raw_transactions(paths.data_raw)
    df = drop_duplicates(df)
    split = split_data(df, settings.split, seed=settings.random_seed)

    write_parquet(split.train, paths.data_processed / "train.parquet")
    write_parquet(split.val, paths.data_processed / "val.parquet")
    write_parquet(split.test, paths.data_processed / "test.parquet")

    return PreprocessingOutput(split=split, data_hash=data_hash)
