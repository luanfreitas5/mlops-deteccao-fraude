"""Pipeline de pré-processamento: bronze -> silver -> gold.

Carrega o CSV bruto (bronze, imutável), valida o contrato, remove duplicatas e
persiste a tabela limpa em Parquet (silver). Em seguida divide em
treino/validação/teste e persiste cada partição (gold). Retorna também o hash do
arquivo bruto para rastreabilidade.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config.logging import get_logger
from src.config.paths import get_paths
from src.config.settings import Settings
from src.data.loader import load_raw_transactions
from src.data.splitter import DataSplit, split_data
from src.data.writer import write_manifest, write_parquet
from src.preprocessing.cleaning import drop_duplicates
from src.utils.hashing import hash_file

logger = get_logger(__name__)

SILVER_FILENAME = "transactions.parquet"


@dataclass(frozen=True)
class PreprocessingOutput:
    """Saída do pipeline de pré-processamento.

    Attributes
    ----------
    split : DataSplit
        Partições treino/validação/teste (camada gold).
    data_hash : str
        Hash SHA-256 do arquivo bruto (linhagem).
    """

    split: DataSplit
    data_hash: str


def run_preprocessing_pipeline(settings: Settings) -> PreprocessingOutput:
    """Executa o pré-processamento completo (bronze -> silver -> gold).

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.

    Returns
    -------
    PreprocessingOutput
        Partições e hash dos dados brutos.

    Raises
    ------
    RawDataNotFoundError
        Se o CSV da camada bronze não existir.
    DataValidationError
        Se os dados violarem o contrato (pandera).

    Examples
    --------
    >>> output = run_preprocessing_pipeline(settings)  # doctest: +SKIP
    """
    paths = get_paths()
    paths.ensure_dirs()

    data_hash = hash_file(paths.data_bronze)
    logger.info("Hash dos dados brutos: %s", data_hash)
    write_manifest(paths.data_bronze, data_hash)

    # Bronze -> silver: validado contra o contrato e deduplicado.
    transactions_df = load_raw_transactions(paths.data_bronze)
    transactions_df = drop_duplicates(transactions_df)
    write_parquet(transactions_df, paths.data_silver / SILVER_FILENAME)

    # Silver -> gold: partições prontas para modelagem.
    split = split_data(transactions_df, settings.split, seed=settings.random_seed)
    write_parquet(split.train, paths.data_gold / "train.parquet")
    write_parquet(split.val, paths.data_gold / "val.parquet")
    write_parquet(split.test, paths.data_gold / "test.parquet")

    return PreprocessingOutput(split=split, data_hash=data_hash)
