"""Escrita de DataFrames em Parquet (armazenamento colunar) e manifesto da camada bronze."""

from __future__ import annotations

import json
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
    >>> write_parquet(df, Path("data/gold/train.parquet"))  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Parquet escrito em %s (%d linhas)", path, df.height)
    return path


def write_manifest(source: Path, sha256: str, path: Path | None = None) -> Path:
    """Grava o manifesto de um arquivo da camada bronze (nome, tamanho e SHA-256).

    O manifesto permite detectar alterações silenciosas no dado bruto, que é
    imutável: qualquer mudança de hash entre execuções deve ser investigada.

    Parameters
    ----------
    source : Path
        Arquivo bruto descrito pelo manifesto.
    sha256 : str
        Hash SHA-256 do arquivo (ver :func:`utils.hashing.hash_file`).
    path : Path, optional
        Destino do manifesto. Por padrão ``manifest.json`` ao lado do arquivo.

    Returns
    -------
    Path
        Caminho do manifesto escrito.

    Examples
    --------
    >>> write_manifest(Path("data/bronze/creditcard.csv"), "ab12...")  # doctest: +SKIP
    """
    source = Path(source)
    path = Path(path) if path is not None else source.with_name("manifest.json")
    manifest = {"file": source.name, "bytes": source.stat().st_size, "sha256": sha256}
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Manifesto da camada bronze escrito em %s", path)
    return path
