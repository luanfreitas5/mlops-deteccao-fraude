"""Hashing de arquivos para rastrear a versão dos dados (reprodutibilidade)."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1 << 20  # 1 MiB


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Retorna o hash de um arquivo, lido em blocos (adequado a arquivos grandes).

    Parameters
    ----------
    path : Path
        Caminho do arquivo a ser hasheado.
    algorithm : str, optional
        Algoritmo de hash suportado por ``hashlib``, by default ``"sha256"``.

    Returns
    -------
    str
        Digest hexadecimal do arquivo.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> digest = hash_file(Path("data/raw/creditcard.csv"))  # doctest: +SKIP
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para hashing: {path}")

    hasher = hashlib.new(algorithm)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
