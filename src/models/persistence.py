"""Persistência de modelos e metadados (joblib + JSON).

Salva o pipeline treinado (pré-processador + modelo) junto de um arquivo de
metadados com informações de linhagem: hash dos dados, threshold de decisão,
métricas e SHA do Git — essenciais para reprodutibilidade e auditoria.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.config.logging import get_logger
from src.exceptions.model import ModelNotFoundError

logger = get_logger(__name__)


def save_model(
    model: Any,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serializa o modelo com joblib e opcionalmente grava metadados ao lado.

    Parameters
    ----------
    model : Any
        Estimador ou pipeline treinado a persistir.
    path : Path
        Caminho de destino do arquivo ``.joblib``.
    metadata : dict[str, Any], optional
        Metadados de linhagem gravados em ``<path>.json`` (mesmo diretório).

    Returns
    -------
    Path
        Caminho do modelo salvo.

    Examples
    --------
    >>> save_model(pipeline, Path("models/fraud_model.joblib"))  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Modelo salvo em %s", path)

    if metadata is not None:
        meta_path = path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        logger.info("Metadados do modelo salvos em %s", meta_path)
    return path


def load_model(path: Path) -> Any:
    """Carrega um modelo serializado com joblib.

    Parameters
    ----------
    path : Path
        Caminho do arquivo ``.joblib``.

    Returns
    -------
    Any
        Modelo/pipeline desserializado.

    Raises
    ------
    ModelNotFoundError
        Se o artefato não existir.

    Examples
    --------
    >>> model = load_model(Path("models/fraud_model.joblib"))  # doctest: +SKIP
    """
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"Modelo não encontrado em {path}")
    logger.info("Carregando modelo de %s", path)
    return joblib.load(path)
