"""Configura o logging do projeto com ``RichHandler`` e rotação diária de arquivo.

Todos os logs são emitidos em pt-BR, com console colorido (rich) e um arquivo
diário em ``logs/log_YYYY-MM-DD.log``. O nível é configurável via
``configs/logging.yaml``. Nunca registre PII ou segredos.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from rich.logging import RichHandler

from src.config.paths import get_paths

_DEFAULT_FORMAT = "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _load_logging_config(configs_dir: Path | None = None) -> dict[str, Any]:
    """Lê ``configs/logging.yaml`` retornando um dicionário com defaults seguros."""
    configs_dir = configs_dir or get_paths().configs
    path = configs_dir / "logging.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def configure_logging(configs_dir: Path | None = None) -> logging.Logger:
    """Configura os handlers de console e arquivo e retorna o logger raiz do projeto.

    Parameters
    ----------
    configs_dir : Path, optional
        Diretório dos YAML de configuração. Por padrão usa ``configs/``.

    Returns
    -------
    logging.Logger
        Logger raiz configurado (nome ``"fraude"``).

    Examples
    --------
    >>> logger = configure_logging()
    >>> logger.info("Pipeline iniciado")
    """
    config = _load_logging_config(configs_dir)
    level = str(config.get("level", "INFO")).upper()
    file_format = config.get("file_format", _DEFAULT_FORMAT)
    date_format = config.get("date_format", _DEFAULT_DATE_FORMAT)
    show_path = bool(config.get("show_path", True))

    log_dir = get_paths().logs
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log_{date.today().isoformat()}.log"

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=show_path, markup=True),
    ]
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(file_format, datefmt=date_format))
    handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(level)
    # Evita duplicar handlers em reconfigurações (ex.: chamadas repetidas em testes).
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)

    logger = logging.getLogger("fraude")
    logger.debug("Logging configurado no nível %s (arquivo: %s)", level, log_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger nomeado sob o namespace ``fraude``.

    Parameters
    ----------
    name : str
        Nome do módulo/componente (tipicamente ``__name__``).

    Returns
    -------
    logging.Logger
        Logger filho pronto para uso.
    """
    return logging.getLogger(f"fraude.{name}")
