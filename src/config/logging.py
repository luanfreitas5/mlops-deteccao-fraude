"""Configuração única de logging: console ``rich`` + arquivo com rotação diária.

O logging é configurado **uma vez**, no ponto de entrada, a partir de
``configs/logging.yaml`` (formato ``logging.config.dictConfig``). Os demais
módulos apenas obtêm um logger com :func:`get_logger` — nunca configuram
handlers. Todas as mensagens são em pt-BR e nunca devem conter PII ou segredos
(registre contagens, formas e métricas, jamais linhas de dados).
"""

from __future__ import annotations

import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler

from src.config.paths import ROOT

CONSOLE = Console(stderr=True)  # compartilhado pelo RichHandler e por rich.progress
LOGGER_NAMESPACE = "fraude"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "logging.yaml"


def create_console_handler() -> RichHandler:
    """Cria o handler de console ligado ao console compartilhado.

    Returns
    -------
    RichHandler
        Handler colorido com tracebacks formatados.
    """
    return RichHandler(console=CONSOLE, rich_tracebacks=True, show_path=True, markup=False)


def _rename_rotated(default_name: str) -> str:
    """Converte ``logs/log.log.2026-09-27`` em ``logs/log_2026-09-27.log``."""
    path = Path(default_name)
    stem, date_str = path.name.split(".log.")
    return str(path.with_name(f"{stem}_{date_str}.log"))


def create_daily_file_handler(filename: str, backup_count: int = 30) -> TimedRotatingFileHandler:
    """Cria o handler de arquivo com rotação diária à meia-noite.

    Parameters
    ----------
    filename : str
        Caminho do arquivo ativo, relativo à raiz do projeto.
    backup_count : int, optional
        Quantidade de arquivos diários mantidos, by default 30.

    Returns
    -------
    TimedRotatingFileHandler
        Handler que rotaciona para ``log_YYYY-MM-DD.log``.

    Examples
    --------
    >>> handler = create_daily_file_handler("logs/log.log")  # doctest: +SKIP
    """
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)  # dictConfig falha se o diretório não existe
    handler = TimedRotatingFileHandler(
        path, when="midnight", backupCount=backup_count, encoding="utf-8"
    )
    handler.namer = _rename_rotated
    return handler


def configure_logging(level: str | None = None, config_path: Path | None = None) -> logging.Logger:
    """Aplica a configuração de logging do YAML. Chamar apenas no ponto de entrada.

    Parameters
    ----------
    level : str, optional
        Sobrescreve o nível do root (ex.: ``--log-level DEBUG``), by default None.
    config_path : Path, optional
        YAML no formato ``dictConfig``. Por padrão ``configs/logging.yaml``.

    Returns
    -------
    logging.Logger
        Logger raiz do projeto (namespace ``fraude``).

    Raises
    ------
    FileNotFoundError
        Se o YAML não existir.
    ValueError
        Se o YAML ou o nível informado forem inválidos.

    Examples
    --------
    >>> logger = configure_logging(level="DEBUG")  # doctest: +SKIP
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if level is not None:
        config["root"]["level"] = level.upper()
    logging.config.dictConfig(config)  # substitui os handlers do root: reconfigurar não duplica
    logging.captureWarnings(True)  # warnings.warn() passa pelos mesmos handlers
    return logging.getLogger(LOGGER_NAMESPACE)


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger nomeado sob o namespace ``fraude``.

    Parameters
    ----------
    name : str
        Nome do módulo/componente (tipicamente ``__name__``).

    Returns
    -------
    logging.Logger
        Logger filho; os handlers ficam apenas no root.

    Examples
    --------
    >>> logger = get_logger(__name__)
    """
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")
