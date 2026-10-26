"""Testes da configuração de logging (dictConfig, arquivo diário, nível, idempotência)."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from src.config.logging import (
    LOGGER_NAMESPACE,
    _rename_rotated,
    configure_logging,
    create_console_handler,
    create_daily_file_handler,
    get_logger,
)


@pytest.fixture
def _isolated_root_logger() -> Iterator[None]:
    """Salva e restaura o estado do logger raiz para não vazar entre testes.

    ``configure_logging`` substitui os handlers do logger raiz (inclusive o
    handler de captura do pytest); o fixture restaura o estado original e fecha
    os handlers de arquivo criados no teste.
    """
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for handler in root.handlers:
        if handler not in original_handlers:
            handler.close()
    root.handlers = original_handlers
    root.setLevel(original_level)
    logging.captureWarnings(False)


@pytest.fixture
def logging_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """YAML dictConfig mínimo que grava o arquivo de log em ``tmp_path``."""
    monkeypatch.setattr("src.config.logging.ROOT", tmp_path)
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"file": {"format": "%(levelname)s | %(name)s | %(message)s"}},
        "handlers": {
            "console": {"()": "src.config.logging.create_console_handler"},
            "file": {
                "()": "src.config.logging.create_daily_file_handler",
                "filename": "logs/log.log",
                "backup_count": 3,
                "formatter": "file",
            },
        },
        "root": {"level": "INFO", "handlers": ["console", "file"]},
    }
    path = tmp_path / "logging.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_configure_logging_creates_log_file_and_returns_namespace_logger(
    tmp_path: Path, logging_config: Path, _isolated_root_logger: None
) -> None:
    """``configure_logging`` cria o arquivo de log e retorna o logger do projeto."""
    logger = configure_logging(config_path=logging_config)
    get_logger("teste").info("mensagem de teste")
    for handler in logging.getLogger().handlers:
        handler.flush()

    log_file = tmp_path / "logs" / "log.log"
    assert logger.name == LOGGER_NAMESPACE
    assert log_file.exists()
    assert "mensagem de teste" in log_file.read_text(encoding="utf-8")


def test_configure_logging_level_argument_overrides_yaml(
    logging_config: Path, _isolated_root_logger: None
) -> None:
    """O argumento ``level`` sobrescreve o nível definido no YAML."""
    configure_logging(level="warning", config_path=logging_config)
    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_is_idempotent(logging_config: Path, _isolated_root_logger: None) -> None:
    """Reconfigurar não duplica handlers no root."""
    configure_logging(config_path=logging_config)
    configure_logging(config_path=logging_config)
    assert len(logging.getLogger().handlers) == 2


def test_configure_logging_raises_when_yaml_missing(tmp_path: Path) -> None:
    """YAML inexistente levanta ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        configure_logging(config_path=tmp_path / "ausente.yaml")


def test_configure_logging_routes_warnings_to_log_file(
    tmp_path: Path, logging_config: Path, _isolated_root_logger: None
) -> None:
    """``warnings.warn`` passa pelos mesmos handlers do logging."""
    configure_logging(config_path=logging_config)
    warnings.warn("aviso de teste", UserWarning, stacklevel=1)
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "aviso de teste" in (tmp_path / "logs" / "log.log").read_text(encoding="utf-8")


def test_rename_rotated_uses_date_suffix_before_extension() -> None:
    """O arquivo rotacionado vira ``log_AAAA-MM-DD.log``."""
    renamed = _rename_rotated(str(Path("logs") / "log.log.2026-09-27"))
    assert Path(renamed).name == "log_2026-09-27.log"


def test_create_daily_file_handler_creates_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O handler cria o diretório de logs e rotaciona à meia-noite."""
    monkeypatch.setattr("src.config.logging.ROOT", tmp_path)
    handler = create_daily_file_handler("logs/log.log", backup_count=5)
    try:
        assert (tmp_path / "logs").is_dir()
        assert handler.backupCount == 5
        assert handler.when == "MIDNIGHT"
    finally:
        handler.close()


def test_create_console_handler_shares_console() -> None:
    """O handler de console usa o ``Console`` compartilhado com ``rich.progress``."""
    from src.config.logging import CONSOLE

    assert create_console_handler().console is CONSOLE


def test_get_logger_uses_fraude_namespace() -> None:
    """``get_logger`` prefixa o nome do módulo com o namespace ``fraude``."""
    logger = get_logger("meu_modulo")
    assert logger.name == "fraude.meu_modulo"
