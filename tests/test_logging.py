"""Testes da configuração de logging (handlers, arquivo diário, YAML)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from src.config.logging import _load_logging_config, configure_logging, get_logger
from src.config.paths import ProjectPaths


@pytest.fixture
def _isolated_root_logger() -> Iterator[None]:
    """Salva e restaura o estado do logger raiz para não vazar entre testes.

    ``configure_logging`` limpa e substitui os handlers do logger raiz (inclusive
    o handler de captura do pytest); este fixture garante que o estado original
    seja restaurado e que os handlers de arquivo criados no teste sejam fechados.
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


def test_load_logging_config_returns_empty_dict_when_missing(tmp_path: Path) -> None:
    """Sem ``logging.yaml``, retorna um dicionário vazio."""
    assert _load_logging_config(tmp_path) == {}


def test_load_logging_config_reads_yaml(tmp_path: Path) -> None:
    """Lê o nível definido no YAML de configuração de logging."""
    (tmp_path / "logging.yaml").write_text("level: DEBUG\n", encoding="utf-8")
    config = _load_logging_config(tmp_path)
    assert config["level"] == "DEBUG"


def test_configure_logging_creates_daily_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_root_logger: None
) -> None:
    """``configure_logging`` cria o arquivo de log diário no diretório configurado."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "logging.yaml").write_text("level: INFO\n", encoding="utf-8")

    fake_paths = ProjectPaths(root=tmp_path, logs=tmp_path / "logs")
    monkeypatch.setattr("src.config.logging.get_paths", lambda: fake_paths)

    logger = configure_logging(configs_dir)
    expected_log_file = tmp_path / "logs" / f"log_{date.today().isoformat()}.log"
    assert expected_log_file.exists()
    assert logger.name == "fraude"
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_applies_custom_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_root_logger: None
) -> None:
    """Um nível customizado no YAML é aplicado ao logger raiz."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "logging.yaml").write_text("level: WARNING\n", encoding="utf-8")

    fake_paths = ProjectPaths(root=tmp_path, logs=tmp_path / "logs")
    monkeypatch.setattr("src.config.logging.get_paths", lambda: fake_paths)

    configure_logging(configs_dir)
    assert logging.getLogger().level == logging.WARNING


def test_get_logger_uses_fraude_namespace() -> None:
    """``get_logger`` prefixa o nome do módulo com o namespace ``fraude``."""
    logger = get_logger("meu_modulo")
    assert logger.name == "fraude.meu_modulo"
