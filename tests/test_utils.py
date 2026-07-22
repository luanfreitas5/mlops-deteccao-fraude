"""Testes dos utilitários de hashing, reexportação de seed e cronometragem."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.config.environment import seed_everything as canonical_seed_everything
from src.utils.hashing import hash_file
from src.utils.seed import RANDOM_SEED, seed_everything
from src.utils.timing import timed


def test_hash_file_is_deterministic(tmp_path: Path) -> None:
    """O mesmo conteúdo sempre produz o mesmo hash."""
    path = tmp_path / "arquivo.txt"
    path.write_text("conteudo de teste", encoding="utf-8")
    assert hash_file(path) == hash_file(path)


def test_hash_file_differs_for_different_content(tmp_path: Path) -> None:
    """Conteúdos diferentes produzem hashes diferentes."""
    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    path_a.write_text("conteudo A", encoding="utf-8")
    path_b.write_text("conteudo B", encoding="utf-8")
    assert hash_file(path_a) != hash_file(path_b)


def test_hash_file_missing_raises(tmp_path: Path) -> None:
    """Arquivo inexistente levanta FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "ausente.bin")


def test_hash_file_supports_alternative_algorithm(tmp_path: Path) -> None:
    """Um algoritmo alternativo (ex.: md5) também é suportado."""
    path = tmp_path / "arquivo.txt"
    path.write_text("conteudo", encoding="utf-8")
    digest = hash_file(path, algorithm="md5")
    assert len(digest) == 32  # digest hexadecimal md5 tem 32 caracteres


def test_utils_seed_reexports_config_environment() -> None:
    """`src.utils.seed` reexporta a implementação canônica sem duplicá-la."""
    assert seed_everything is canonical_seed_everything
    assert RANDOM_SEED == 42


def test_timed_logs_elapsed_time(caplog: pytest.LogCaptureFixture) -> None:
    """O context manager loga a conclusão da etapa cronometrada."""
    with caplog.at_level(logging.INFO, logger="fraude.src.utils.timing"), timed("etapa de teste"):
        pass
    assert any("etapa de teste" in record.getMessage() for record in caplog.records)


def test_timed_logs_even_when_block_raises(caplog: pytest.LogCaptureFixture) -> None:
    """O tempo é logado mesmo quando o bloco protegido levanta uma exceção."""
    with (
        caplog.at_level(logging.INFO, logger="fraude.src.utils.timing"),
        pytest.raises(ValueError),
        timed("etapa com erro"),
    ):
        raise ValueError("falha proposital")
    assert any("etapa com erro" in record.getMessage() for record in caplog.records)
