"""Medição de tempo de execução para etapas do pipeline."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from src.config.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Context manager que mede e loga o tempo de um bloco de código.

    Parameters
    ----------
    label : str
        Descrição da etapa cronometrada (usada na mensagem de log).

    Yields
    ------
    None
        Executa o bloco protegido e loga a duração ao final.

    Examples
    --------
    >>> with timed("carregamento dos dados"):  # doctest: +SKIP
    ...     load_data()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Etapa '%s' concluída em %.2fs", label, elapsed)
