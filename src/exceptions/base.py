"""Exceção base do projeto de detecção de fraude."""

from __future__ import annotations


class FraudDetectionError(Exception):
    """Exceção base da qual todas as exceções do projeto herdam.

    Permite capturar qualquer erro específico do domínio com um único
    ``except FraudDetectionError``.
    """
