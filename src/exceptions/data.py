"""Exceções relacionadas a dados."""

from __future__ import annotations

from src.exceptions.base import FraudDetectionError


class RawDataNotFoundError(FraudDetectionError):
    """Levantada quando o arquivo de dados brutos não é encontrado."""


class DataValidationError(FraudDetectionError):
    """Levantada quando um DataFrame viola o contrato de dados (pandera)."""
