"""Exceções relacionadas a modelos."""

from __future__ import annotations

from src.exceptions.base import FraudDetectionError


class ModelNotFittedError(FraudDetectionError):
    """Levantada ao tentar prever com um modelo ainda não treinado."""


class ModelNotFoundError(FraudDetectionError):
    """Levantada quando o artefato de modelo serializado não é encontrado."""
