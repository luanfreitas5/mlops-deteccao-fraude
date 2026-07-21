"""Exceções relacionadas à orquestração de pipelines."""

from __future__ import annotations

from src.exceptions.base import FraudDetectionError


class PipelineError(FraudDetectionError):
    """Levantada quando uma etapa de pipeline falha de forma irrecuperável."""
