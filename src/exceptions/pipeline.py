"""Exceções relacionadas à orquestração de pipelines."""

from __future__ import annotations

from src.exceptions.base import FraudDetectionError


class PipelineError(FraudDetectionError):
    """Levantada quando uma etapa de pipeline falha de forma irrecuperável."""


class DirtyWorkingTreeError(PipelineError):
    """Levantada ao registrar um experimento com alterações Git não commitadas.

    O SHA de uma árvore suja não descreve o código que gerou o modelo; por isso o
    rastreamento é recusado até que as alterações sejam commitadas.
    """
