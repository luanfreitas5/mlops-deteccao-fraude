"""Exceções customizadas do projeto.

Módulos
-------
base
    Exceção base da qual todas as demais herdam.
data
    Erros de dados (arquivo ausente, schema inválido, colunas faltantes).
model
    Erros de modelo (não treinado, artefato ausente).
pipeline
    Erros de orquestração de pipeline.
"""

from src.exceptions.base import FraudDetectionError
from src.exceptions.data import DataValidationError, RawDataNotFoundError
from src.exceptions.model import ModelNotFittedError, ModelNotFoundError
from src.exceptions.pipeline import PipelineError

__all__ = [
    "DataValidationError",
    "FraudDetectionError",
    "ModelNotFittedError",
    "ModelNotFoundError",
    "PipelineError",
    "RawDataNotFoundError",
]
