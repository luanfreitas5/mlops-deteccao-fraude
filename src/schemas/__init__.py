"""Contratos de dados (pandera) e schemas de I/O da API (pydantic).

Módulos
-------
dataset
    Contrato pandera para as transações (valida tipos, faixas e nulos).
prediction
    Schemas pydantic de requisição/resposta da API de inferência.
"""

from src.schemas.dataset import TransactionsSchema, validate_transactions
from src.schemas.prediction import (
    PredictionResponse,
    TransactionRequest,
    build_feature_frame,
)

__all__ = [
    "PredictionResponse",
    "TransactionRequest",
    "TransactionsSchema",
    "build_feature_frame",
    "validate_transactions",
]
