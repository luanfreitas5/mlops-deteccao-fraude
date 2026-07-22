"""Aplicação FastAPI para inferência de fraude em tempo real.

Endpoints:

- ``GET /health``: verificação de saúde e disponibilidade do modelo.
- ``POST /predict``: classifica uma transação como fraude ou legítima.

A requisição é validada contra o mesmo conjunto de features do treino (via
``TransactionRequest``), protegendo contra *train-serve skew*.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI

from app.dependencies import get_predictor
from src.inference.predictor import FraudPredictor
from src.schemas.prediction import (
    PredictionResponse,
    TransactionRequest,
    build_feature_frame,
)

app = FastAPI(
    title="API de Detecção de Fraude",
    description="Classifica transações de cartão de crédito como fraude ou legítimas.",
    version="0.1.0",
)


@app.get("/health", tags=["infra"])
def health() -> dict[str, str]:
    """Verifica a saúde do serviço.

    Returns
    -------
    dict[str, str]
        Status do serviço.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["inferência"])
def predict(
    transaction: TransactionRequest,  # type: ignore[valid-type]
    predictor: Annotated[FraudPredictor, Depends(get_predictor)],
) -> PredictionResponse:
    """Classifica uma transação como fraude ou legítima.

    Parameters
    ----------
    transaction : TransactionRequest
        Features da transação (Time, V1-V28, Amount).
    predictor : FraudPredictor
        Preditor injetado, carregado no startup.

    Returns
    -------
    PredictionResponse
        Probabilidade de fraude, decisão binária e threshold aplicado.
    """
    frame = build_feature_frame(transaction)
    probability = float(predictor.predict_proba(frame)[0])
    is_fraud = probability >= predictor.threshold
    return PredictionResponse(
        fraud_probability=probability,
        is_fraud=is_fraud,
        threshold=predictor.threshold,
    )
