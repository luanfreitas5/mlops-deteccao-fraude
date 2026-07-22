"""Testes da API FastAPI de inferência.

Usa um preditor treinado em dado sintético, injetado via override da dependência,
para testar os endpoints sem depender de um modelo persistido em disco.
"""

from __future__ import annotations

import polars as pl
import pytest
from app.dependencies import get_predictor
from app.main import app
from fastapi.testclient import TestClient

from src.config.settings import Settings
from src.constants.columns import FEATURE_COLUMNS
from src.inference.predictor import FraudPredictor
from src.models.factory import ModelName
from src.training.trainer import build_training_pipeline, prepare_xy, train_model


@pytest.fixture
def client(synthetic_transactions: pl.DataFrame, settings: Settings) -> TestClient:
    """Cliente de teste com o preditor injetado (override da dependência)."""
    x, y = prepare_xy(synthetic_transactions)
    assert y is not None
    pipeline = build_training_pipeline(ModelName.XGBOOST, settings, y)
    pipeline = train_model(pipeline, x, y)
    predictor = FraudPredictor(pipeline=pipeline, threshold=0.5)

    app.dependency_overrides[get_predictor] = lambda: predictor
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.smoke
def test_health_endpoint(client: TestClient) -> None:
    """O endpoint de saúde responde 200 e status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_valid_response(client: TestClient) -> None:
    """O endpoint de predição retorna probabilidade e decisão coerentes."""
    payload = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_predict_endpoint_rejects_missing_feature(client: TestClient) -> None:
    """Requisição sem uma feature obrigatória é rejeitada (422)."""
    payload = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    del payload["Amount"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
