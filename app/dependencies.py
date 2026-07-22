"""Carregamento e injeção do preditor treinado na API.

Lê os caminhos e o threshold de ``configs/deploy.yaml`` e mantém uma única
instância do preditor em memória (carregada no startup da aplicação).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.config.paths import get_paths
from src.inference.predictor import FraudPredictor


def _load_deploy_config() -> dict[str, Any]:
    """Lê ``configs/deploy.yaml`` e retorna a seção ``api``."""
    path = get_paths().configs / "deploy.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return (yaml.safe_load(file) or {}).get("api", {})


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    """Retorna o preditor de fraude, carregado uma única vez (cache).

    Returns
    -------
    FraudPredictor
        Preditor pronto para inferência, com o threshold de produção aplicado.

    Raises
    ------
    ModelNotFoundError
        Se o artefato de modelo não for encontrado.
    """
    config = _load_deploy_config()
    root = get_paths().root
    model_path = root / config.get("model_path", "models/fraud_model.joblib")
    threshold = float(config.get("decision_threshold", 0.5))
    return FraudPredictor.from_path(Path(model_path), threshold=threshold)
