"""Implementações e gerenciamento de modelos.

Módulos
-------
base
    Define o Protocol ``Estimator`` (contrato estrutural fit/predict/predict_proba).
factory
    Cria estimadores (baseline e XGBoost) a partir dos hiperparâmetros validados.
persistence
    Salva e carrega modelos e metadados (joblib + JSON).
"""

from src.models.base import Estimator
from src.models.factory import ModelName, create_estimator
from src.models.persistence import load_model, save_model

__all__ = ["Estimator", "ModelName", "create_estimator", "load_model", "save_model"]
