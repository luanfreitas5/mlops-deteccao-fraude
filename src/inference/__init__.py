"""Inferência e predição.

Módulos
-------
predictor
    Carrega o pipeline treinado e aplica as mesmas transformações do treino,
    garantindo consistência treino-servidor.
"""

from src.inference.predictor import FraudPredictor

__all__ = ["FraudPredictor"]
