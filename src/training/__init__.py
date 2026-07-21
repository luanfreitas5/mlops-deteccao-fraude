"""Treino de modelos: montagem do pipeline, ajuste e validação cruzada.

Módulos
-------
trainer
    Monta o pipeline (pré-processador + desbalanceamento + modelo) e o treina.
cross_validation
    Validação cruzada estratificada com PR-AUC e intervalo de confiança.
"""

from src.training.cross_validation import cross_validate_pr_auc
from src.training.trainer import build_training_pipeline, prepare_xy, train_model

__all__ = [
    "build_training_pipeline",
    "cross_validate_pr_auc",
    "prepare_xy",
    "train_model",
]
