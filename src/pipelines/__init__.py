"""Pipelines de ponta a ponta.

Módulos
-------
preprocessing
    Carrega, limpa, divide e persiste os dados.
training
    Treina o modelo, otimiza o threshold e persiste artefatos + métricas.
evaluation
    Avalia o modelo no conjunto de teste e gera figuras/relatórios.
"""

from src.pipelines.evaluation import run_evaluation_pipeline
from src.pipelines.preprocessing import run_preprocessing_pipeline
from src.pipelines.training import run_training_pipeline

__all__ = [
    "run_evaluation_pipeline",
    "run_preprocessing_pipeline",
    "run_training_pipeline",
]
