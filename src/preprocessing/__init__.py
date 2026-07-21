"""Limpeza e transformação de dados para modelagem.

Módulos
-------
cleaning
    Remove duplicatas e garante integridade básica.
pipeline
    Constrói o pré-processador scikit-learn (escalonamento das features numéricas).
"""

from src.preprocessing.cleaning import drop_duplicates
from src.preprocessing.pipeline import build_preprocessor

__all__ = ["build_preprocessor", "drop_duplicates"]
