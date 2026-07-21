"""Engenharia e seleção de features.

Módulos
-------
engineering
    Cria features derivadas (log do valor, hora do dia) a partir das colunas
    brutas, sem vazamento de informação futura.
"""

from src.features.engineering import (
    ENGINEERED_COLUMNS,
    engineer_features,
    get_model_features,
)

__all__ = ["ENGINEERED_COLUMNS", "engineer_features", "get_model_features"]
