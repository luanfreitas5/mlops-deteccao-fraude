"""Fixtures compartilhadas dos testes.

Gera DataFrames sintéticos pequenos que imitam o dataset Credit Card Fraud
(Time, V1-V28, Amount, Class) — nunca usar dados de produção nos testes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.config.settings import Settings
from src.constants.columns import PCA_COLUMNS


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Gerador aleatório determinístico para os testes."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_transactions(rng: np.random.Generator) -> pl.DataFrame:
    """DataFrame sintético desbalanceado com sinal aprendível na classe positiva.

    As fraudes recebem um deslocamento em V1-V4 para que os modelos consigam
    separar as classes, permitindo testes comportamentais e de métrica.
    """
    n = 400
    n_fraud = 40
    labels = np.array([0] * (n - n_fraud) + [1] * n_fraud)
    rng.shuffle(labels)

    data: dict[str, np.ndarray] = {}
    data["Time"] = rng.uniform(0, 172_800, size=n)
    for col in PCA_COLUMNS:
        base = rng.normal(0, 1, size=n)
        # Injeta sinal nas 4 primeiras componentes para as fraudes.
        if col in {"V1", "V2", "V3", "V4"}:
            base = base + labels * 3.0
        data[col] = base
    data["Amount"] = rng.exponential(scale=88.0, size=n)
    data["Class"] = labels

    return pl.DataFrame(data)


@pytest.fixture
def settings() -> Settings:
    """Configuração padrão do projeto para os testes."""
    return Settings()
