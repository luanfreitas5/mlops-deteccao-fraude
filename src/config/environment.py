"""Reprodutibilidade: fixa todas as fontes de aleatoriedade do projeto.

``random_state`` sozinho não garante reprodutibilidade. Este módulo fixa
``random``, ``numpy`` e ``PYTHONHASHSEED`` de forma centralizada. O hash do
dataset é feito em :mod:`src.utils.hashing`.
"""

from __future__ import annotations

import os
import random

import numpy as np

RANDOM_SEED: int = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Fixa todas as fontes de aleatoriedade para garantir reprodutibilidade.

    Parameters
    ----------
    seed : int, optional
        Semente a ser aplicada, by default ``RANDOM_SEED`` (42).

    Notes
    -----
    Cobre ``PYTHONHASHSEED``, ``random`` e ``numpy``. Bibliotecas de modelagem
    (scikit-learn, XGBoost) recebem a semente explicitamente via ``random_state``
    nos respectivos construtores.

    Examples
    --------
    >>> seed_everything(123)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed=RANDOM_SEED)
    rng.normal()
