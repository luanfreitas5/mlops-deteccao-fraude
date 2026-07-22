"""Testes de reprodutibilidade: fixação de sementes aleatórias."""

from __future__ import annotations

import os
import random

from src.config.environment import RANDOM_SEED, seed_everything


def test_seed_everything_sets_pythonhashseed() -> None:
    """PYTHONHASHSEED reflete a semente informada."""
    seed_everything(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_seed_everything_makes_random_deterministic() -> None:
    """Duas chamadas com a mesma semente reproduzem a mesma sequência do módulo random."""
    seed_everything(7)
    first = random.random()
    seed_everything(7)
    second = random.random()
    assert first == second


def test_seed_everything_default_uses_random_seed_constant() -> None:
    """Sem argumento, usa a constante RANDOM_SEED do módulo."""
    seed_everything()
    assert os.environ["PYTHONHASHSEED"] == str(RANDOM_SEED)
