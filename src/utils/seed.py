"""Reexporta :func:`seed_everything` a partir de :mod:`src.config.environment`.

Mantido em ``utils`` por conveniência de importação; a implementação canônica
vive em :mod:`src.config.environment`.
"""

from __future__ import annotations

from src.config.environment import RANDOM_SEED, seed_everything

__all__ = ["RANDOM_SEED", "seed_everything"]
