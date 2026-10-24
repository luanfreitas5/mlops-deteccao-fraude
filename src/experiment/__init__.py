"""Rastreamento de experimentos e reprodutibilidade.

Módulos
-------
tracker
    Wrapper fino sobre o MLflow: run pai com tags de linhagem (``git_sha``,
    ``data_hash``, ``seed``), run filho por fold de validação cruzada e modelo
    registrado com assinatura. Recusa registrar com a árvore Git suja.
"""

from src.experiment.tracker import MLflowTracker, compute_git_sha

__all__ = ["MLflowTracker", "compute_git_sha"]
