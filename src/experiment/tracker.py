"""Wrapper fino sobre o MLflow para rastreamento de experimentos.

Cada run é um registro reprodutível: tags de linhagem (``git_sha``,
``data_hash``, ``seed``) no run pai **e** nos filhos, parâmetros, métricas por
fold (um run filho por fold) e um modelo com assinatura e exemplo de entrada.
O registro é recusado quando a árvore Git tem alterações não commitadas, pois o
SHA de uma árvore suja não descreve o código que gerou o modelo.
"""

from __future__ import annotations

import subprocess  # nosec B404 - uso restrito a comandos fixos do git para linhagem
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.config.logging import get_logger
from src.exceptions.pipeline import DirtyWorkingTreeError

logger = get_logger(__name__)

_SQLITE_PREFIX = "sqlite:///"


def compute_git_sha(*, allow_dirty: bool = False) -> str:
    """Retorna o SHA completo do ``HEAD`` para registrar a linhagem do código.

    Parameters
    ----------
    allow_dirty : bool, optional
        Se ``True``, aceita árvore suja (só para depuração local), by default False.

    Returns
    -------
    str
        SHA completo do commit atual, ou ``"unknown"`` se o Git estiver indisponível.

    Raises
    ------
    DirtyWorkingTreeError
        Se houver alterações não commitadas e ``allow_dirty`` for ``False``.

    Examples
    --------
    >>> compute_git_sha(allow_dirty=True)  # doctest: +SKIP
    """
    try:
        status = subprocess.run(  # nosec B603 B607 - comando fixo, sem input externo
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        sha = subprocess.run(  # nosec B603 B607 - comando fixo, sem input externo
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    if status and not allow_dirty:
        raise DirtyWorkingTreeError(
            "Árvore Git suja: faça commit das alterações antes de registrar o experimento."
        )
    return sha


class MLflowTracker:
    """Encapsula o ciclo de vida de um experimento MLflow.

    Parameters
    ----------
    experiment_name : str
        Nome do experimento no MLflow.
    tracking_uri : str
        URI de rastreamento com backend de banco de dados
        (ex.: ``"sqlite:///mlruns/mlflow.db"``) ou servidor remoto.
    seed : int, optional
        Semente global, registrada como tag de linhagem, by default 42.
    allow_dirty : bool, optional
        Aceita árvore Git suja (só depuração local), by default False.

    Raises
    ------
    DirtyWorkingTreeError
        Se a árvore Git tiver alterações não commitadas e ``allow_dirty`` for False.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        *,
        seed: int = 42,
        allow_dirty: bool = False,
    ) -> None:
        self._mlflow = mlflow
        self._seed = seed
        self._git_sha = compute_git_sha(allow_dirty=allow_dirty)
        self._lineage_tags: dict[str, str] = {}
        if tracking_uri.startswith(_SQLITE_PREFIX):  # o SQLite não cria o diretório do arquivo
            Path(tracking_uri.removeprefix(_SQLITE_PREFIX)).parent.mkdir(
                parents=True, exist_ok=True
            )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(
            "MLflow configurado (experimento=%s, uri=%s)",
            experiment_name,
            tracking_uri,
        )

    @contextmanager
    def start_run(self, run_name: str, data_hash: str | None = None) -> Iterator[Any]:
        """Inicia o run pai marcado com ``git_sha``, ``data_hash`` e ``seed``.

        Parameters
        ----------
        run_name : str
            Nome do run.
        data_hash : str, optional
            Hash SHA-256 do dataset usado, registrado como tag de linhagem.

        Yields
        ------
        Any
            O objeto de run ativo do MLflow.
        """
        tags = {"git_sha": self._git_sha, "seed": str(self._seed)}
        if data_hash is not None:
            tags["data_hash"] = data_hash
        self._lineage_tags = tags
        with self._mlflow.start_run(run_name=run_name, tags=tags) as run:
            yield run

    def log_params(self, params: dict[str, Any]) -> None:
        """Registra um dicionário de parâmetros no run ativo."""
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Registra um dicionário de métricas no run ativo."""
        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        """Registra um artefato (arquivo) no run ativo."""
        self._mlflow.log_artifact(str(path))

    def log_cv_folds(self, scores: Sequence[float], metric_name: str) -> None:
        """Registra um run filho (aninhado) por fold com a métrica de validação.

        Parameters
        ----------
        scores : Sequence[float]
            Métrica de cada fold, na ordem dos folds.
        metric_name : str
            Nome da métrica registrada em cada run filho.

        Examples
        --------
        >>> tracker.log_cv_folds([0.81, 0.84], "val_average_precision")  # doctest: +SKIP
        """
        for fold, score in enumerate(scores):
            with self._mlflow.start_run(
                run_name=f"fold{fold}",
                nested=True,
                tags={**self._lineage_tags, "fold": str(fold)},
            ):
                self._mlflow.log_metric(metric_name, float(score))

    def log_model(self, model: Any, x_example: pd.DataFrame, name: str = "model") -> None:
        """Registra o modelo (já ajustado em todo o treino) com assinatura e exemplo.

        Parameters
        ----------
        model : Any
            Pipeline scikit-learn/imblearn ajustado, com ``predict_proba``.
        x_example : pandas.DataFrame
            Amostra de features usada para inferir a assinatura de entrada/saída.
        name : str, optional
            Nome do artefato do modelo, by default ``"model"``.
        """
        sample = x_example.head(100)
        signature = infer_signature(sample, model.predict_proba(sample))
        # cloudpickle: modelo próprio e confiável; o formato skops recusa pipelines com XGBoost.
        mlflow.sklearn.log_model(
            model,
            name=name,
            signature=signature,
            input_example=x_example.head(5),
            serialization_format="cloudpickle",
        )
