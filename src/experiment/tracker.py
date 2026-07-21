"""Wrapper fino sobre o MLflow para rastreamento de experimentos.

Registra parâmetros, métricas e artefatos por execução, marcando cada run com o
hash do dataset e o SHA do Git — garantindo a linhagem (dados + código + params)
que produziu cada modelo. O import do MLflow é adiado para não penalizar quem só
usa as demais partes do projeto.
"""

from __future__ import annotations

import subprocess  # nosec B404 - uso restrito a `git rev-parse` para linhagem
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.sklearn import log_model as log_sklearn_model

from src.config.logging import get_logger

logger = get_logger(__name__)


def _current_git_sha() -> str:
    """Retorna o SHA curto do commit atual, ou ``"unknown"`` se indisponível."""
    try:
        result = subprocess.run(  # nosec B603 B607 - comando fixo, sem input externo
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class MLflowTracker:
    """Encapsula o ciclo de vida de um experimento MLflow.

    Parameters
    ----------
    experiment_name : str
        Nome do experimento no MLflow.
    tracking_uri : str
        URI de rastreamento (ex.: ``"file:./mlruns"`` ou um servidor remoto).
    """

    def __init__(self, experiment_name: str, tracking_uri: str) -> None:
        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(
            "MLflow configurado (experimento=%s, uri=%s)",
            experiment_name,
            tracking_uri,
        )

    @contextmanager
    def start_run(self, run_name: str, data_hash: str | None = None) -> Iterator[Any]:
        """Inicia um run do MLflow marcado com Git SHA e hash dos dados.

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
        with self._mlflow.start_run(run_name=run_name) as run:
            self._mlflow.set_tag("git_sha", _current_git_sha())
            if data_hash is not None:
                self._mlflow.set_tag("data_hash", data_hash)
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

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Registra um modelo scikit-learn/imblearn no run ativo."""
        log_sklearn_model(model, artifact_path)
