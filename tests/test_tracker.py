"""Testes do wrapper MLflowTracker (rastreamento de experimentos)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from sklearn.linear_model import LogisticRegression

from src.experiment.tracker import MLflowTracker, _current_git_sha


@pytest.fixture
def tracker(tmp_path: Path) -> MLflowTracker:
    """Tracker MLflow apontando para um banco SQLite temporário (não polui o repo).

    O backend de arquivos (``file:./mlruns``) está em modo de manutenção nas
    versões recentes do MLflow; SQLite é o backend local recomendado.
    """
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    return MLflowTracker(experiment_name="teste-fraude", tracking_uri=tracking_uri)


def test_start_run_tags_git_sha_and_data_hash(tracker: MLflowTracker) -> None:
    """O run iniciado recebe as tags git_sha e data_hash quando informado."""
    with tracker.start_run(run_name="run-teste", data_hash="abc123") as run:
        run_id = run.info.run_id
    finished = mlflow.get_run(run_id)
    assert finished.data.tags["data_hash"] == "abc123"
    assert "git_sha" in finished.data.tags


def test_start_run_without_data_hash_skips_tag(tracker: MLflowTracker) -> None:
    """Sem ``data_hash``, a tag correspondente não é registrada."""
    with tracker.start_run(run_name="run-sem-hash") as run:
        run_id = run.info.run_id
    finished = mlflow.get_run(run_id)
    assert "data_hash" not in finished.data.tags


def test_log_params_metrics_and_artifact(tracker: MLflowTracker, tmp_path: Path) -> None:
    """Params, métricas e artefato são registrados no run ativo."""
    artifact = tmp_path / "nota.txt"
    artifact.write_text("artefato de teste", encoding="utf-8")

    with tracker.start_run(run_name="run-completo") as run:
        run_id = run.info.run_id
        tracker.log_params({"n_estimators": 10})
        tracker.log_metrics({"average_precision": 0.9})
        tracker.log_artifact(artifact)

    finished = mlflow.get_run(run_id)
    assert finished.data.params["n_estimators"] == "10"
    assert finished.data.metrics["average_precision"] == 0.9


def test_log_model_delegates_to_mlflow_sklearn(
    tracker: MLflowTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``log_model`` delega ao ``log_model`` do sabor sklearn do MLflow.

    A inferência real de ambiente/dependências do MLflow para modelos sklearn é
    lenta demais para um teste unitário; aqui apenas garantimos que o wrapper
    repassa o modelo e o caminho do artefato corretamente.
    """
    calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        "src.experiment.tracker.log_sklearn_model",
        lambda model, artifact_path: calls.append((model, artifact_path)),
    )
    model = LogisticRegression().fit([[0], [1]], [0, 1])

    with tracker.start_run(run_name="run-modelo"):
        tracker.log_model(model, artifact_path="modelo")

    assert calls == [(model, "modelo")]


def test_current_git_sha_returns_short_sha_in_repo() -> None:
    """Dentro de um repositório Git, retorna um SHA curto não vazio."""
    sha = _current_git_sha()
    assert sha != "unknown"
    assert len(sha) > 0


def test_current_git_sha_returns_unknown_when_git_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Se o comando git não estiver disponível, retorna ``'unknown'``."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("src.experiment.tracker.subprocess.run", _raise)
    assert _current_git_sha() == "unknown"
