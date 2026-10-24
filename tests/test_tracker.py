"""Testes do wrapper MLflowTracker (rastreamento de experimentos)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.exceptions.pipeline import DirtyWorkingTreeError
from src.experiment.tracker import MLflowTracker, compute_git_sha


@pytest.fixture
def tracker(tmp_path: Path) -> MLflowTracker:
    """Tracker MLflow com banco SQLite temporário (não polui o repositório).

    O backend de arquivos (``file:./mlruns``) está em modo de manutenção; o
    SQLite é o backend local usado pelo projeto. ``allow_dirty`` evita que a
    árvore Git de desenvolvimento (suja) impeça os testes.
    """
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    return MLflowTracker(
        experiment_name="teste-fraude", tracking_uri=tracking_uri, seed=7, allow_dirty=True
    )


def test_start_run_tags_lineage_on_parent(tracker: MLflowTracker) -> None:
    """O run pai recebe as tags git_sha, data_hash e seed."""
    with tracker.start_run(run_name="run-teste", data_hash="abc123") as run:
        run_id = run.info.run_id
    tags = mlflow.get_run(run_id).data.tags
    assert tags["data_hash"] == "abc123"
    assert tags["seed"] == "7"
    assert "git_sha" in tags


def test_start_run_without_data_hash_skips_tag(tracker: MLflowTracker) -> None:
    """Sem ``data_hash``, a tag correspondente não é registrada."""
    with tracker.start_run(run_name="run-sem-hash") as run:
        run_id = run.info.run_id
    assert "data_hash" not in mlflow.get_run(run_id).data.tags


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


def test_log_cv_folds_creates_nested_child_run_per_fold(tracker: MLflowTracker) -> None:
    """Cada fold vira um run filho com as tags de linhagem e a métrica do fold."""
    with tracker.start_run(run_name="cv", data_hash="h") as parent:
        tracker.log_cv_folds([0.7, 0.8, 0.9], "val_average_precision")
        parent_id = parent.info.run_id

    children = mlflow.search_runs(
        experiment_names=["teste-fraude"],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        output_format="list",
    )
    assert sorted(child.data.metrics["val_average_precision"] for child in children) == [
        0.7,
        0.8,
        0.9,
    ]
    assert all(child.data.tags["data_hash"] == "h" for child in children)
    assert sorted(child.data.tags["fold"] for child in children) == ["0", "1", "2"]


def test_log_model_passes_signature_and_input_example(
    tracker: MLflowTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``log_model`` registra o modelo com assinatura e exemplo de entrada.

    A serialização real do MLflow é lenta para um teste unitário; aqui garantimos
    que o wrapper repassa assinatura, exemplo e formato corretamente.
    """
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        mlflow.sklearn,
        "log_model",
        lambda model, **kwargs: calls.append({"model": model, **kwargs}),
    )
    x = pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.linspace(1, 0, 20)})
    model = LogisticRegression().fit(x, [0, 1] * 10)

    with tracker.start_run(run_name="run-modelo"):
        tracker.log_model(model, x, name="modelo")

    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["name"] == "modelo"
    assert calls[0]["signature"] is not None
    assert len(calls[0]["input_example"]) == 5  # type: ignore[arg-type]
    assert calls[0]["serialization_format"] == "cloudpickle"


def test_tracker_creates_sqlite_parent_directory(tmp_path: Path) -> None:
    """O diretório do arquivo SQLite é criado se não existir."""
    db_path = tmp_path / "novo" / "mlflow.db"
    MLflowTracker("exp", f"sqlite:///{db_path.as_posix()}", allow_dirty=True)
    assert db_path.parent.is_dir()


def _fake_git(status: str) -> object:
    """Cria um substituto de ``subprocess.run`` com ``git status`` fixo."""

    def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        out = status if cmd[1] == "status" else "a" * 40
        return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")

    return _run


def test_compute_git_sha_raises_on_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com alterações não commitadas, o registro é recusado."""
    monkeypatch.setattr("src.experiment.tracker.subprocess.run", _fake_git(" M src/main.py"))
    with pytest.raises(DirtyWorkingTreeError):
        compute_git_sha()


def test_compute_git_sha_returns_full_sha_on_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Árvore limpa retorna o SHA completo do HEAD."""
    monkeypatch.setattr("src.experiment.tracker.subprocess.run", _fake_git(""))
    assert compute_git_sha() == "a" * 40


def test_compute_git_sha_allow_dirty_returns_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    """``allow_dirty`` aceita árvore suja (apenas depuração local)."""
    monkeypatch.setattr("src.experiment.tracker.subprocess.run", _fake_git(" M x.py"))
    assert compute_git_sha(allow_dirty=True) == "a" * 40


def test_compute_git_sha_returns_unknown_when_git_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Se o comando git não estiver disponível, retorna ``'unknown'``."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("src.experiment.tracker.subprocess.run", _raise)
    assert compute_git_sha() == "unknown"
