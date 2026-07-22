"""Testes de integração dos pipelines: pré-processamento, treino e avaliação.

Os pipelines usam ``get_paths()`` internamente; aqui redirecionamos os caminhos
para um diretório temporário (via monkeypatch) para não tocar dados reais do
projeto. O MLflow real é substituído por um dublê rápido — a integração com o
MLflow é coberta separadamente em ``test_tracker.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.models.factory import ModelName
from src.models.persistence import save_model
from src.pipelines.evaluation import run_evaluation_pipeline
from src.pipelines.preprocessing import run_preprocessing_pipeline
from src.pipelines.training import run_training_pipeline
from src.training.trainer import build_training_pipeline, prepare_xy, train_model


class _FakeMLflowTracker:
    """Dublê rápido de MLflowTracker: evita I/O real de MLflow nos testes de pipeline."""

    def __init__(self, experiment_name: str, tracking_uri: str) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.logged_params: dict[str, Any] = {}
        self.logged_metrics: dict[str, float] = {}

    @contextmanager
    def start_run(self, run_name: str, data_hash: str | None = None) -> Iterator[None]:
        self.run_name = run_name
        self.data_hash = data_hash
        yield None

    def log_params(self, params: dict[str, Any]) -> None:
        self.logged_params = params

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.logged_metrics = metrics


@pytest.fixture
def project_paths(tmp_path: Path) -> ProjectPaths:
    """Caminhos canônicos do projeto redirecionados para um diretório temporário."""
    return ProjectPaths(
        root=tmp_path,
        configs=tmp_path / "configs",
        data_raw=tmp_path / "data" / "raw" / "creditcard.csv",
        data_interim=tmp_path / "data" / "interim",
        data_processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
        figures=tmp_path / "reports" / "figures",
        logs=tmp_path / "logs",
    )


def _write_raw_csv(paths: ProjectPaths, df: pl.DataFrame) -> None:
    paths.data_raw.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(paths.data_raw)


def test_run_preprocessing_pipeline_writes_partitions(
    project_paths: ProjectPaths,
    synthetic_transactions: pl.DataFrame,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O pré-processamento gera train/val/test em Parquet e retorna o hash dos dados."""
    _write_raw_csv(project_paths, synthetic_transactions)
    monkeypatch.setattr("src.pipelines.preprocessing.get_paths", lambda: project_paths)

    output = run_preprocessing_pipeline(settings)

    assert (project_paths.data_processed / "train.parquet").exists()
    assert (project_paths.data_processed / "val.parquet").exists()
    assert (project_paths.data_processed / "test.parquet").exists()
    assert len(output.data_hash) == 64  # digest sha256 hexadecimal
    total = output.split.train.height + output.split.val.height + output.split.test.height
    assert total == synthetic_transactions.height


def _run_preprocessing(
    project_paths: ProjectPaths,
    synthetic_transactions: pl.DataFrame,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Roda o pré-processamento e devolve o hash dos dados brutos."""
    _write_raw_csv(project_paths, synthetic_transactions)
    monkeypatch.setattr("src.pipelines.preprocessing.get_paths", lambda: project_paths)
    return run_preprocessing_pipeline(settings).data_hash


def test_run_training_pipeline_persists_model_and_metrics(
    project_paths: ProjectPaths,
    synthetic_transactions: pl.DataFrame,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O treino persiste o modelo, os metadados e as métricas, e registra no MLflow."""
    data_hash = _run_preprocessing(project_paths, synthetic_transactions, settings, monkeypatch)
    monkeypatch.setattr("src.pipelines.training.get_paths", lambda: project_paths)
    monkeypatch.setattr("src.pipelines.training.MLflowTracker", _FakeMLflowTracker)

    summary = run_training_pipeline(settings, ModelName.LOGISTIC_REGRESSION, data_hash=data_hash)

    model_path = project_paths.models / "fraud_model.joblib"
    assert model_path.exists()
    assert model_path.with_suffix(".json").exists()
    assert (project_paths.reports / "metrics.json").exists()
    assert "average_precision" in summary["metrics"]  # type: ignore[operator]
    assert 0.0 <= summary["best_threshold"] <= 1.0  # type: ignore[operator]


def test_run_training_pipeline_raises_when_processed_data_missing(
    project_paths: ProjectPaths, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem dados processados, levanta FileNotFoundError com mensagem clara."""
    monkeypatch.setattr("src.pipelines.training.get_paths", lambda: project_paths)
    with pytest.raises(FileNotFoundError):
        run_training_pipeline(settings, ModelName.LOGISTIC_REGRESSION)


def test_run_training_pipeline_degrades_gracefully_when_mlflow_unavailable(
    project_paths: ProjectPaths,
    synthetic_transactions: pl.DataFrame,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falha ao registrar no MLflow não interrompe o treino (degrada com um warning)."""
    data_hash = _run_preprocessing(project_paths, synthetic_transactions, settings, monkeypatch)
    monkeypatch.setattr("src.pipelines.training.get_paths", lambda: project_paths)

    def _raise_on_init(*args: object, **kwargs: object) -> None:
        raise RuntimeError("MLflow indisponível")

    monkeypatch.setattr("src.pipelines.training.MLflowTracker", _raise_on_init)

    summary = run_training_pipeline(settings, ModelName.LOGISTIC_REGRESSION, data_hash=data_hash)
    assert "average_precision" in summary["metrics"]  # type: ignore[operator]


def test_run_evaluation_pipeline_generates_report_and_figures(
    project_paths: ProjectPaths,
    synthetic_transactions: pl.DataFrame,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A avaliação gera métricas de teste, figuras e usa o threshold salvo no treino."""
    data_hash = _run_preprocessing(project_paths, synthetic_transactions, settings, monkeypatch)
    monkeypatch.setattr("src.pipelines.training.get_paths", lambda: project_paths)
    monkeypatch.setattr("src.pipelines.training.MLflowTracker", _FakeMLflowTracker)
    run_training_pipeline(settings, ModelName.LOGISTIC_REGRESSION, data_hash=data_hash)

    monkeypatch.setattr("src.pipelines.evaluation.get_paths", lambda: project_paths)
    summary = run_evaluation_pipeline()

    assert "average_precision" in summary["test_metrics"]  # type: ignore[operator]
    assert len(summary["pr_auc_ci_95"]) == 2  # type: ignore[arg-type]
    assert (project_paths.figures / "pr_curve.png").exists()
    assert (project_paths.figures / "roc_curve.png").exists()
    assert (project_paths.figures / "confusion_matrix.png").exists()
    assert (project_paths.reports / "test_metrics.json").exists()


def test_run_evaluation_pipeline_raises_when_model_missing(
    project_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem modelo treinado nem dados de teste, levanta FileNotFoundError."""
    monkeypatch.setattr("src.pipelines.evaluation.get_paths", lambda: project_paths)
    with pytest.raises(FileNotFoundError):
        run_evaluation_pipeline()


def test_run_evaluation_pipeline_defaults_threshold_when_metadata_missing(
    project_paths: ProjectPaths,
    synthetic_transactions: pl.DataFrame,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem o JSON de metadados, o threshold padrão (0.5) é utilizado."""
    _write_raw_csv(project_paths, synthetic_transactions)
    monkeypatch.setattr("src.pipelines.preprocessing.get_paths", lambda: project_paths)
    run_preprocessing_pipeline(settings)

    x, y = prepare_xy(synthetic_transactions)
    assert y is not None
    pipeline = build_training_pipeline(ModelName.LOGISTIC_REGRESSION, settings, y)
    pipeline = train_model(pipeline, x, y)
    save_model(pipeline, project_paths.models / "fraud_model.joblib")  # sem metadata

    monkeypatch.setattr("src.pipelines.evaluation.get_paths", lambda: project_paths)
    summary = run_evaluation_pipeline()
    assert summary["threshold"] == 0.5
