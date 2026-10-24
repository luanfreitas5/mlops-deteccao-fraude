"""Pipeline de treino: dados gold -> modelo + threshold + métricas.

Fluxo:

1. Carrega treino/validação da camada gold.
2. Estima a PR-AUC por validação cruzada estratificada (média, desvio e IC 95%).
3. Treina o pipeline (pré-processador + desbalanceamento + modelo) em todo o treino.
4. Otimiza o threshold de decisão na validação (recall-alvo e custo de negócio).
5. Persiste o modelo, os metadados de linhagem e as métricas.
6. Registra no MLflow: run pai com tags de linhagem, um run filho por fold e o
   modelo final com assinatura.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from src.config.logging import get_logger
from src.config.paths import get_paths
from src.config.settings import Settings
from src.evaluation.evaluator import evaluate_model
from src.evaluation.threshold import (
    find_cost_optimal_threshold,
    find_threshold_for_recall,
)
from src.exceptions.pipeline import DirtyWorkingTreeError
from src.experiment.tracker import MLflowTracker
from src.models.factory import ModelName
from src.models.persistence import save_model
from src.training.cross_validation import CVResult, cross_validate_pr_auc
from src.training.trainer import build_training_pipeline, prepare_xy, train_model

logger = get_logger(__name__)


def run_training_pipeline(
    settings: Settings,
    model_name: ModelName | str = ModelName.XGBOOST,
    *,
    data_hash: str | None = None,
) -> dict[str, object]:
    """Treina o modelo, escolhe o threshold e persiste artefatos e métricas.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    model_name : ModelName or str, optional
        Modelo a treinar, by default ``ModelName.XGBOOST``.
    data_hash : str, optional
        Hash dos dados brutos, gravado nos metadados de linhagem.

    Returns
    -------
    dict[str, object]
        Métricas de validação, threshold escolhido e caminho do modelo.

    Raises
    ------
    FileNotFoundError
        Se os dados gold não existirem (rode o pré-processamento antes).
    DirtyWorkingTreeError
        Se houver alterações Git não commitadas e ``experiment.allow_dirty`` for falso.

    Examples
    --------
    >>> summary = run_training_pipeline(settings, "xgboost")  # doctest: +SKIP
    """
    paths = get_paths()
    train_path = paths.data_gold / "train.parquet"
    val_path = paths.data_gold / "val.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Dados gold não encontrados. Execute o pré-processamento antes."
        )

    # O tracker é criado antes do treino: árvore suja falha cedo, sem gastar o treino.
    tracker = _create_tracker(settings)

    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    x_train, y_train = prepare_xy(train_df)
    x_val, y_val = prepare_xy(val_df)
    assert y_train is not None and y_val is not None  # nosec B101

    pipeline = build_training_pipeline(model_name, settings, y_train)
    cv = cross_validate_pr_auc(
        pipeline,
        x_train,
        y_train,
        n_folds=settings.evaluation.cv_folds,
        seed=settings.random_seed,
    )
    pipeline = train_model(pipeline, x_train, y_train)

    y_val_prob = pipeline.predict_proba(x_val)[:, 1]

    # Escolha do threshold: por recall-alvo e por custo de negócio.
    threshold_recall = find_threshold_for_recall(
        y_val, y_val_prob, settings.evaluation.target_recall
    )
    threshold_cost, expected_cost = find_cost_optimal_threshold(
        y_val,
        y_val_prob,
        settings.business_cost.false_negative,
        settings.business_cost.false_positive,
    )
    # Operamos no threshold orientado a custo (defensável em fraude).
    best_threshold = threshold_cost

    result = evaluate_model(y_val, y_val_prob, threshold=best_threshold)
    logger.info(
        "Validação PR-AUC=%.4f (IC95 %.4f-%.4f) | recall=%.4f | threshold=%.3f",
        result.metrics["average_precision"],
        result.pr_auc_ci[0],
        result.pr_auc_ci[1],
        result.metrics["recall"],
        best_threshold,
    )

    metadata = {
        "model_name": str(ModelName(model_name).value),
        "data_hash": data_hash,
        "random_seed": settings.random_seed,
        "imbalance_strategy": settings.imbalance.strategy,
        "threshold_cost_optimal": threshold_cost,
        "threshold_target_recall": threshold_recall,
        "expected_cost": expected_cost,
        "best_threshold": best_threshold,
        "val_metrics": result.metrics,
        "pr_auc_ci_95": list(result.pr_auc_ci),
        "cv_pr_auc": {
            "n_folds": settings.evaluation.cv_folds,
            "mean": cv.mean,
            "std": cv.std,
            "ci_95_half_width": cv.ci_95,
            "scores": cv.scores,
        },
    }
    save_model(pipeline, paths.models / "fraud_model.joblib", metadata=metadata)

    metrics_path = paths.metrics / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    _log_to_mlflow(tracker, settings, model_name, pipeline, x_train, metadata, cv, data_hash)

    return {
        "metrics": result.metrics,
        "best_threshold": best_threshold,
        "model_path": str(paths.models / "fraud_model.joblib"),
    }


def _create_tracker(settings: Settings) -> MLflowTracker | None:
    """Cria o tracker MLflow; degrada para ``None`` se o MLflow estiver indisponível.

    Uma árvore Git suja **não** degrada: o erro é propagado, pois registrar um run
    com SHA que não descreve o código quebraria a linhagem.
    """
    try:
        return MLflowTracker(
            experiment_name=settings.experiment.experiment_name or settings.project_name,
            tracking_uri=settings.experiment.tracking_uri,
            seed=settings.random_seed,
            allow_dirty=settings.experiment.allow_dirty,
        )
    except DirtyWorkingTreeError:
        raise
    except Exception as exc:
        logger.warning("MLflow indisponível; o experimento não será registrado: %s", exc)
        return None


def _log_to_mlflow(
    tracker: MLflowTracker | None,
    settings: Settings,
    model_name: ModelName | str,
    pipeline: Any,
    x_train: Any,
    metadata: dict[str, Any],
    cv: CVResult,
    data_hash: str | None,
) -> None:
    """Registra a execução: params, CV (um run filho por fold), métricas e modelo."""
    if tracker is None:
        return
    try:
        with tracker.start_run(run_name=str(ModelName(model_name).value), data_hash=data_hash):
            tracker.log_params(
                {
                    **pipeline.named_steps["model"].get_params(),
                    "imbalance_strategy": settings.imbalance.strategy,
                    "cv": "StratifiedKFold",
                    "n_splits": settings.evaluation.cv_folds,
                }
            )
            tracker.log_cv_folds(cv.scores, "val_average_precision")
            tracker.log_metrics(
                {
                    "cv_average_precision_mean": cv.mean,
                    "cv_average_precision_std": cv.std,
                    **{f"val_{k}": float(v) for k, v in metadata["val_metrics"].items()},
                }
            )
            tracker.log_model(pipeline, x_train)
    except Exception as exc:
        logger.warning("Não foi possível registrar no MLflow: %s", exc)
