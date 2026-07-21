"""Pipeline de avaliação: modelo + teste -> métricas e figuras.

Avalia o modelo no conjunto de teste (nunca usado no treino/threshold), gera as
figuras de avaliação e persiste as métricas finais. É aqui que se afere se a
métrica principal atende ao mínimo acordado (regressão de métrica).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.config.logging import get_logger
from src.config.paths import get_paths
from src.evaluation.evaluator import evaluate_model
from src.inference.predictor import FraudPredictor
from src.training.trainer import prepare_xy
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)

logger = get_logger(__name__)


def run_evaluation_pipeline() -> dict[str, object]:
    """Avalia o modelo treinado no conjunto de teste e gera relatórios/figuras.

    Returns
    -------
    dict[str, object]
        Métricas de teste, IC 95% da PR-AUC e threshold utilizado.

    Raises
    ------
    FileNotFoundError
        Se o modelo ou os dados de teste não existirem.

    Examples
    --------
    >>> summary = run_evaluation_pipeline()  # doctest: +SKIP
    """
    paths = get_paths()
    model_path = paths.models / "fraud_model.joblib"
    metadata_path = paths.models / "fraud_model.json"
    test_path = paths.data_processed / "test.parquet"

    if not model_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Modelo ou dados de teste ausentes. Treine o modelo antes de avaliar."
        )

    threshold = 0.5
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            threshold = float(json.load(file).get("best_threshold", 0.5))

    predictor = FraudPredictor.from_path(model_path, threshold=threshold)
    test_df = pl.read_parquet(test_path)
    _, y_test = prepare_xy(test_df)
    assert y_test is not None  # nosec B101

    y_prob = predictor.predict_proba(test_df)
    y_pred = (y_prob >= threshold).astype(int)

    result = evaluate_model(y_test, y_prob, threshold=threshold)
    logger.info(
        "Teste PR-AUC=%.4f (IC95 %.4f-%.4f) | recall=%.4f | precision=%.4f",
        result.metrics["average_precision"],
        result.pr_auc_ci[0],
        result.pr_auc_ci[1],
        result.metrics["recall"],
        result.metrics["precision"],
    )

    figures_dir = paths.figures
    plot_precision_recall_curve(y_test, y_prob, figures_dir / "pr_curve")
    plot_roc_curve(y_test, y_prob, figures_dir / "roc_curve")
    plot_confusion_matrix(y_test, y_pred.astype(np.int_), figures_dir / "confusion_matrix")

    summary = {
        "test_metrics": result.metrics,
        "pr_auc_ci_95": list(result.pr_auc_ci),
        "threshold": threshold,
    }
    test_metrics_path = paths.reports / "test_metrics.json"
    with test_metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    logger.info("Métricas de teste salvas em %s", test_metrics_path)
    return summary
