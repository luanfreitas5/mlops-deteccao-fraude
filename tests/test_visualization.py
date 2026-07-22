"""Testes do tema visual e dos gráficos de avaliação."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.visualization.theme import COLORS, apply_theme


def test_apply_theme_sets_expected_rcparams() -> None:
    """O tema aplica os parâmetros de DPI esperados para figuras de publicação."""
    apply_theme()
    assert plt.rcParams["figure.dpi"] == 120
    assert plt.rcParams["savefig.dpi"] == 300


def test_colors_palette_has_expected_keys() -> None:
    """A paleta compartilhada define as cores usadas nos gráficos do projeto."""
    assert {"fraude", "legitima", "neutro", "destaque"} <= set(COLORS)


def _synthetic_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Rótulos e probabilidades sintéticos com separação razoável entre classes."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 80 + [1] * 20)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.15, size=100), 0, 1)
    return y_true, y_prob


def test_plot_precision_recall_curve_saves_png_and_svg(tmp_path: Path) -> None:
    """A curva PR é salva em PNG e SVG no caminho informado."""
    y_true, y_prob = _synthetic_predictions()
    output_path = tmp_path / "pr_curve"
    result = plot_precision_recall_curve(y_true, y_prob, output_path)
    assert result == output_path
    assert output_path.with_suffix(".png").exists()
    assert output_path.with_suffix(".svg").exists()


def test_plot_roc_curve_saves_png_and_svg(tmp_path: Path) -> None:
    """A curva ROC é salva em PNG e SVG no caminho informado."""
    y_true, y_prob = _synthetic_predictions()
    output_path = tmp_path / "roc_curve"
    result = plot_roc_curve(y_true, y_prob, output_path)
    assert result == output_path
    assert output_path.with_suffix(".png").exists()
    assert output_path.with_suffix(".svg").exists()


def test_plot_confusion_matrix_saves_png_and_svg(tmp_path: Path) -> None:
    """A matriz de confusão é salva em PNG e SVG no caminho informado."""
    y_true, y_prob = _synthetic_predictions()
    y_pred = (y_prob >= 0.5).astype(int)
    output_path = tmp_path / "confusion_matrix"
    result = plot_confusion_matrix(y_true, y_pred, output_path)
    assert result == output_path
    assert output_path.with_suffix(".png").exists()
    assert output_path.with_suffix(".svg").exists()


def test_plot_creates_missing_parent_directory(tmp_path: Path) -> None:
    """O diretório de saída é criado automaticamente, se ainda não existir."""
    y_true, y_prob = _synthetic_predictions()
    output_path = tmp_path / "nested" / "dir" / "pr_curve"
    plot_precision_recall_curve(y_true, y_prob, output_path)
    assert output_path.with_suffix(".png").exists()
