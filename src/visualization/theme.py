"""Paleta de cores e estilo compartilhados para todas as figuras do projeto."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Paleta consistente reutilizada em todo o projeto.
COLORS: dict[str, str] = {
    "fraude": "#d62728",  # vermelho — classe positiva (fraude)
    "legitima": "#1f77b4",  # azul — classe negativa (legítima)
    "neutro": "#7f7f7f",
    "destaque": "#2ca02c",
}


def apply_theme() -> None:
    """Aplica o tema visual padrão (seaborn + matplotlib) do projeto.

    Configura estilo, contexto e DPI para figuras de qualidade de publicação.

    Examples
    --------
    >>> apply_theme()
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.autolayout"] = True
