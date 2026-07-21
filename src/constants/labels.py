"""Rótulos e mapeamentos da classificação binária de fraude."""

from __future__ import annotations

LEGITIMATE: int = 0
FRAUD: int = 1

# Nome legível de cada classe (para relatórios e figuras).
LABEL_NAMES: dict[int, str] = {
    LEGITIMATE: "legitima",
    FRAUD: "fraude",
}
