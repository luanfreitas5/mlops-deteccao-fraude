"""Nomes de colunas do dataset Credit Card Fraud Detection (ULB).

O dataset contém 28 componentes principais anonimizados (V1-V28) obtidos por PCA,
além de ``Time`` (segundos desde a primeira transação), ``Amount`` (valor) e
``Class`` (alvo binário: 1 = fraude, 0 = legítima).
"""

from __future__ import annotations

TIME_COLUMN: str = "Time"
AMOUNT_COLUMN: str = "Amount"
TARGET_COLUMN: str = "Class"

# Componentes principais anonimizados por PCA.
PCA_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]

# Todas as features de entrada do dataset bruto (exclui o alvo).
FEATURE_COLUMNS: list[str] = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN]

# Todas as colunas esperadas no arquivo bruto, na ordem.
ALL_COLUMNS: list[str] = [*FEATURE_COLUMNS, TARGET_COLUMN]
