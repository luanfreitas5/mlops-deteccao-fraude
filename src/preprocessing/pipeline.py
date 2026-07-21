"""Construção do pré-processador scikit-learn.

As colunas V1-V28 já estão em escala comparável (saída de PCA), portanto passam
direto. As features de valor/tempo (``Amount``, ``Amount_log``, ``Hour``) recebem
``RobustScaler`` — robusto a outliers, abundantes em valores de transação. Um
único pré-processador é ajustado no treino e reaplicado no serving, garantindo
consistência treino-servidor.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src.constants.columns import AMOUNT_COLUMN, PCA_COLUMNS
from src.features.engineering import ENGINEERED_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Constrói o ``ColumnTransformer`` de pré-processamento das features.

    Returns
    -------
    ColumnTransformer
        Transformer que escala ``Amount``/derivadas com ``RobustScaler`` e mantém
        as componentes de PCA inalteradas. Preserva a ordem das colunas de saída.

    Examples
    --------
    >>> pre = build_preprocessor()
    >>> pre.__class__.__name__
    'ColumnTransformer'
    """
    scaled_columns = [AMOUNT_COLUMN, *ENGINEERED_COLUMNS]
    return ColumnTransformer(
        transformers=[
            ("scale", RobustScaler(), scaled_columns),
            ("passthrough", "passthrough", PCA_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
