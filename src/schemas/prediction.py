"""Schemas pydantic de requisição/resposta da API de inferência.

O ``TransactionRequest`` é construído dinamicamente a partir de
``FEATURE_COLUMNS`` para manter uma única fonte de verdade das features e evitar
divergência treino-servidor. Validar a requisição contra o mesmo conjunto de
colunas do treino é a defesa contra *train-serve skew*.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.constants.columns import FEATURE_COLUMNS

# Constrói dinamicamente um modelo pydantic com todas as features como float
# obrigatórios. Ex.: Time, V1, ..., V28, Amount. O tipo é anotado como Any
# porque o create_model do pydantic não é tipável com definições dinâmicas.
_field_definitions: dict[str, Any] = {
    name: (float, Field(..., description=f"Valor da feature {name}")) for name in FEATURE_COLUMNS
}

TransactionRequest = create_model(
    "TransactionRequest",
    __doc__=(
        "Requisição de inferência: uma transação com todas as features do "
        "dataset (Time, V1-V28, Amount)."
    ),
    **_field_definitions,  # pyright: ignore[reportArgumentType]
)


class PredictionResponse(BaseModel):
    """Resposta da API para uma predição de fraude.

    Attributes
    ----------
    fraud_probability : float
        Probabilidade estimada de a transação ser fraude (0 a 1).
    is_fraud : bool
        Decisão binária após aplicar o threshold de decisão.
    threshold : float
        Threshold de decisão utilizado.
    """

    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float = Field(ge=0, le=1)
    is_fraud: bool
    threshold: float = Field(ge=0, le=1)


def build_feature_frame(request: BaseModel) -> pl.DataFrame:
    """Converte uma requisição de transação em ``pl.DataFrame`` de uma linha.

    Parameters
    ----------
    request : BaseModel
        Instância de :data:`TransactionRequest` com as features da transação.

    Returns
    -------
    pl.DataFrame
        DataFrame de uma linha com as colunas na ordem de ``FEATURE_COLUMNS``.

    Examples
    --------
    >>> payload = {name: 0.0 for name in FEATURE_COLUMNS}
    >>> req = TransactionRequest(**payload)
    >>> frame = build_feature_frame(req)
    >>> frame.height
    1
    """
    data = request.model_dump()
    return pl.DataFrame({name: [data[name]] for name in FEATURE_COLUMNS})
