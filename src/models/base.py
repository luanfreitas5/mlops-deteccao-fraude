"""Interface estrutural (Protocol) dos estimadores usados no projeto.

Define o contrato mínimo — ``fit``, ``predict`` e ``predict_proba`` — que
qualquer pipeline scikit-learn/imblearn treinado satisfaz. Tipar por esse
Protocol (em vez de ``object``) dá segurança de tipo ao acessar os métodos do
estimador sem acoplar o código a uma classe concreta.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Estimator(Protocol):
    """Contrato estrutural de um estimador scikit-learn/imblearn.

    Notes
    -----
    Protocol estrutural: qualquer objeto que exponha ``fit``, ``predict`` e
    ``predict_proba`` é compatível, sem necessidade de herança explícita.
    """

    def fit(self, x: Any, y: Any = ...) -> Any:
        """Ajusta o estimador aos dados de treino."""
        ...

    def predict(self, x: Any) -> NDArray[Any]:
        """Retorna a predição de classe para cada amostra."""
        ...

    def predict_proba(self, x: Any) -> NDArray[np.float64]:
        """Retorna a probabilidade de cada classe para cada amostra."""
        ...
