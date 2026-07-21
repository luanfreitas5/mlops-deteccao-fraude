"""Carrega e valida a configuração do projeto com Pydantic.

Os arquivos YAML de ``configs/`` são lidos e validados na inicialização, de modo
que uma configuração inválida falha imediatamente com um erro tipado e claro —
nunca no meio de uma execução. Segredos vêm do ``.env`` (nunca commitado).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.paths import get_paths


class SplitConfig(BaseModel):
    """Parâmetros de divisão treino/validação/teste."""

    strategy: Literal["stratified", "temporal"] = "stratified"
    test_size: float = Field(gt=0, lt=1, default=0.20)
    val_size: float = Field(gt=0, lt=1, default=0.20)
    target_column: str = "Class"
    time_column: str = "Time"


class ImbalanceConfig(BaseModel):
    """Estratégia de tratamento do desbalanceamento de classes."""

    strategy: Literal["scale_pos_weight", "smote", "none"] = "scale_pos_weight"
    smote_sampling_strategy: float = Field(gt=0, le=1, default=0.10)


class EvaluationConfig(BaseModel):
    """Configuração da avaliação e escolha do threshold de decisão."""

    primary_metric: str = "average_precision"
    target_recall: float = Field(gt=0, le=1, default=0.85)
    cv_folds: int = Field(gt=1, default=5)


class BusinessCostConfig(BaseModel):
    """Custos de negócio para otimização do threshold.

    Um falso negativo (fraude não detectada) costuma custar muito mais do que
    um falso positivo (transação legítima revisada).
    """

    false_negative: float = Field(gt=0, default=100.0)
    false_positive: float = Field(gt=0, default=1.0)


class LogisticRegressionParams(BaseModel):
    """Hiperparâmetros do baseline de Regressão Logística."""

    C: float = Field(gt=0, default=1.0)
    max_iter: int = Field(gt=0, default=1000)
    class_weight: str | None = "balanced"
    solver: str = "lbfgs"


class XGBoostParams(BaseModel):
    """Hiperparâmetros do modelo principal XGBoost."""

    n_estimators: int = Field(gt=0, default=400)
    max_depth: int = Field(gt=0, default=6)
    learning_rate: float = Field(gt=0, le=1, default=0.05)
    subsample: float = Field(gt=0, le=1, default=0.8)
    colsample_bytree: float = Field(gt=0, le=1, default=0.8)
    min_child_weight: float = Field(ge=0, default=1.0)
    reg_lambda: float = Field(ge=0, default=1.0)
    reg_alpha: float = Field(ge=0, default=0.0)
    tree_method: str = "hist"
    eval_metric: str = "aucpr"


class ModelParams(BaseModel):
    """Agrupa os hiperparâmetros de todos os modelos."""

    logistic_regression: LogisticRegressionParams = LogisticRegressionParams()
    xgboost: XGBoostParams = XGBoostParams()


class Settings(BaseSettings):
    """Configuração global do projeto, carregada de YAML + ``.env`` e validada.

    Parameters
    ----------
    project_name : str
        Nome do projeto.
    random_seed : int
        Semente global para reprodutibilidade.
    split : SplitConfig
        Configuração da divisão dos dados.
    imbalance : ImbalanceConfig
        Estratégia de desbalanceamento.
    evaluation : EvaluationConfig
        Configuração de avaliação.
    business_cost : BusinessCostConfig
        Custos de negócio para o threshold.
    model : ModelParams
        Hiperparâmetros dos modelos.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=(),
    )

    project_name: str = "mlops-deteccao-fraude"
    random_seed: int = 42
    split: SplitConfig = SplitConfig()
    imbalance: ImbalanceConfig = ImbalanceConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    business_cost: BusinessCostConfig = BusinessCostConfig()
    model: ModelParams = ModelParams()


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lê um arquivo YAML e retorna um dicionário.

    Parameters
    ----------
    path : Path
        Caminho do arquivo YAML.

    Returns
    -------
    dict[str, Any]
        Conteúdo do YAML; dicionário vazio se o arquivo não existir.

    Raises
    ------
    yaml.YAMLError
        Se o conteúdo do arquivo for um YAML inválido.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_settings(configs_dir: Path | None = None) -> Settings:
    """Carrega e valida a configuração combinando ``config.yaml`` e ``model_params.yaml``.

    Parameters
    ----------
    configs_dir : Path, optional
        Diretório dos YAML de configuração. Por padrão usa ``configs/``.

    Returns
    -------
    Settings
        Configuração validada do projeto.

    Raises
    ------
    pydantic.ValidationError
        Se algum valor violar as restrições declaradas.

    Examples
    --------
    >>> settings = load_settings()
    >>> settings.evaluation.primary_metric
    'average_precision'
    """
    configs_dir = configs_dir or get_paths().configs
    general = _read_yaml(configs_dir / "config.yaml")
    model_params = _read_yaml(configs_dir / "model_params.yaml")
    if model_params:
        general["model"] = model_params
    return Settings(**general)
