"""Orquestração principal do pipeline de detecção de fraude.

Expõe subcomandos via ``argparse``:

- ``preprocess``: carrega, limpa, divide e persiste os dados.
- ``train``: treina o modelo, otimiza o threshold e salva artefatos/métricas.
- ``evaluate``: avalia o modelo no teste e gera figuras/relatórios.
- ``all``: executa todo o fluxo de ponta a ponta.

Uso
---
>>> python -m src.main all --model xgboost  # doctest: +SKIP
"""

from __future__ import annotations

import argparse

from src.config.environment import seed_everything
from src.config.logging import configure_logging
from src.config.settings import Settings, load_settings
from src.models.factory import ModelName
from src.pipelines.evaluation import run_evaluation_pipeline
from src.pipelines.preprocessing import run_preprocessing_pipeline
from src.pipelines.training import run_training_pipeline


def _build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos da linha de comando.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado com os subcomandos do projeto.
    """
    parser = argparse.ArgumentParser(
        prog="mlops-deteccao-fraude",
        description="Pipeline MLOps de detecção de fraude em cartão de crédito.",
    )
    parser.add_argument(
        "stage",
        choices=["preprocess", "train", "evaluate", "all"],
        help="Etapa do pipeline a executar.",
    )
    parser.add_argument(
        "--model",
        choices=[m.value for m in ModelName],
        default=ModelName.XGBOOST.value,
        help="Modelo a treinar (padrão: xgboost).",
    )
    return parser


def run(stage: str, model_name: str, settings: Settings) -> None:
    """Executa a etapa solicitada do pipeline.

    Parameters
    ----------
    stage : str
        Etapa: ``preprocess``, ``train``, ``evaluate`` ou ``all``.
    model_name : str
        Nome do modelo a treinar.
    settings : Settings
        Configuração validada do projeto.
    """
    data_hash: str | None = None

    if stage in {"preprocess", "all"}:
        output = run_preprocessing_pipeline(settings)
        data_hash = output.data_hash

    if stage in {"train", "all"}:
        run_training_pipeline(settings, model_name, data_hash=data_hash)

    if stage in {"evaluate", "all"}:
        run_evaluation_pipeline()


def main() -> None:
    """Ponto de entrada da CLI: configura ambiente e despacha a etapa."""
    args = _build_parser().parse_args()
    logger = configure_logging()
    settings = load_settings()
    seed_everything(settings.random_seed)
    logger.info("Iniciando etapa '%s' (modelo=%s)", args.stage, args.model)
    run(args.stage, args.model, settings)
    logger.info("Etapa '%s' finalizada", args.stage)


if __name__ == "__main__":
    main()
