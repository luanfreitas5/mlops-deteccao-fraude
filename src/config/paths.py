"""Centraliza os caminhos do projeto usando ``pathlib.Path``.

Todos os caminhos derivam da raiz do repositório, resolvida a partir da
localização deste arquivo. Nunca use strings de caminho hardcoded no código;
importe :func:`get_paths` ou :class:`ProjectPaths`.

Os dados seguem a arquitetura medalhão: ``bronze`` (como recebido, imutável),
``silver`` (limpo, tipado e validado) e ``gold`` (pronto para modelagem).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# src/config/paths.py -> parents[2] == raiz do projeto
ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Coleção imutável de caminhos canônicos do projeto.

    Attributes
    ----------
    root : Path
        Raiz do repositório.
    configs : Path
        Diretório de arquivos YAML de configuração.
    data_bronze : Path
        Caminho do CSV bruto de transações, exatamente como recebido (imutável).
    data_silver : Path
        Diretório dos dados limpos, tipados, deduplicados e validados.
    data_gold : Path
        Diretório dos dados prontos para modelagem (partições treino/validação/teste).
    models : Path
        Diretório de modelos e artefatos serializados.
    reports : Path
        Diretório de relatórios, figuras e métricas.
    figures : Path
        Diretório das figuras de avaliação.
    metrics : Path
        Diretório dos JSONs de métricas (versionados no Git).
    logs : Path
        Diretório de logs com rotação diária.
    """

    root: Path = ROOT
    configs: Path = ROOT / "configs"
    data_bronze: Path = ROOT / "data" / "bronze" / "creditcard.csv"
    data_silver: Path = ROOT / "data" / "silver"
    data_gold: Path = ROOT / "data" / "gold"
    models: Path = ROOT / "models"
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"
    metrics: Path = ROOT / "reports" / "metrics"
    logs: Path = ROOT / "logs"

    def ensure_dirs(self) -> None:
        """Cria os diretórios de saída caso ainda não existam.

        A camada bronze não é criada — o dado bruto deve preexistir.
        """
        for directory in (
            self.data_silver,
            self.data_gold,
            self.models,
            self.reports,
            self.figures,
            self.metrics,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_paths() -> ProjectPaths:
    """Retorna a coleção padrão de caminhos do projeto.

    Returns
    -------
    ProjectPaths
        Instância imutável com todos os caminhos canônicos.

    Examples
    --------
    >>> paths = get_paths()
    >>> paths.data_bronze.name
    'creditcard.csv'
    """
    return ProjectPaths()
