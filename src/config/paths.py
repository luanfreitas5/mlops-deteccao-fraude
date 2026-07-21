"""Centraliza os caminhos do projeto usando ``pathlib.Path``.

Todos os caminhos derivam da raiz do repositório, resolvida a partir da
localização deste arquivo. Nunca use strings de caminho hardcoded no código;
importe :func:`get_paths` ou :class:`ProjectPaths`.
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
    data_raw : Path
        Caminho do CSV bruto de transações (nunca modificado).
    data_processed : Path
        Diretório com os dados prontos para modelagem.
    models : Path
        Diretório de modelos e artefatos serializados.
    reports : Path
        Diretório de relatórios, figuras e métricas.
    logs : Path
        Diretório de logs com rotação diária.
    """

    root: Path = ROOT
    configs: Path = ROOT / "configs"
    data_raw: Path = ROOT / "data" / "raw" / "creditcard.csv"
    data_interim: Path = ROOT / "data" / "interim"
    data_processed: Path = ROOT / "data" / "processed"
    models: Path = ROOT / "models"
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"
    logs: Path = ROOT / "logs"

    def ensure_dirs(self) -> None:
        """Cria os diretórios de saída caso ainda não existam.

        Diretórios de dados brutos não são criados — devem preexistir.
        """
        for directory in (
            self.data_interim,
            self.data_processed,
            self.models,
            self.reports,
            self.figures,
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
    >>> paths.data_raw.name
    'creditcard.csv'
    """
    return ProjectPaths()
