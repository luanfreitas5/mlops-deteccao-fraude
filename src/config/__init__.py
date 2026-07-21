"""Gerenciamento e validação de configuração do projeto.

Módulos
-------
paths
    Centraliza todos os caminhos do projeto com ``pathlib.Path``.
settings
    Carrega e valida os YAML de ``configs/`` e o ``.env`` com Pydantic.
logging
    Configura o logging com ``RichHandler`` e rotação diária de arquivo.
environment
    Fixa sementes e variáveis de ambiente para reprodutibilidade.
"""

from src.config.paths import ProjectPaths, get_paths
from src.config.settings import Settings, load_settings

__all__ = ["ProjectPaths", "Settings", "get_paths", "load_settings"]
