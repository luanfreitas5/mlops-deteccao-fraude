"""Testes de caminhos do projeto e carregamento/validação de configuração."""

from __future__ import annotations

from pathlib import Path

from src.config.paths import ProjectPaths, get_paths
from src.config.settings import Settings, _read_yaml, load_settings


def test_get_paths_returns_default_project_paths() -> None:
    """``get_paths`` retorna os caminhos canônicos do projeto."""
    paths = get_paths()
    assert paths.data_raw.name == "creditcard.csv"
    assert paths.configs.name == "configs"


def test_project_paths_ensure_dirs_creates_output_directories(tmp_path: Path) -> None:
    """``ensure_dirs`` cria os diretórios de saída, exceto os de dados brutos."""
    paths = ProjectPaths(
        root=tmp_path,
        configs=tmp_path / "configs",
        data_raw=tmp_path / "data" / "raw" / "creditcard.csv",
        data_interim=tmp_path / "data" / "interim",
        data_processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
        figures=tmp_path / "reports" / "figures",
        logs=tmp_path / "logs",
    )
    paths.ensure_dirs()
    assert paths.data_interim.is_dir()
    assert paths.data_processed.is_dir()
    assert paths.models.is_dir()
    assert paths.reports.is_dir()
    assert paths.figures.is_dir()
    assert paths.logs.is_dir()
    assert not paths.data_raw.parent.exists()  # dados brutos nunca são criados


def test_read_yaml_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    """Sem o arquivo YAML, retorna um dicionário vazio."""
    assert _read_yaml(tmp_path / "ausente.yaml") == {}


def test_read_yaml_reads_file_content(tmp_path: Path) -> None:
    """Lê e desserializa o conteúdo de um YAML existente."""
    path = tmp_path / "config.yaml"
    path.write_text("project_name: teste\nrandom_seed: 7\n", encoding="utf-8")
    content = _read_yaml(path)
    assert content == {"project_name": "teste", "random_seed": 7}


def test_load_settings_defaults_when_configs_dir_missing(tmp_path: Path) -> None:
    """Sem YAMLs de configuração, ``load_settings`` retorna os valores padrão."""
    settings = load_settings(tmp_path / "configs_inexistente")
    assert settings == Settings()


def test_load_settings_merges_config_and_model_params(tmp_path: Path) -> None:
    """``load_settings`` combina ``config.yaml`` e ``model_params.yaml``."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "config.yaml").write_text(
        "project_name: projeto-teste\nrandom_seed: 7\n", encoding="utf-8"
    )
    (configs_dir / "model_params.yaml").write_text(
        "logistic_regression:\n  C: 2.5\n", encoding="utf-8"
    )

    settings = load_settings(configs_dir)
    assert settings.project_name == "projeto-teste"
    assert settings.random_seed == 7
    assert settings.model.logistic_regression.C == 2.5
