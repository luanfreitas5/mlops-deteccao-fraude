# Atalhos reprodutíveis. `make check` replica o CI localmente.
# Pipeline de dados em medalhão: bronze (data/bronze) -> silver -> gold -> modelo -> avaliação.
UV := uv
MODEL ?= xgboost

.DEFAULT_GOAL := help
.PHONY: help install hooks lint format typecheck security quality test smoke check \
	data preprocess train evaluate pipeline reproduce serve mlflow \
	docker-build docker-up docker-down docs docs-serve docs-deploy \
	pre-commit update-hooks update lock export bump profile clean clean-gold clean-reports

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Ambiente ---------------------------------------------------------------
install:  ## Instala todos os grupos de dependências e os hooks do pre-commit
	$(UV) sync --all-groups
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg

hooks:  ## Reinstala os hooks do pre-commit
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg
	$(UV) run detect-secrets scan > .secrets.baseline

update:  ## Atualiza o lock e sincroniza o ambiente
	$(UV) lock --upgrade
	$(UV) sync --all-groups

lock:  ## Gera/atualiza o uv.lock
	$(UV) lock

export:  ## Exporta requirements.txt a partir do lock (apenas runtime)
	$(UV) export --locked --no-dev --no-hashes -o requirements.txt

# --- Qualidade ---------------------------------------------------------------
lint:  ## Lint e verificação de formatação (ruff)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Corrige lint e formata o código (ruff)
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:  ## Type checking estático (basedPyright)
	$(UV) run basedpyright

security:  ## Segurança do código e das dependências (bandit + pip-audit)
	$(UV) run bandit -c pyproject.toml -r src
	$(UV) run pip-audit

quality:  ## Código morto, modernização, complexidade e docstrings
	$(UV) run vulture
	$(UV) run refurb src
	$(UV) run xenon --max-absolute B --max-modules A --max-average A src
	$(UV) run interrogate -v src

test:  ## Testes com cobertura (mínimo 80%)
	$(UV) run pytest -m "not slow" --cov --cov-report=term-missing --cov-report=xml

smoke:  ## Smoke tests (os mesmos do pre-commit)
	$(UV) run pytest -m smoke -q --no-cov

check: lint typecheck security quality test  ## Replica o CI completo localmente

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	$(UV) run pre-commit run --all-files

update-hooks:  ## Atualiza as versões dos hooks do pre-commit
	$(UV) run pre-commit autoupdate

bump:  ## Incrementa a versão (SemVer) e atualiza o CHANGELOG (commitizen)
	$(UV) run cz bump --changelog --yes

# --- Pipeline de ML ----------------------------------------------------------
data:  ## Bronze -> silver -> gold (valida, deduplica e divide)
	$(UV) run python -m src.main preprocess

preprocess: data  ## Alias de `data`

train:  ## Treina (CV + modelo final), escolhe o threshold e registra no MLflow
	$(UV) run python -m src.main train --model $(MODEL)

evaluate:  ## Avalia no teste e gera figuras e reports/metrics/test_metrics.json
	$(UV) run python -m src.main evaluate

pipeline:  ## Fluxo completo: data -> train -> evaluate
	$(UV) run python -m src.main all --model $(MODEL)

reproduce:  ## Reexecuta apenas os estágios afetados (requer DVC instalado)
	dvc repro

# --- Serving e MLflow --------------------------------------------------------
serve:  ## Sobe a API de inferência (FastAPI + uvicorn) em http://localhost:8000/docs
	$(UV) run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

mlflow:  ## UI do MLflow com backend SQLite em http://localhost:5000
	$(UV) run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# --- Docker ------------------------------------------------------------------
docker-build:  ## Constrói a imagem da API (sem o modelo: ele é montado em runtime)
	docker build -t mlops-deteccao-fraude:dev .

docker-up:  ## Sobe API + MLflow + Postgres (requer .env com POSTGRES_USER/PASSWORD)
	docker compose up -d --build

docker-down:  ## Derruba os serviços
	docker compose down

# --- Documentação ------------------------------------------------------------
docs:  ## Constrói a documentação em modo estrito
	$(UV) run --group docs mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	$(UV) run --group docs mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	$(UV) run --group docs mkdocs gh-deploy --force

# --- Utilitários -------------------------------------------------------------
profile:  ## Profiling com scalene (meça antes de otimizar)
	$(UV) run --group profiling scalene src/main.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis htmlcov coverage.xml .coverage site
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

clean-gold:  ## Remove as camadas silver e gold (o bronze é imutável e nunca é apagado)
	rm -f data/silver/*.parquet data/gold/*.parquet

clean-reports:  ## Remove figuras e métricas geradas
	rm -f reports/figures/*.png reports/figures/*.svg reports/metrics/*.json
