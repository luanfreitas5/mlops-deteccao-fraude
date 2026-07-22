# 🛡️ MLOps — Detecção de Fraude em Cartão de Crédito

[![CI](https://github.com/luanfreitas5/mlops-deteccao-fraude/actions/workflows/ci.yml/badge.svg)](https://github.com/luanfreitas5/mlops-deteccao-fraude/actions/workflows/ci.yml)
[![Tests](https://github.com/luanfreitas5/mlops-deteccao-fraude/actions/workflows/tests.yml/badge.svg)](https://github.com/luanfreitas5/mlops-deteccao-fraude/actions/workflows/tests.yml)
[![Docs](https://github.com/luanfreitas5/mlops-deteccao-fraude/actions/workflows/docs.yml/badge.svg)](https://github.com/luanfreitas5/mlops-deteccao-fraude/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Classificação em dados **fortemente desbalanceados** empacotada como **sistema de
> produção**: API de inferência, versionamento de dados/modelos, rastreamento de
> experimentos e CI/CD. É o projeto que prova **engenharia**, não só notebook.

---

## 🎯 Problema

Detectar transações fraudulentas em ~284 mil transações de cartão de crédito
anonimizadas, com **~0,17% de fraudes**. O desbalanceamento extremo é o coração
do desafio — e a razão de a **métrica principal ser a PR-AUC** (*Average
Precision*), não a acurácia nem a ROC-AUC.

| | |
|---|---|
| **Domínio** | Finanças / detecção de fraude |
| **Alvo** | `Class` — binário (1 = fraude, 0 = legítima) |
| **Métrica principal** | **PR-AUC** (foca na classe positiva sob desbalanceamento extremo) |
| **Complementares** | ROC-AUC, recall/precision, Brier (calibração) |
| **Modelo** | XGBoost (`scale_pos_weight`); baseline Regressão Logística; SMOTE opcional |
| **Threshold** | Otimizado por **custo de negócio** (FN ≫ FP), não fixo em 0,5 |
| **Deploy** | API **FastAPI** com validação de schema (anti train–serve skew) |
| **Dados** | [Credit Card Fraud (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## 🧱 Stack

`Python` · `Polars` · `scikit-learn` · `XGBoost` · `imbalanced-learn` ·
`pandera` · `Pydantic` · `FastAPI` · `MLflow` · `DVC` · `Docker` ·
`GitHub Actions` · `pytest` · `MkDocs`

---

## 📂 Estrutura

```text
├── app/               # API FastAPI (inferência)
├── configs/           # YAML validados por Pydantic
├── data/              # raw / interim / processed (versionados por DVC)
├── docs/              # MkDocs Material
├── notebooks/         # EDA (exploração)
├── reports/           # figuras, model cards, datasheets, métricas
├── src/
│   ├── config/        # settings, paths, logging, seeds
│   ├── constants/     # colunas, rótulos, métricas
│   ├── schemas/       # contrato pandera + schemas da API
│   ├── data/          # loader, splitter, writer
│   ├── preprocessing/ # limpeza + pré-processador sklearn
│   ├── features/      # engenharia de features
│   ├── models/        # fábrica + persistência
│   ├── training/      # trainer + validação cruzada
│   ├── evaluation/    # métricas, threshold, avaliador (IC bootstrap)
│   ├── inference/     # preditor (consistente com o treino)
│   ├── experiment/    # tracker MLflow
│   ├── pipelines/     # orquestração ponta a ponta
│   ├── visualization/ # tema + figuras
│   └── main.py        # CLI (preprocess | train | evaluate | all)
└── tests/             # unit, property-based, comportamentais, API
```

---

## 🚀 Início rápido

```bash
# 1. Dependências (uv)
make install
uv sync --dev --extra api

# 2. Coloque o dataset em data/raw/creditcard.csv

# 3. Pipeline completo: raw -> processed -> modelo -> avaliação
make pipeline

# 4. Suba a API de inferência
make serve      # http://localhost:8000/docs
```

### Comandos úteis

| Comando | Descrição |
|---|---|
| `make preprocess` | Carrega, valida, limpa, divide e persiste os dados |
| `make train` | Treina, otimiza o threshold e salva modelo + métricas |
| `make evaluate` | Avalia no teste e gera figuras/relatórios |
| `make quality` | Espelha o CI (lint, tipos, segurança, complexidade, docstrings) |
| `make test` | Testes com cobertura (≥ 80%) |
| `make mlflow` | UI do MLflow (`http://localhost:5000`) |
| `make docker-up` | Sobe API + MLflow via docker-compose |

---

## 🧪 Decisões de engenharia (senior bar)

- **PR-AUC como métrica principal:** com 0,17% de positivos, a ROC-AUC é otimista
  demais; a PR-AUC reflete o desempenho sobre a fraude.
- **Incerteza sempre:** toda métrica principal vem com **IC 95%** (bootstrap), nunca
  um ponto isolado.
- **Threshold por custo de negócio:** o corte é escolhido minimizando o custo
  esperado (FN ≫ FP), não fixado em 0,5.
- **Validação de dados (pandera):** contrato aplicado nas fronteiras do pipeline e
  em cada requisição da API — falha cedo, não corrompe em silêncio.
- **Consistência treino–serving:** a mesma `prepare_xy` gera features no treino e no
  serving; SMOTE só no ajuste (dentro de `imblearn.Pipeline`), nunca vaza.
- **Reprodutibilidade:** seeds fixadas, hash SHA-256 dos dados e linhagem no MLflow
  (Git SHA + hash do dataset).

---

## 📊 Reprodutibilidade com DVC

```bash
dvc repro        # reexecuta só os estágios afetados (preprocess -> train -> evaluate)
```

---

## 📖 Documentação

Guias e referência de API (gerada das docstrings) em MkDocs Material:

```bash
make docs-serve  # http://127.0.0.1:8000
```

Veja também o [Model Card](reports/model_cards/fraud_xgboost_model_card.md) e o
[Datasheet](reports/datasheets/creditcard_datasheet.md).

---

## 📝 Licença

Distribuído sob a licença [MIT](LICENSE).
