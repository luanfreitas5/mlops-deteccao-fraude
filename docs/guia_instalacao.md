# Instalação

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- O dataset [Credit Card Fraud (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  em `data/bronze/creditcard.csv` (camada bronze: imutável, nunca editar).

## Passos

```bash
# Instala todos os grupos de dependências (dev, quality, docs, notebooks) e os hooks
make install
```

A API (FastAPI) já vem com as dependências de runtime; o extra `app` (Streamlit)
é opcional: `uv sync --extra app`.

## Variáveis de ambiente

Copie `.env.example` para `.env` e preencha o necessário (ex.: credenciais do
Postgres do Docker Compose). Nunca commite o `.env`.
