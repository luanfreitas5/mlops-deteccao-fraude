# Instalação

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- O dataset [Credit Card Fraud (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  em `data/raw/creditcard.csv`.

## Passos

```bash
# Instala dependências de runtime + dev
make install

# Instala também o extra da API
uv sync --dev --extra api

# Configura os hooks de pre-commit
make hooks
```

## Variáveis de ambiente

Copie `.env.example` para `.env` e preencha o necessário (ex.: URI do MLflow).
Nunca commite o `.env`.
