# Uso

## Pipeline de ML

```bash
make preprocess   # raw -> processed (limpeza, divisão, Parquet)
make train        # treina, otimiza o threshold, salva modelo + métricas
make evaluate     # avalia no teste, gera figuras e relatórios
make pipeline     # executa tudo de ponta a ponta
```

Ou diretamente:

```bash
python -m src.main all --model xgboost
```

## Reprodutibilidade com DVC

```bash
dvc repro         # reexecuta apenas os estágios afetados
```

## API de inferência

```bash
make serve        # http://localhost:8000/docs
```

Exemplo de requisição:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 0.0, "V1": -1.36, "V2": -0.07, ..., "Amount": 149.62}'
```

## MLflow

```bash
make mlflow       # UI em http://localhost:5000
```
