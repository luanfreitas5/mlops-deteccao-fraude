# Model Card — Detector de Fraude (XGBoost)

> Documento vivo. Preencha os campos de métricas após o primeiro treino real
> (`make pipeline`), a partir de `reports/test_metrics.json`.

## Detalhes do modelo

- **Nome:** `fraud_model` (XGBoost, `XGBClassifier`).
- **Versão:** 0.1.0.
- **Tipo:** classificação binária supervisionada.
- **Baseline de referência:** Regressão Logística com `class_weight="balanced"`.
- **Tratamento de desbalanceamento:** `scale_pos_weight = n_neg / n_pos`
  (SMOTE disponível como alternativa).
- **Pré-processamento:** `RobustScaler` em `Amount`/derivadas; V1-V28 (PCA) passam
  direto. Features: V1-V28, `Amount`, `Amount_log`, `Hour`.

## Uso pretendido

- **Uso previsto:** priorização de transações para revisão antifraude (score de
  risco + decisão limiarizada por custo de negócio).
- **Usuários:** times de risco/antifraude.
- **Fora de escopo:** decisão automática irreversível sem revisão humana; uso em
  populações/instituições com distribuição de dados muito distinta da de treino.

## Dados de treino

- Dataset Credit Card Fraud Detection (ULB), ~284k transações, ~0,17% de fraudes.
- Features anonimizadas por PCA (V1-V28) — sem PII direta. Ver o
  [Datasheet](../datasheets/creditcard_datasheet.md).

## Avaliação

- **Métrica principal:** PR-AUC (*Average Precision*), com IC 95% por bootstrap.
- **Complementares:** ROC-AUC, recall/precision/F1 no threshold de operação,
  Brier score (calibração).
- **Threshold:** otimizado por custo de negócio (FN >> FP) na validação.

| Métrica | Valor (teste) | IC 95% |
|---|---|---|
| PR-AUC | a preencher | a preencher |
| ROC-AUC | a preencher | — |
| Recall | a preencher | — |
| Precision | a preencher | — |
| Brier | a preencher | — |

### Avaliação por fatia

As features são anonimizadas por PCA, então não há atributos sensíveis diretos
(região, idade, sexo) disponíveis para auditoria de fairness. Recomenda-se, se
dados demográficos existirem em produção, reavaliar por subgrupo antes do deploy.

## Limitações e riscos

- **Deriva (drift):** o padrão de fraude muda com o tempo; monitorar dados e
  desempenho e retreinar quando justificado.
- **Threshold dependente de custo:** revisar os custos de FN/FP com o negócio.
- **Anonimização por PCA:** limita interpretabilidade e auditoria de viés.

## Considerações éticas

- Falsos positivos bloqueiam clientes legítimos; falsos negativos permitem
  fraude. O balanço deve ser decidido com o negócio, não apenas estatisticamente.
- Manter revisão humana nas decisões de maior impacto.
