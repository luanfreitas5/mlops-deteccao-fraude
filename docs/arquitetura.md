# Arquitetura

O código em `src/` é organizado por funcionalidade (Clean Architecture, baixo
acoplamento), nunca por um pacote nomeado pelo projeto.

| Pacote | Responsabilidade |
|---|---|
| `config` | Configuração validada (Pydantic), caminhos, logging, seeds. |
| `constants` | Colunas, rótulos e nomes de métricas do domínio. |
| `schemas` | Contrato de dados (pandera) e schemas de I/O da API (pydantic). |
| `data` | Carregamento, divisão e escrita de dados. |
| `preprocessing` | Limpeza e pré-processador scikit-learn. |
| `features` | Engenharia de features sem vazamento. |
| `models` | Fábrica de estimadores e persistência. |
| `training` | Montagem/treino de pipeline e validação cruzada. |
| `evaluation` | Métricas, threshold e avaliação com IC bootstrap. |
| `inference` | Preditor consistente com o treino (anti train–serve skew). |
| `experiment` | Rastreamento MLflow com linhagem (Git SHA + hash dos dados). |
| `pipelines` | Orquestração de ponta a ponta. |
| `visualization` | Tema e figuras de avaliação. |

## Decisões de projeto

- **Métrica principal PR-AUC:** com ~0,17% de fraudes, a ROC-AUC superestima o
  desempenho. A PR-AUC reflete o que importa: acertar a classe positiva.
- **Threshold por custo de negócio:** um falso negativo (fraude não detectada)
  custa muito mais que um falso positivo; o threshold é escolhido minimizando o
  custo esperado, não fixado em 0,5.
- **Consistência treino–serving:** a mesma função (`prepare_xy`) gera as features
  no treino e no serving; a API valida as requisições contra o schema de treino.
- **Desbalanceamento:** `scale_pos_weight` como padrão (rápido, sem sintetizar
  dados); SMOTE dentro de um `imblearn.Pipeline` (aplicado só no ajuste) como
  alternativa comparável.
