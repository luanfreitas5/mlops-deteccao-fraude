# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não lançado]

### Adicionado
- Estrutura inicial do projeto MLOps de detecção de fraude.
- Pipeline de pré-processamento (carregamento, validação com pandera, limpeza,
  divisão estratificada/temporal e persistência em Parquet).
- Engenharia de features (`Amount_log`, `Hour`) sem vazamento.
- Fábrica de modelos (Regressão Logística como baseline e XGBoost como principal).
- Tratamento de desbalanceamento por `scale_pos_weight` e SMOTE.
- Otimização de threshold por recall-alvo e por custo de negócio.
- Avaliação rigorosa: PR-AUC como métrica principal, IC 95% por bootstrap,
  calibração (Brier) e figuras (PR, ROC, matriz de confusão).
- Rastreamento de experimentos com MLflow (linhagem via Git SHA + hash dos dados).
- API de inferência com FastAPI, validando requisições contra o schema de treino.
- Suíte de testes (unitários, property-based, comportamentais e de API).
- CI/CD, Docker, docker-compose, DVC, MkDocs, Model Card e Datasheet.

[Não lançado]: https://github.com/luanfreitas5/mlops-deteccao-fraude
