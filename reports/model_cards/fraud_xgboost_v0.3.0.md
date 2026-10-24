# Model Card — Detector de Fraude XGBoost v0.3.0

**Status:** DRAFT
**Bloqueios pendentes:** (1) comparação com o baseline (Regressão Logística) e teste de significância — bloqueiam qualquer afirmação de "melhor que o baseline"; (2) análise por fatias e curva de calibração — bloqueiam aprovação para uso em produção; (3) Git SHA e run do MLflow do modelo publicado — bloqueiam a promoção no Model Registry.

## 1. Detalhes do modelo
- **Responsável / contato:** Luan Freitas — luan.freitas@gmail.com (projeto de portfólio)
- **Data:** 2026-10-24 (card); artefato `models/fraud_model.joblib` gerado em 2026-07-30
- **Algoritmo:** `imblearn.Pipeline` = pré-processador (`RobustScaler` em `Amount`/`Amount_log`; V1–V28 e `Hour` passam direto) + `XGBClassifier` (400 árvores, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight = n_neg / n_pos`). Hiperparâmetros em `configs/model_params.yaml`; SMOTE disponível como alternativa (`imbalance.strategy`).
- **Rastreabilidade:** Git SHA `TODO(owner): SHA do commit que gerou o artefato (o modelo foi treinado antes do registro de linhagem completa no MLflow)` · MLflow run `TODO(owner): run_id do treino que gerou o artefato` · hash dos dados `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` (SHA-256 de `data/bronze/creditcard.csv`)
- **Licença:** MIT (código). Dados sob a Database Contents License do dataset ULB — ver [Datasheet](../datasheets/creditcard.md).

## 2. Uso pretendido
- **Uso primário:** priorizar transações de cartão para revisão antifraude, combinando score de risco e decisão limiarizada por custo de negócio.
- **Usuários pretendidos:** times de risco/antifraude que consomem o score via API (`POST /predict`).
- **Humano no circuito:** sim — a decisão limiarizada encaminha a transação para revisão; bloqueio automático irreversível não é um uso validado.

## 3. Usos fora do escopo
- Bloquear ou negar transações/clientes automaticamente, sem revisão humana.
- Uso em instituições, países, bandeiras ou períodos diferentes do dataset (cartões europeus, 2 dias de setembro de 2013): **não há validação externa**.
- Qualquer decisão sobre pessoas (crédito, cadastro, perfilamento): as features são componentes PCA anonimizados e não permitem auditoria de viés.
- Uso como detector único em produção: o desempenho foi medido em um recorte de 2 dias, sem deriva temporal.

## 4. Fatores
- **Grupos relevantes:** nenhum atributo demográfico ou sensível está disponível (V1–V28 são PCA). Faixas de valor (`Amount`) e hora derivada de `Time` são os únicos recortes viáveis — ainda não avaliados (ver seção 8).
- **Ambientes:** transações de cartão de crédito de portadores europeus; `Time` é o deslocamento em segundos desde a primeira transação do dataset, e `Hour` derivada dele **não** é a hora do relógio local.

## 5. Métricas
- **Métrica principal:** PR-AUC (*Average Precision*) com IC 95% por bootstrap (1000 reamostragens). Justificativa: com ~0,17% de fraudes, a ROC-AUC é otimista; a PR-AUC mede o desempenho sobre a classe rara. Um falso negativo (fraude não detectada) custa muito mais que um falso positivo (revisão de transação legítima).
- **Limiar de decisão:** 0,0402 — escolhido na **validação** minimizando o custo esperado com custo FN = 100 e FP = 1 (`business_cost` em `configs/config.yaml`; custo esperado na validação = 1028). Os custos são hipóteses de projeto, não valores medidos de um negócio real.
- **Calibração:** Brier no teste = 0,000404. `TODO(owner): gerar a curva de confiabilidade (reports/figures/calibration_curve.png)`; sem ela, não usar a probabilidade como risco absoluto.

## 6. Dados de avaliação
Partição de **teste** estratificada (20%) do dataset deduplicado, nunca usada no ajuste nem na escolha do limiar: ≈ 56,7 mil transações com ≈ 95 fraudes (a recall de 0,8105 equivale a 77 de 95). Divisão **aleatória estratificada**, não temporal: transações do mesmo período aparecem no treino e no teste, o que tende a ser otimista em relação a um cenário prospectivo. Fonte dos números: [`reports/metrics/test_metrics.json`](../metrics/test_metrics.json).

## 7. Dados de treino
Partição de treino (60%) e validação (20%) do mesmo dataset (ULB Credit Card Fraud, 284.807 transações, 492 fraudes; 283.726 após remover duplicatas exatas). Detalhes, licença e considerações de privacidade no [Datasheet](../datasheets/creditcard.md).

## 8. Análise quantitativa
Resultados no conjunto de teste (limiar 0,0402). IC 95% da PR-AUC por bootstrap; demais métricas são pontuais.

| Recorte | n | Métrica ± IC 95% | Baseline |
|---|---|---|---|
| Geral — PR-AUC | ≈ 56,7 mil (≈ 95 fraudes) | **0,8153** (0,7324–0,8921) | `TODO(owner): rodar make train MODEL=logistic_regression e registrar a PR-AUC` |
| Geral — ROC-AUC | idem | 0,9682 | `TODO(owner)` |
| Geral — Recall / Precision / F1 | idem | 0,8105 / 0,7264 / 0,7662 | `TODO(owner)` |
| Geral — Brier | idem | 0,000404 | `TODO(owner)` |
| Faixa de `Amount` (baixo/médio/alto) | `TODO(owner)` | `TODO(owner): PR-AUC e recall por faixa, com IC` | `TODO(owner)` |

A validação (usada para escolher o limiar) apresentou PR-AUC 0,8891 (IC 95%: 0,8230–0,9422) e recall 0,8936 — acima do teste, como esperado para métricas sobre dados usados na seleção do limiar. Os ICs são largos porque há poucas dezenas de fraudes por partição: a diferença entre validação e teste está dentro dessa incerteza.

Significância contra o baseline: `TODO(owner): McNemar no mesmo conjunto de teste e teste t reamostrado corrigido sobre os folds da CV`.

## 9. Considerações éticas e fairness
- **Atributos sensíveis auditados:** nenhum — o dataset não contém atributos demográficos; a auditoria de fairness com `fairlearn` **não é possível** com estes dados. Não se pode afirmar ausência de viés, apenas que ele não foi medido.
- **Disparidades encontradas:** não avaliadas.
- **Mitigação aplicada:** nenhuma.
- **Impacto dos erros:** falsos positivos bloqueiam ou incomodam clientes legítimos; falsos negativos permitem fraude. O equilíbrio (custos FN/FP) deve ser decidido com o negócio, não só estatisticamente.
- **Base legal LGPD:** o dataset é público e anonimizado por PCA (sem PII direta); ver o Datasheet. `TODO(owner): confirmar com o jurídico antes de uso com dados reais de clientes.`

## 10. Limitações, riscos e monitoramento
- **Limitações conhecidas:** apenas 2 dias de dados; ≈ 95 fraudes no teste (ICs largos); divisão aleatória (não temporal); sem validação externa; baseline ainda não medido; `Hour` é derivada de `Time` relativo; PCA limita interpretabilidade.
- **Mudanças de distribuição previstas:** padrões de fraude mudam com o tempo e com a reação dos fraudadores; sazonalidade (datas comemorativas) não está representada.
- **Monitoramento:** `TODO(owner)`: nenhum monitoramento implantado. Plano: relatório semanal de deriva de dados (`evidently`) e, quando os rótulos chegarem, PR-AUC em produção; alerta quando a PR-AUC cair abaixo do limite inferior do IC do teste.
- **Critério de retreino / descontinuação:** retreinar quando a deriva ou a queda de PR-AUC em produção ultrapassar o limite acordado; descontinuar se a PR-AUC ficar abaixo da do baseline.
