# Datasheet — Credit Card Fraud Detection (ULB)

## Motivação

- **Propósito:** detectar transações fraudulentas de cartão de crédito.
- **Origem:** conjunto público disponibilizado pela ULB (Université Libre de
  Bruxelles) em parceria com a Worldline, hospedado no Kaggle.

## Composição

- ~284.807 transações realizadas por portadores europeus em setembro de 2013.
- **Colunas:** `Time` (segundos desde a 1ª transação), `V1`–`V28` (componentes
  principais de PCA, anonimizados), `Amount` (valor da transação), `Class` (alvo:
  1 = fraude, 0 = legítima).
- **Desbalanceamento:** ~492 fraudes (~0,17%).

## Processo de coleta

- Transações reais anonimizadas. As features V1-V28 resultam de uma transformação
  PCA aplicada pelos publicadores para preservar a confidencialidade; os componentes
  originais e o significado de negócio não são divulgados.

## Pré-processamento (neste projeto)

- Remoção de linhas duplicadas exatas.
- Divisão estratificada (ou temporal) em treino/validação/teste.
- Engenharia de `Amount_log` (log1p) e `Hour` (hora do dia derivada de `Time`).
- Escalonamento robusto de `Amount` e derivadas.

## Licenciamento e uso

- Base pública para fins de pesquisa/educacionais (Database Contents License —
  ver a página do Kaggle). Verifique os termos antes de uso comercial.

## Privacidade / LGPD

- As features estão anonimizadas por PCA; **não há PII direta** no dataset.
- Ainda assim: nunca registrar dados sensíveis em logs/figuras; preferir bases
  públicas e devidamente licenciadas; considerar risco de reidentificação via
  quase-identificadores (aqui, mínimo, dado o PCA).
