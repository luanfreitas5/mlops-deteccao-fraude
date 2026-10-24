# Datasheet — Credit Card Fraud Detection (ULB)

## 1. Motivação
- **Por que foi criado? Para qual tarefa?** Pesquisa em detecção de fraude em cartões de crédito sob desbalanceamento extremo de classes (classificação binária: fraude × legítima).
- **Quem criou e quem financiou?** Publicado pelo Machine Learning Group da ULB (Université Libre de Bruxelles) em colaboração com a Worldline, e hospedado no Kaggle ([mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)). Referência: Dal Pozzolo et al. (2015), *Calibrating Probability with Undersampling for Unbalanced Classification*. `TODO(owner): confirmar a fonte de financiamento na página do dataset.`
- **Uso neste projeto:** dados públicos para demonstrar um pipeline MLOps completo (portfólio); não são dados da organização do autor.

## 2. Composição
- **O que cada instância representa?** Uma transação de cartão de crédito.
- **Número de instâncias e período coberto:** 284.807 transações em dois dias de setembro de 2013 (cartões europeus); 283.726 após remover duplicatas exatas neste projeto.
- **Campos, tipos e contrato:** `Time` (segundos desde a primeira transação), `V1`–`V28` (componentes de PCA, anonimizados), `Amount` (valor), `Class` (alvo). Contrato validado com pandera em `src/schemas/dataset.py` (tipos, não negativos em `Time`/`Amount`, `Class ∈ {0, 1}`, sem nulos).
- **Rótulo:** `Class = 1` indica fraude, `0` transação legítima. A definição operacional do rótulo e a janela de confirmação não são documentadas pelos publicadores. `TODO(owner): citar a descrição da página do Kaggle, se houver.`
- **Dados ausentes (taxa por coluna):** 0% — o contrato exige `nullable=False` em `Time`, `Amount` e `Class`, e a carga falha se houver ausentes.
- **Campos sensíveis / PII e tratamento:** não há PII direta; V1–V28 resultam de PCA aplicada pelos publicadores e os atributos originais não são divulgados.
- **É amostra de uma população maior? Como amostrado?** Sim, é um recorte de dois dias; o método de amostragem não é documentado. Prevalência de fraude ≈ 0,172% (492 de 284.807).

## 3. Processo de coleta
- **Fonte(s) e sistema(s) de origem:** transações reais de cartões europeus, processadas pela Worldline; anonimizadas antes da publicação.
- **Período e frequência de coleta:** dois dias consecutivos de setembro de 2013; coleta única (sem atualização).
- **Consentimento / aviso aos titulares:** não documentado pelos publicadores. `TODO(owner): registrar se houve aviso/consentimento; relevante antes de qualquer uso comercial.`

## 4. Pré-processamento
- **Limpeza, filtragem e linhas removidas por regra:** remoção de linhas duplicadas exatas (`src/preprocessing/cleaning.py`); a contagem removida é registrada no log da execução. Engenharia de `Amount_log` (log1p) e `Hour` (hora derivada de `Time`) e `RobustScaler` em `Amount` ocorrem dentro do pipeline de modelagem.
- **Camadas (medalhão):** `data/bronze/creditcard.csv` (como recebido, imutável) → `data/silver/transactions.parquet` (validado e deduplicado) → `data/gold/{train,val,test}.parquet` (divisão estratificada 60/20/20, semente 42).
- **Anonimização / pseudonimização:** já aplicada na origem (PCA); nenhuma etapa adicional neste projeto. Não há segredo ou *salt* envolvido.
- **Hash SHA-256 dos dados brutos:** `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` (também em `data/bronze/manifest.json`, gerado a cada pré-processamento).

## 5. Usos
- **Usos pretendidos:** pesquisa e ensino em classificação com classes desbalanceadas; demonstração de práticas de MLOps.
- **Usos desaconselhados:** estimar o desempenho de um detector em produção real (dados de 2013, dois dias, só cartões europeus); decisões sobre pessoas; qualquer inferência sobre atributos demográficos (não existem no dataset).
- **Riscos de viés conhecidos:** viés temporal e geográfico; sem atributos sensíveis, não é possível auditar disparidades entre grupos; a PCA impede identificar variáveis proxy.

## 6. Distribuição
- **Licença e restrições:** Database Contents License (DbCL) v1.0, conforme a página do Kaggle. Verifique os termos antes de qualquer uso comercial. `TODO(owner): confirmar a licença vigente na página do dataset.`
- **Onde está armazenado:** o CSV **não** é versionado no Git (`/data/bronze/*` no `.gitignore`); é baixado manualmente do Kaggle. `TODO(owner): configurar o remote do DVC (S3, GDrive, etc.) para as camadas bronze/silver/gold.`

## 7. Manutenção
- **Responsável:** Luan Freitas (cópia de trabalho do projeto); o dataset original é mantido pela ULB/Worldline.
- **Frequência de atualização:** nenhuma — o dataset original é estático. Mudanças de hash no bronze devem ser tratadas como incidente.
- **Como atender pedidos de exclusão de titulares:** os dados são anonimizados e não permitem localizar um titular; pedidos devem ser encaminhados aos publicadores originais. Se o pipeline passar a usar dados reais, é preciso incluir uma chave pseudonimizada para localizar e remover registros.

## 8. LGPD
- **Base legal (art. 7 / art. 11):** dados considerados anonimizados (art. 12 da LGPD) pelos publicadores. `TODO(owner): confirmar com o jurídico se o risco de reidentificação por quase-identificadores é irrelevante para o seu uso.`
- **Prazo de retenção:** `TODO(owner): definir — para uso em portfólio, manter apenas enquanto o projeto existir.`
- **Relatório de impacto (RIPD) necessário?** Não para este dataset público e anonimizado; seria necessário ao adotar dados reais de clientes. `TODO(owner): link do RIPD, se aplicável.`
