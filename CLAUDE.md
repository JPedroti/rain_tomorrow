# CLAUDE.md — Projeto 04: Rain Tomorrow (classificação + estabilidade temporal)

Regras permanentes deste repositório. Valem para qualquer sessão, script, notebook ou teste.
Em caso de conflito, a ordem de precedência é: **enunciado oficial** (`rain_tomorrow_classificacao_estabilidade_temporal.docx.pdf`)
> este arquivo > convenções gerais de ML.

## 1. Objetivo (do enunciado)

Estimar `P(RainTomorrow = Yes)` com **Regressão Logística** (único algoritmo) sobre `data/weatherAUS.csv` e responder:
*o modelo desenvolvido com dados até 2016 mantém a capacidade de discriminar chuva/não chuva em 2017?*
Avaliação quantitativa obrigatória: **ROC-AUC, PR-AUC e KS**. **Não otimizar threshold.**

## 2. Fatos verificados do dataset (auditoria de 2026-09-28)

Verificados por execução de código sobre o arquivo real. Não reescrever de memória; se algo mudar, re-verificar.

- Arquivo: Kaggle `jsphyg/weather-dataset-rattle-package`, Version 2. SHA-256
  `573fd715cd69fcacc4df32024d823b450ae3edaae7e8ff2eeb623adbed424014`. 145.460 linhas × 23 colunas. **Não contém `RISK_MM`.**
- Datas: 2007-11-01 → 2017-06-25. Sem duplicatas de (Location, Date). 49 localidades.
- Meses inteiramente ausentes no histórico: 2011-04, 2012-12, 2013-02. Nhil, Uluru e Katherine só começam em 2013-03.
- 2017 (OOT) — **apenas metadados estruturais foram inspecionados**: 8.623 linhas, 49 localidades (todas presentes antes de 2017),
  meses 2017-01 a 2017-06 (junho parcial, até dia 25), 157 linhas sem target.
- Target (verificado em `Date < 2017`): `RainTomorrow(t) == RainToday(t+1)` em 100% dos 133.546 pares de dias consecutivos por
  localidade; `RainToday = 1{Rainfall > 1,0 mm}` (estritamente maior; 1,0 mm → `No`). Pela página do Kaggle, a chuva do dia é
  medida "nas 24h até as 9h" ⇒ a janela do target vai **das 9h do dia t às 9h do dia t+1**.
- Em `Date < 2017`, os 3.110 targets ausentes correspondem todos a chuva do dia seguinte não medida.

## 3. Regra temporal e OOT (inegociável)

1. **OOT = todas as linhas com `Date >= 2017-01-01`.** Desenvolvimento = `Date < 2017-01-01`. O corte é feito **antes de qualquer fit**.
2. **Nada** de 2017 pode influenciar: escolha de features, preprocessing, imputação, encoding, scaling, hiperparâmetros,
   regularização, class_weight, janelas/bins de análise, critérios de volume mínimo ou qualquer decisão de modelagem.
3. Antes do congelamento, sobre 2017 só é permitido ler metadados estruturais (contagem de linhas, localidades, meses, linhas sem target).
   É **proibido** olhar taxa de target, distribuições de features ou missingness por coluna de 2017 antes do congelamento.
4. **Protocolo de congelamento:** (a) seleção de modelo concluída só com CV no TRAIN; (b) `scripts/train.py` salva
   `artifacts/model.joblib` + hash SHA-256 em `artifacts/metadata.json`; (c) só então a avaliação OOT carrega o artefato
   congelado, verifica o hash e **nunca chama `fit`**.
5. Depois que o OOT for aberto, **não se volta a mudar o modelo** com base nele. Se um bug real for encontrado após abrir o OOT,
   documentar explicitamente no README (o que mudou, por quê, e que o OOT deixou de ser totalmente "cego").
6. O teste aleatório **não substitui** o OOT; o OOT **não é** conjunto de validação.

## 4. Data leakage — regras

- Features permitidas: apenas colunas brutas do dataset exceto `RainTomorrow` (e `RISK_MM` se algum dia aparecer), mais
  features simples derivadas de `Date` (mês/dia do ano). **Proibido** usar `year` (extrapolação para 2017 e codifica tendência).
- Proibido qualquer feature construída com linhas futuras (`shift(-k)`, janelas centradas, agregados por localidade calculados
  fora do TRAIN). Como `RainTomorrow(t) == RainToday(t+1)`, qualquer lead de `RainToday`/`Rainfall` é o próprio target.
- Proibido target encoding. `Location` é categórica comum (OneHotEncoder).
- Todo preprocessing mora dentro de **um único `Pipeline`** (ColumnTransformer + LogisticRegression) ajustado **somente no TRAIN**.
  CV de hiperparâmetros usa o Pipeline inteiro dentro de cada fold. TEST, OOT e inferência só recebem `transform`/`predict_proba`.
- O ColumnTransformer seleciona colunas explicitamente (`remainder="drop"`): colunas extras no input (ex.: `RainTomorrow`) são ignoradas.
- Linhas sem `RainTomorrow` são excluídas de treino e de métricas (não se imputa target). Reportar quantas foram excluídas.

### Decisões aprovadas pelo usuário (2026-09-28)

- Previsão tratada como emitida **ao fim do dia t** (00:00 de t+1). Janelas de medição em `src/config.py` (`FEATURE_WINDOWS`,
  fonte: notas do BoM, *Notes to accompany Daily Weather Observations*).
- **Modelo final com 20 features** (`FINAL_FEATURE_SET = "full_without_maxtemp"`): `MaxTemp` excluída porque sua janela
  (09:00 de t → 09:00 de t+1) é a janela do alvo e fecha depois do momento da previsão. O conjunto com as 21 features e o
  conjunto "só até 09:00" existem apenas como **análise de sensibilidade informativa** (CV + TEST), nunca avaliados no OOT.
- Split 80/20 estratificado, `random_state=42`; linhas sem target removidas; README/notebook em português, código em inglês;
  o PDF do enunciado não é versionado.
- **Modelo congelado em 2026-09-28T22:41:09Z** (SHA-256 `b274388f39b24a3361f914cd78161d3dcd09fbe96a16273a6ebc2534c67be3ff`;
  `indicators`, C=0,1, L2, `class_weight=None`). **OOT aberto em 2026-09-28T23:12:29Z.** Desde então modelo, preprocessing,
  features, hiperparâmetros, threshold e calibração são **imutáveis**. Qualquer mudança de modelagem é um novo ciclo de
  desenvolvimento, que exige novo período OOT e registro explícito no README.

## 5. Modelagem

- Único algoritmo: `sklearn.linear_model.LogisticRegression`. Não comparar outros algoritmos.
- scikit-learn 1.9: `penalty` está **deprecated**. Expressar a penalização via `l1_ratio` (0 = L2, 1 = L1) + `C`;
  solvers compatíveis: L2 → `lbfgs`/`newton-cholesky`/`saga`; L1 → `liblinear`/`saga`. Documentar o mapeamento penalty ↔ l1_ratio.
- Seleção de hiperparâmetros: somente CV no TRAIN (StratifiedKFold com seed). A métrica de seleção deve ser documentada.
  O TEST é usado para reportar baseline e modelo final, **não** para escolher configuração.
- Classe positiva: `RainTomorrow = "Yes"` → 1. Score = `predict_proba[:, 1]`.

## 6. Métricas (definições fixas; mesmo procedimento em todos os conjuntos)

- ROC-AUC: `sklearn.metrics.roc_auc_score`.
- PR-AUC: `sklearn.metrics.average_precision_score` (Average Precision, sem interpolação trapezoidal). Sempre reportar a prevalência junto.
- KS: `max(TPR − FPR)` da curva ROC (= estatística KS de duas amostras entre scores de positivos e negativos), em escala 0–1.
- Janela com uma única classe ou volume abaixo do mínimo pré-definido → métrica `NaN` + flag explicando o motivo. Nunca forçar.
- Métricas dependentes de threshold (accuracy, precision, recall, specificity, F1, matriz de confusão) só aparecem para fins
  didáticos, com thresholds ilustrativos. **Nenhum threshold é escolhido ou otimizado.** `predict.py` devolve probabilidade, não classe.

## 7. Estabilidade e drift

- Estabilidade: métricas do OOT por mês (`reports/temporal_performance.csv`), com n, positivos e taxa de target por janela.
- PSI: bins definidos **no TRAIN** (quantis) e reaplicados ao OOT; missing é um bin próprio; epsilon para bins vazios.
- KS de drift: `scipy.stats.ks_2samp` TRAIN vs OOT; reportar principalmente a estatística (p-value com ressalva de n grande).
- Cortes de PSI (~0,10 / ~0,25) são heurística, não regra automática. Sempre discutir **sazonalidade** (OOT só tem jan–jun).

## 8. Reprodutibilidade e engenharia

- Caminhos relativos à raiz do projeto. Seeds fixas (`RANDOM_STATE = 42` salvo decisão documentada).
- Versões fixadas em `requirements.txt` (ambiente de referência: Python 3.11.9, Windows 11).
- Notebook sem estado oculto: deve rodar do início ao fim com `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/01_exploration_modeling.ipynb`,
  **com o `.venv` ativado** (o kernelspec chama `python` pelo PATH; sem ativar, nesta máquina o kernel roda o Python global).
- `pip` nesta máquina: o Avast intercepta HTTPS; instalar com `--cert "C:\ProgramData\Avast Software\Avast\wscert.pem"` (a verificação SSL continua ligada).
- Notebook e scripts usam as **mesmas funções de `src/`**. Nada de lógica de preprocessing duplicada.
- `data/*.csv` **nunca** é versionado (`.gitignore`); `data/README.md` explica como obter o arquivo e traz o SHA-256.
- Commits apenas quando o usuário pedir.

## 9. Validação obrigatória

- Após qualquer mudança relevante: executar o código, ler erros **e warnings**, corrigir, reexecutar.
- Rodar a suíte: `python -m pytest -q`. Não declarar nada como "funcionando" sem execução.
- Testes cobrem obrigatoriamente: separação temporal/OOT, ausência de leakage no preprocessing (fit só no TRAIN),
  cálculo de métricas (ROC-AUC, PR-AUC, KS) e de drift (PSI, KS), contrato do `predict.py` (ordem das linhas, colunas,
  probabilidades em [0, 1]), persistência/carregamento do modelo em processo novo.

## 10. Proibição de resultados inventados

- **Todo número** em README, notebook, relatórios ou mensagens deve vir de código executado ou de artefato gerado por ele.
- Nunca escrever métricas, taxas, contagens ou conclusões "esperadas" antes de rodar. Se ainda não foi executado, escrever `PENDENTE`.
- Afirmações sobre o dataset precisam de verificação no arquivo; afirmações externas (ex.: convenções do BoM) devem ser marcadas
  como tal, com a fonte, e separadas do que foi verificado nos dados.
- Ao revisar resultados, preferir explicar do que esconder: resultado ruim no OOT é resultado, não erro a ser "corrigido".
