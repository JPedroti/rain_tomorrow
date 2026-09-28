# Dados

O arquivo de dados **não é versionado** (ver `.gitignore`). Para reproduzir o projeto:

1. Baixe o dataset oficial em <https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package>
   (*Rain in Australia*, Version 2; é preciso estar logado no Kaggle).
2. Descompacte e coloque o arquivo em `data/weatherAUS.csv`.
3. Confira a integridade: o arquivo usado neste projeto tem SHA-256

   ```
   573fd715cd69fcacc4df32024d823b450ae3edaae7e8ff2eeb623adbed424014
   ```

   ```bash
   python -c "import hashlib;print(hashlib.sha256(open('data/weatherAUS.csv','rb').read()).hexdigest())"
   ```

   `scripts/train.py` avisa se o hash for diferente.

## O que o arquivo contém (verificado)

- 145.460 linhas × 23 colunas; uma linha = observações de uma estação meteorológica em um dia.
- Período: 2007-11-01 a 2017-06-25; 49 localidades; não há `RISK_MM` nesta versão.
- Alvo `RainTomorrow` (Yes/No) = `RainToday` do dia seguinte da mesma estação, isto é,
  chuva > 1 mm nas 24h entre 09:00 do dia t e 09:00 do dia t+1.
- Fonte primária: Australian Bureau of Meteorology (observações diárias), compiladas pelo projeto Rattle.
  As janelas de medição de cada variável estão em `src/config.py` (`FEATURE_WINDOWS`), conforme as notas
  do BoM (<https://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml>).
