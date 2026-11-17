# 🚦 Predição da Severidade de Sinistros nas Rodovias Federais Brasileiras

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-entrega%20final-green.svg)](#)

Código do trabalho de Resolução de Problemas II (EACH-USP): **comparação de modelos de aprendizado de máquina para prever a severidade de sinistros nas rodovias federais brasileiras** (sem vítimas, com feridos ou fatal), com os dados abertos da Polícia Rodoviária Federal (PRF) de 2023 a 2025.

São comparados quatro modelos com os mesmos dados e a mesma divisão treino/teste: um **baseline** (prevê sempre a classe mais comum), a **Regressão Logística Multinomial**, o **Gradient Boosting** (HistGradientBoosting) e o **Random Forest**. Além do teste, a comparação é conferida em validação cruzada, temporal (2025) e espacial (estados fora do treino), com ajuste de hiperparâmetros, ajuste da regra de decisão e ablação de atributos.

---

## 👥 Integrantes do Grupo

*   **André P. Lino**
*   **Karina Y. Chen**
*   **Yago B. Costa**

---

## 📌 Objetivo

Avaliar em que medida os atributos registrados pela PRF sobre as condições de cada sinistro (horário, clima, via e localização) permitem prever a sua severidade, comparando os modelos entre si e com o baseline, e identificar quais grupos de atributos mais contribuem para essa predição.

---

## 📁 Estrutura

```text
notebooks/
  01_coleta_e_limpeza.ipynb        consolida e limpa os CSVs da PRF -> data/processed/
  02_analise_exploratoria_eda.ipynb  análise exploratória completa (figuras 01 a 05)
src/
  data_loader.py     lê a base tratada
  preprocessing.py   atributos de entrada (sem vazamento), divisão 80/20 e pré-processadores
  modelos.py         os quatro modelos (um único lugar para todos os scripts)
  avaliacao.py       métricas, intervalo de confiança (bootstrap), regra de decisão e importância
  visualization.py   figuras
  train.py           resultados principais (teste, validação cruzada, importância)
  experimentos.py    validações temporal e espacial, ajuste, regra de decisão e ablação
  descritiva.py      proporção de sinistros fatais por condição (caracterização da base)
results/             tabelas e métricas citadas no artigo
figures/             figuras do artigo
data/                dados (não vão para o GitHub; ver abaixo)
```

---

## 💾 Acesso aos Dados Brutos (Google Drive)

Devido ao limite de tamanho de arquivos suportado pelo GitHub, os arquivos brutos (`.csv`) da PRF (2023–2025) e as tabelas complementares estão hospedados em uma pasta compartilhada do Google Drive.

*   🔗 **[Acessar Repositório de Dados no Google Drive](https://drive.google.com/drive/folders/1zCGJHtUm_F3HZ3kj2-8XtjhBuqwpa_8A?usp=sharing)**

1. Baixe os arquivos brutos da PRF e salve-os em `data/raw/`.
2. Rode o notebook `notebooks/01_coleta_e_limpeza.ipynb`, que gera `data/processed/prf_2023_2025_limpo.csv` (ou baixe essa base tratada do Drive e salve-a nessa pasta).

---

## 🤖 Como rodar

Rode tudo a partir da **raiz do projeto**, de preferência em um ambiente virtual (`.venv`):

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

> No PowerShell, se aparecer erro de execução de scripts, rode `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de novo. Se a instalação do `geopandas` falhar no Windows, ele só é necessário para os notebooks de mapas; para os modelos bastam `pandas numpy matplotlib scikit-learn`.

```bash
pip install -r requirements.txt
python -m src.descritiva      # caracterização da base (segundos)
python -m src.train           # resultados principais (cerca de 10 minutos)
python -m src.experimentos    # experimentos complementares (cerca de 40 minutos)
```

Opções úteis: `python -m src.train --folds 0` pula a validação cruzada; `python -m src.experimentos --etapas temporal ablacao` roda só algumas etapas; `--n-candidatos 4` deixa a busca de hiperparâmetros menor.

**Atenção a vazamento de dados:** as colunas `mortos`, `feridos_*`, `ilesos`, `pessoas`, `classificacao_acidente` e `indice_severidade` entregam a resposta e **não** são usadas como entrada. A causa e o tipo do acidente só são conhecidos depois do sinistro: ficam fora do modelo principal e aparecem só na ablação (análise de sensibilidade).

A divisão é 80% treino / 20% teste, estratificada, com `random_state=42`. O conjunto de teste **nunca** é usado para escolher nada (hiperparâmetros, pesos da regra de decisão): só mede o resultado final.

---

## 📊 Principais resultados (conjunto de teste, 42.690 sinistros)

| Modelo | F1 macro [IC 95%] | ROC-AUC | AP fatal | F1 macro com regra ajustada |
|---|---|---|---|---|
| Baseline (classe mais comum) | 0,290 [0,289; 0,291] | 0,500 | 0,072 | — |
| Regressão Logística Multinomial | 0,375 [0,371; 0,380] | 0,643 | 0,165 | 0,428 [0,422; 0,434] |
| Gradient Boosting | 0,386 [0,381; 0,391] | 0,660 | 0,170 | 0,435 [0,429; 0,441] |
| Random Forest | 0,410 [0,405; 0,415] | 0,658 | 0,162 | 0,432 [0,426; 0,438] |

- Com a regra de decisão padrão, o Random Forest tem o maior F1 macro, e a ordem se repete na validação cruzada (0,410; 0,386; 0,374) e em 2025 (0,408; 0,382; 0,373). Com estados inteiros fora do treino, a vantagem some (0,364; 0,364; 0,358).
- Nas métricas que não dependem do ponto de decisão (ROC-AUC e AP), os três modelos ficam próximos, e com a regra de decisão ajustada no treino eles praticamente empatam: a diferença vinha mais do ponto de operação do que de uma separação melhor das classes.
- Sem os atributos de localização, o F1 macro cai para 0,354 a 0,367; com a causa e o tipo do acidente (só conhecidos depois do sinistro), sobe para 0,482 a 0,524. O limite está mais na informação disponível do que no algoritmo.
- Localização e horário são os grupos de atributos mais importantes nos três modelos.

### Saídas

| Arquivo | Conteúdo |
|---|---|
| `results/metricas.json`, `results/relatorio_classificacao.txt` | métricas de cada modelo no teste (geral e por classe) |
| `results/intervalos_bootstrap.csv` | IC 95% do F1 macro e das diferenças entre modelos |
| `results/precisao_por_revocacao.csv` | precisão dos fatais detectando 30%, 50% e 70% deles |
| `results/validacao_cruzada.csv` | 5 partes do treino: média, desvio e F1 de cada parte |
| `results/importancia_grupos.csv`, `results/importancia_variaveis.csv` | importância por permutação de grupos (3 modelos) e por impureza (Random Forest) |
| `results/validacao_temporal.csv`, `results/validacao_espacial.csv` | treino 2023–2024 e teste 2025; estados fora do treino |
| `results/ajuste_hiperparametros.csv`, `results/modelos_ajustados.csv` | busca de hiperparâmetros e avaliação no teste |
| `results/decisao.csv`, `results/decisao_curva.csv`, `results/decisao_bootstrap.csv` | regra de decisão ajustada fora da amostra do treino; curva por peso da classe fatal; IC 95% com a regra ajustada |
| `results/ablacao.csv` | completo × sem localização × com causa e tipo do acidente |
| `results/descritiva_letalidade.csv` | % de sinistros fatais por horário, pista, traçado e UF |
| `figures/06` a `figures/10` | matrizes de confusão, importância, curva precisão-revocação e letalidade por condição |
| `data/outputs/random_forest.joblib` | modelo treinado (grande, não vai para o GitHub) |
