"""Definição dos modelos comparados no artigo (um único lugar para todos os scripts).

Cada modelo é um Pipeline do scikit-learn: primeiro o pré-processamento (transformar textos
em números), depois o classificador. Assim o mesmo tratamento vale no treino e no teste, e
src/train.py e src/experimentos.py usam exatamente os mesmos modelos.

Modelos (todos com os hiperparâmetros fixados ANTES de olhar o teste):
    baseline : sempre prevê a classe mais comum (referência mínima)
    lr       : Regressão Logística Multinomial (modelo estatístico clássico)
    hgb      : Gradient Boosting baseado em histogramas (HistGradientBoostingClassifier)
    rf       : Random Forest Classifier
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import preprocessador_arvores, preprocessador_linear

# "Semente" do sorteio aleatório. Fixar o valor faz o resultado ser o mesmo toda vez que o
# script roda, o que permite reproduzir os números do artigo.
SEMENTE = 42

# Random Forest: 200 árvores e limites que evitam "decorar" os dados.
#   max_depth        = profundidade máxima de cada árvore
#   min_samples_leaf = mínimo de sinistros em cada folha
#   class_weight     = "balanced" dá mais peso às classes raras (ex.: FATAL)
N_ARVORES = 200
PARAMS_RF = dict(max_depth=20, min_samples_leaf=5, class_weight="balanced")

# Gradient Boosting: valores padrão da biblioteca (taxa de aprendizado 0,1; até 100 iterações,
# com parada antecipada usando 10% do treino; folhas com pelo menos 20 sinistros), mais a mesma
# ponderação de classes. Diferente do Random Forest, as árvores são treinadas em sequência:
# cada nova árvore corrige os erros das anteriores.
PARAMS_HGB = dict(class_weight="balanced")

# Regressão Logística: mesma ponderação de classes; max_iter é o limite de iterações do ajuste
# (ela converge em cerca de 150).
PARAMS_LR = dict(max_iter=500, class_weight="balanced")

# Nomes usados em tabelas, figuras e no artigo
NOMES = {
    "baseline": "Baseline (classe mais comum)",
    "lr": "Regressão Logística Multinomial",
    "hgb": "Gradient Boosting",
    "rf": "Random Forest",
}
# Ordem em que os modelos aparecem nas tabelas (do mais simples ao mais complexo)
ORDEM = ["baseline", "lr", "hgb", "rf"]


def montar_modelo(tipo, cat, num, n_arvores=N_ARVORES, **params):
    """Monta o Pipeline de um modelo. Os `params` substituem os valores padrão acima.

    tipo: "baseline", "lr", "hgb" ou "rf"; cat/num: listas de colunas categóricas e numéricas.
    """
    if tipo == "baseline":
        return Pipeline([("pre", preprocessador_arvores(cat, num)),
                         ("clf", DummyClassifier(strategy="most_frequent"))])
    if tipo == "lr":
        # Regressão Logística precisa de one-hot nas categorias e escala nas numéricas
        return Pipeline([("pre", preprocessador_linear(cat, num)),
                         ("clf", LogisticRegression(**{**PARAMS_LR, **params}))])
    if tipo == "hgb":
        # As primeiras colunas que saem do pré-processamento são as categóricas: o Gradient
        # Boosting as trata como categorias de verdade (sem impor uma ordem entre elas).
        return Pipeline([("pre", preprocessador_arvores(cat, num)),
                         ("clf", HistGradientBoostingClassifier(
                             categorical_features=list(range(len(cat))), random_state=SEMENTE,
                             **{**PARAMS_HGB, **params}))])
    if tipo == "rf":
        # n_jobs=-1 usa todos os núcleos do processador
        return Pipeline([("pre", preprocessador_arvores(cat, num)),
                         ("clf", RandomForestClassifier(n_estimators=n_arvores, n_jobs=-1,
                                                        random_state=SEMENTE,
                                                        **{**PARAMS_RF, **params}))])
    raise ValueError(f"tipo de modelo desconhecido: {tipo}")
