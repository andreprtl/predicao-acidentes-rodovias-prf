"""Métricas e análises de avaliação usadas no treino e nos experimentos.

Reúne em um só arquivo:
    - resumir / avaliar: as métricas do artigo (acurácia, F1, ROC-AUC, precisão média dos fatais...)
    - intervalos_bootstrap: intervalo de confiança de 95% do F1 macro no teste
    - escolher_pesos / prever_com_pesos: ajuste da regra de decisão (ponto de operação)
    - importancia_por_grupos: importância por permutação de grupos de variáveis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, roc_auc_score)

from src.preprocessing import CLASSES

# Ordem alfabética das classes = ordem das colunas de predict_proba no scikit-learn
ORDEM_PROBA = sorted(CLASSES)
# Pesos testados no ajuste da regra de decisão (1,0 = não mexe na probabilidade da classe)
GRADE_PESOS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def resumir(y_real, previsto, proba):
    """Métricas de um conjunto de previsões.

    y_real/previsto: classes (texto); proba: probabilidades na ordem alfabética das classes.
    """
    rel = classification_report(y_real, previsto, labels=CLASSES, output_dict=True, zero_division=0)
    eh_fatal = (np.asarray(y_real) == "FATAL").astype(int)
    return {
        # Acurácia: % de sinistros classificados corretamente (enganosa com classes desbalanceadas)
        "acuracia": accuracy_score(y_real, previsto),
        # F1 macro: média do F1 das 3 classes, todas com o mesmo peso (métrica principal do artigo)
        "f1_macro": f1_score(y_real, previsto, average="macro"),
        # F1 weighted: média do F1 ponderada pelo tamanho de cada classe
        "f1_weighted": f1_score(y_real, previsto, average="weighted"),
        # ROC-AUC: quão bem o modelo ORDENA cada classe contra as demais (0,5 = acaso; 1,0 = perfeito).
        # Não depende do ponto de corte, por isso compara a capacidade de separação dos modelos.
        "roc_auc_ovr_macro": roc_auc_score(y_real, proba, multi_class="ovr", average="macro",
                                           labels=ORDEM_PROBA),
        # Precisão média (área sob a curva precisão-revocação) da classe FATAL: também não depende
        # do ponto de corte. Um modelo sem informação teria o valor da prevalência (cerca de 0,07).
        "ap_fatal": average_precision_score(eh_fatal, proba[:, ORDEM_PROBA.index("FATAL")]),
        "recall_fatal": rel["FATAL"]["recall"],      # dos fatais reais, quantos foram encontrados
        "precisao_fatal": rel["FATAL"]["precision"],  # dos alertas "fatal", quantos eram mesmo fatais
        "f1_por_classe": {c: rel[c]["f1-score"] for c in CLASSES},
        "recall_por_classe": {c: rel[c]["recall"] for c in CLASSES},
        "precisao_por_classe": {c: rel[c]["precision"] for c in CLASSES},
    }


def avaliar(modelo, X, y):
    """Avalia um modelo já treinado. Devolve (métricas, matriz de confusão, relatório, previsões, proba)."""
    previsto = modelo.predict(X)
    proba = modelo.predict_proba(X)
    assert list(modelo.classes_) == ORDEM_PROBA  # garante a ordem das colunas de proba
    cm = confusion_matrix(y, previsto, labels=CLASSES)  # linhas = real, colunas = previsto
    relatorio = classification_report(y, previsto, labels=CLASSES, digits=3, zero_division=0)
    return resumir(y, previsto, proba), cm, relatorio, previsto, proba


# ----------------------------------------------------------------------------------------------
# Intervalos de confiança por bootstrap
# ----------------------------------------------------------------------------------------------
def _f1_macro_rapido(y_idx, p_idx, k=3):
    """F1 macro calculado direto da matriz de confusão (rápido para repetir milhares de vezes)."""
    cm = np.bincount(y_idx * k + p_idx, minlength=k * k).reshape(k, k)
    vp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # (VP + FP) + (VP + FN)
    f1 = np.divide(2 * vp, denom, out=np.zeros(k), where=denom > 0)
    return f1.mean()


def intervalos_bootstrap(y_real, previsoes: dict, n=1000, semente=42):
    """Intervalo de 95% do F1 macro de cada modelo e da diferença entre pares de modelos.

    Sorteia, com reposição, `n` versões do conjunto de teste (do mesmo tamanho) e recalcula o
    F1 macro em cada uma. O intervalo vai do percentil 2,5 ao 97,5. As diferenças usam as MESMAS
    amostras para os dois modelos (comparação pareada).
    """
    idx = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([idx[c] for c in y_real])
    p = {nome: np.array([idx[c] for c in prev]) for nome, prev in previsoes.items()}
    rng = np.random.default_rng(semente)
    amostras = {nome: np.empty(n) for nome in p}
    for b in range(n):
        s = rng.integers(0, len(y), len(y))
        for nome in p:
            amostras[nome][b] = _f1_macro_rapido(y[s], p[nome][s])
    linhas = []
    nomes = list(p)
    for nome in nomes:
        v = amostras[nome]
        linhas.append({"comparacao": nome, "f1_macro": _f1_macro_rapido(y, p[nome]),
                       "ic95_inf": np.percentile(v, 2.5), "ic95_sup": np.percentile(v, 97.5)})
    for i, a in enumerate(nomes):
        for b_ in nomes[i + 1:]:
            d = amostras[b_] - amostras[a]
            linhas.append({"comparacao": f"{b_} - {a}",
                           "f1_macro": _f1_macro_rapido(y, p[b_]) - _f1_macro_rapido(y, p[a]),
                           "ic95_inf": np.percentile(d, 2.5), "ic95_sup": np.percentile(d, 97.5),
                           "pct_amostras_positivas": float((d > 0).mean())})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------------------------
# Ponto de operação (regra de decisão)
# ----------------------------------------------------------------------------------------------
def prever_com_pesos(proba, pesos):
    """Classe prevista = a de maior (probabilidade x peso). Peso 1,0 em todas = decisão padrão."""
    return np.array(ORDEM_PROBA)[np.argmax(proba * np.array([pesos[c] for c in ORDEM_PROBA]), axis=1)]


def escolher_pesos(y_real, proba, grade=GRADE_PESOS):
    """Pesos (COM_FERIDOS fixo em 1) que maximizam o F1 macro nas probabilidades dadas.

    Deve receber probabilidades "fora da amostra" do TREINO (validação cruzada), nunca do teste.
    """
    padrao = {c: 1.0 for c in ORDEM_PROBA}
    melhor = (f1_score(y_real, prever_com_pesos(proba, padrao), average="macro"), padrao)
    for w_fatal in grade:
        for w_sem in grade:
            pesos = {"COM_FERIDOS": 1.0, "FATAL": w_fatal, "SEM_VITIMAS": w_sem}
            f1 = f1_score(y_real, prever_com_pesos(proba, pesos), average="macro")
            if f1 > melhor[0]:
                melhor = (f1, pesos)
    return melhor  # (F1 macro, pesos)


def curva_fatal(y_real, proba):
    """Curva precisão-revocação da classe FATAL (um contra todos), a partir de P(FATAL)."""
    eh_fatal = (np.asarray(y_real) == "FATAL").astype(int)
    precisao, revocacao, _ = precision_recall_curve(eh_fatal, proba[:, ORDEM_PROBA.index("FATAL")])
    return precisao, revocacao


def precisao_na_revocacao(y_real, proba, revocacao_alvo):
    """Maior precisão dos fatais alcançável detectando pelo menos `revocacao_alvo` dos fatais."""
    precisao, revocacao = curva_fatal(y_real, proba)
    return float(precisao[revocacao >= revocacao_alvo].max())


# ----------------------------------------------------------------------------------------------
# Importância por permutação de grupos de variáveis
# ----------------------------------------------------------------------------------------------
def importancia_por_grupos(modelo, X, y, grupos, repeticoes=3, semente=42):
    """Queda do F1 macro quando cada GRUPO de variáveis é embaralhado em conjunto.

    Ideia: se embaralhar um grupo faz o desempenho cair muito, ele era importante. Embaralhar o
    grupo inteiro de uma vez evita que variáveis parecidas (ex.: hora e fase do dia) compensem
    umas às outras e escondam a importância.
    """
    base = f1_score(y, modelo.predict(X), average="macro")  # F1 sem embaralhar nada
    rng = np.random.default_rng(semente)
    quedas = {}
    for nome_grupo, cols in grupos.items():
        valores = []
        for _ in range(repeticoes):  # repete e tira a média, pois o embaralhamento é aleatório
            embaralhada = X.copy()
            ordem = rng.permutation(len(X))
            embaralhada[cols] = X[cols].to_numpy()[ordem]  # o grupo todo é embaralhado junto
            valores.append(base - f1_score(y, modelo.predict(embaralhada), average="macro"))
        quedas[nome_grupo] = float(np.mean(valores))
    return pd.Series(quedas).sort_values(ascending=False)
