"""Experimentos complementares ao treinamento principal (src/train.py).

Testam a robustez da comparação entre os modelos, sem alterar os números principais:

    1. temporal : treina com 2023-2024 e testa em 2025 (dados "do futuro"), em vez de misturar os anos.
    2. espacial : em cada rodada, estados inteiros ficam de fora do treino (validação "por grupos de UF").
    3. ajuste   : busca de hiperparâmetros (Random Forest, Gradient Boosting e Regressão Logística)
                  com validação cruzada SOMENTE nos dados de treino; o teste só avalia o modelo final.
    4. decisao  : ajusta a regra de decisão de cada modelo (pesos por classe aplicados às
                  probabilidades), escolhida com previsões "fora da amostra" do treino. Compara os
                  modelos no ponto de operação mais favorável de cada um.
    5. ablacao  : retreina os modelos (a) sem as variáveis de localização e (b) acrescentando a
                  causa e o tipo do acidente, para medir quanto o desempenho depende da informação
                  disponível, e não do algoritmo.

O conjunto de teste (20%) nunca é usado para escolher nada: só mede o resultado final de cada etapa.

Uso (a partir da raiz do projeto):
    python -m src.experimentos                          # todas as etapas (cerca de 40 minutos)
    python -m src.experimentos --etapas temporal ajuste # só as etapas escolhidas
    python -m src.experimentos --n-candidatos 4         # busca menor, para testar mais rápido

Saídas em results/: validacao_temporal.csv, validacao_espacial.csv, validacao_espacial_ufs.txt,
ajuste_hiperparametros.csv, ajuste_hiperparametros.json, modelos_ajustados.csv, decisao.csv,
decisao_curva.csv, decisao_bootstrap.csv e ablacao.csv.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, ParameterSampler, StratifiedKFold,
                                     cross_val_predict, cross_validate)

from src.avaliacao import avaliar, escolher_pesos, intervalos_bootstrap, prever_com_pesos, resumir
from src.data_loader import RAIZ_PROJETO, carregar_base
from src.modelos import NOMES, ORDEM, PARAMS_RF, SEMENTE, montar_modelo
from src.preprocessing import criar_features, dividir_treino_teste

PASTA_RESULTADOS = RAIZ_PROJETO / "results"
MODELOS_TREINADOS = ["lr", "hgb", "rf"]  # todos menos o baseline

# Valores testados na busca de hiperparâmetros do Random Forest. Evitamos árvores sem limite de
# profundidade e folhas de 1 sinistro, que gerariam modelos enormes (memória) e tenderiam a decorar.
GRADE_RF = {
    "max_depth": [10, 15, 20, 30],
    "min_samples_leaf": [2, 5, 10, 20],
    "max_features": ["sqrt", 0.3, 0.5],  # quantas variáveis cada árvore examina em cada divisão
    "class_weight": ["balanced", "balanced_subsample"],
}
# Gradient Boosting: ritmo de aprendizado, tamanho das árvores, folha mínima e regularização
GRADE_HGB = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 100, 400],
    "l2_regularization": [0.0, 1.0],
    "max_iter": [100, 300],
}
# Intensidade da regularização da Regressão Logística (C menor = modelo mais "contido")
GRADE_C = [0.01, 0.1, 1.0, 10.0]


def metricas_curtas(m):
    """Só as métricas escalares (para tabelas CSV)."""
    return {k: v for k, v in m.items() if not isinstance(v, dict)}


def salvar_csv(linhas, nome):
    pd.DataFrame(linhas).to_csv(PASTA_RESULTADOS / nome, index=False, encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Etapa 1: validação temporal
# ----------------------------------------------------------------------------------------------
def etapa_temporal(X, y, anos, cat, num):
    """Treina com 2023-2024 e testa em 2025: os sinistros do teste acontecem DEPOIS dos do treino."""
    print("\n[temporal] treino 2023-2024 -> teste 2025", flush=True)
    tr, te = (anos <= 2024).to_numpy(), (anos == 2025).to_numpy()
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    print(f"  treino: {len(X_tr):,} | teste: {len(X_te):,}".replace(",", "."))
    linhas = []
    for tipo in ORDEM:
        t0 = time.time()
        modelo = montar_modelo(tipo, cat, num).fit(X_tr, y_tr)
        met = metricas_curtas(avaliar(modelo, X_te, y_te)[0])
        linhas.append({"modelo": NOMES[tipo], "n_treino": len(X_tr), "n_teste": len(X_te),
                       "pct_fatal_teste": float((y_te == "FATAL").mean()), **met,
                       "segundos": round(time.time() - t0, 1)})
        print(f"  {NOMES[tipo]:34s} F1 macro={met['f1_macro']:.3f}  AUC={met['roc_auc_ovr_macro']:.3f}  "
              f"revocação fatal={met['recall_fatal']:.3f}", flush=True)
    salvar_csv(linhas, "validacao_temporal.csv")


# ----------------------------------------------------------------------------------------------
# Etapa 2: validação espacial
# ----------------------------------------------------------------------------------------------
def etapa_espacial(X, y, uf, cat, num, partes=5):
    """Validação por grupos de UF: em cada rodada, alguns estados inteiros ficam fora do treino.

    O modelo precisa prever sinistros de estados que nunca viu (a UF e a BR dessas linhas são
    categorias desconhecidas e só a latitude e a longitude ainda dão pistas do local).
    """
    print(f"\n[espacial] {partes} rodadas, estados inteiros fora do treino", flush=True)
    cv = GroupKFold(n_splits=partes)
    acumulado = {t: [] for t in ORDEM}
    ufs_fora = []
    for k, (tr, te) in enumerate(cv.split(X, y, groups=uf), 1):
        ufs_fora.append("/".join(sorted(uf.iloc[te].unique())))
        for tipo in ORDEM:
            modelo = montar_modelo(tipo, cat, num).fit(X.iloc[tr], y.iloc[tr])
            acumulado[tipo].append(metricas_curtas(avaliar(modelo, X.iloc[te], y.iloc[te])[0]))
        print(f"  rodada {k}: {len(te):,} sinistros fora do treino | ".replace(",", ".") +
              " | ".join(f"{t} F1={a[-1]['f1_macro']:.3f}" for t, a in acumulado.items()), flush=True)
    linhas = []
    for tipo, lista in acumulado.items():
        f1s = np.array([m["f1_macro"] for m in lista])
        linhas.append({
            "modelo": NOMES[tipo], "partes": partes,
            "f1_macro_media": f1s.mean(), "f1_macro_desvio": f1s.std(ddof=1),
            "acuracia_media": np.mean([m["acuracia"] for m in lista]),
            "roc_auc_ovr_macro_media": np.mean([m["roc_auc_ovr_macro"] for m in lista]),
            "ap_fatal_media": np.mean([m["ap_fatal"] for m in lista]),
            "recall_fatal_media": np.mean([m["recall_fatal"] for m in lista]),
            "f1_macro_por_parte": ";".join(f"{v:.4f}" for v in f1s),
        })
        print(f"  {NOMES[tipo]:34s} F1 macro = {f1s.mean():.3f} ± {f1s.std(ddof=1):.3f}")
    salvar_csv(linhas, "validacao_espacial.csv")
    (PASTA_RESULTADOS / "validacao_espacial_ufs.txt").write_text(
        "\n".join(f"rodada {i}: {u}" for i, u in enumerate(ufs_fora, 1)), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Etapa 3: ajuste de hiperparâmetros
# ----------------------------------------------------------------------------------------------
def buscar(tipo, candidatos, X_tr, y_tr, cat, num, cv):
    """Avalia cada candidato por validação cruzada no treino (uma linha por candidato)."""
    linhas = []
    for i, params in enumerate(candidatos, 1):
        t0 = time.time()
        res = cross_validate(montar_modelo(tipo, cat, num, **params), X_tr, y_tr, cv=cv,
                             scoring="f1_macro", n_jobs=1)
        v = res["test_score"]
        linhas.append({"modelo": NOMES[tipo], "candidato": i, "parametros": json.dumps(params),
                       "f1_macro_media": float(v.mean()), "f1_macro_desvio": float(v.std(ddof=1)),
                       "segundos": round(time.time() - t0, 1)})
        print(f"  [{tipo}] {i:2d}/{len(candidatos)} F1 macro = {v.mean():.4f} ± {v.std(ddof=1):.4f}  {params}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    return linhas


def etapa_ajuste(X_tr, y_tr, X_te, y_te, cat, num, n_candidatos):
    """Busca aleatória de hiperparâmetros (3 partes, só no treino) e avaliação única no teste.

    O candidato "padrão" (os valores fixados a priori) é sempre o primeiro da busca: assim a
    comparação com ele usa exatamente as mesmas partes da validação cruzada.
    """
    print("\n[ajuste] busca de hiperparâmetros (validação cruzada de 3 partes no treino)", flush=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMENTE)
    padroes = {
        "rf": {"max_depth": PARAMS_RF["max_depth"], "min_samples_leaf": PARAMS_RF["min_samples_leaf"],
               "max_features": "sqrt", "class_weight": PARAMS_RF["class_weight"]},
        "hgb": {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                "l2_regularization": 0.0, "max_iter": 100},
        "lr": {"C": 1.0},
    }
    candidatos = {
        "rf": [p for p in ParameterSampler(GRADE_RF, n_iter=n_candidatos, random_state=SEMENTE)
               if p != padroes["rf"]],
        "hgb": [p for p in ParameterSampler(GRADE_HGB, n_iter=max(1, n_candidatos - 1), random_state=SEMENTE)
                if p != padroes["hgb"]],
        "lr": [{"C": c} for c in GRADE_C if c != 1.0],
    }
    linhas = []
    for tipo in ["rf", "hgb", "lr"]:
        linhas += buscar(tipo, [padroes[tipo]] + candidatos[tipo], X_tr, y_tr, cat, num, cv)
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(PASTA_RESULTADOS / "ajuste_hiperparametros.csv", index=False, encoding="utf-8")

    # Melhor candidato de cada modelo = maior F1 macro médio na validação cruzada
    escolhidos, resultado = {}, []
    for tipo in ["rf", "hgb", "lr"]:
        sub = tabela[tabela["modelo"] == NOMES[tipo]]
        melhor, ref = sub.loc[sub["f1_macro_media"].idxmax()], sub.iloc[0]
        escolhidos[tipo] = json.loads(melhor["parametros"])
        print(f"  melhor {NOMES[tipo]}: {escolhidos[tipo]} (CV {melhor['f1_macro_media']:.4f}; "
              f"padrão: {ref['f1_macro_media']:.4f})")
        # Avaliação única no teste: o modelo "a priori" e o ajustado, lado a lado
        for rotulo, params, f1_cv in [("a priori", padroes[tipo], ref["f1_macro_media"]),
                                      ("ajustado", escolhidos[tipo], melhor["f1_macro_media"])]:
            modelo = montar_modelo(tipo, cat, num, **params).fit(X_tr, y_tr)
            met = metricas_curtas(avaliar(modelo, X_te, y_te)[0])
            resultado.append({"modelo": NOMES[tipo], "versao": rotulo, "parametros": json.dumps(params),
                              "f1_macro_cv3": f1_cv, **met})
            print(f"    {rotulo:9s} no teste: F1 macro={met['f1_macro']:.4f}  AUC={met['roc_auc_ovr_macro']:.4f}  "
                  f"revocação fatal={met['recall_fatal']:.3f}  precisão fatal={met['precisao_fatal']:.3f}",
                  flush=True)
    salvar_csv(resultado, "modelos_ajustados.csv")
    (PASTA_RESULTADOS / "ajuste_hiperparametros.json").write_text(
        json.dumps(escolhidos, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Etapa 4: ajuste da regra de decisão (ponto de operação) de cada modelo
# ----------------------------------------------------------------------------------------------
def etapa_decisao(X_tr, y_tr, X_te, y_te, cat, num):
    """Compara os modelos no ponto de operação mais favorável de cada um.

    Com class_weight="balanced", cada modelo "decide" a classe com um viés diferente (uns arriscam
    mais a classe FATAL que outros). Isso muda o F1 macro mesmo que a capacidade de separar as
    classes seja parecida. Aqui, para cada modelo, os pesos por classe que maximizam o F1 macro são
    escolhidos nas previsões "fora da amostra" do TREINO (validação cruzada de 5 partes) e só
    depois aplicados uma vez ao teste.
    """
    print("\n[decisao] pesos por classe escolhidos fora da amostra do treino, para cada modelo", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    linhas, curva, previsoes_ajustadas = [], [], {}
    for tipo in MODELOS_TREINADOS:
        t0 = time.time()
        modelo = montar_modelo(tipo, cat, num)
        # Probabilidades de cada sinistro do treino, previstas por modelos que NÃO o viram no treino
        proba_oof = cross_val_predict(modelo, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)
        f1_oof, pesos = escolher_pesos(y_tr, proba_oof)
        # Modelo final treinado em todo o treino e avaliado uma única vez no teste
        modelo.fit(X_tr, y_tr)
        proba_te = modelo.predict_proba(X_te)
        padrao = metricas_curtas(resumir(y_te, modelo.predict(X_te), proba_te))
        previsoes_ajustadas[NOMES[tipo]] = prever_com_pesos(proba_te, pesos)
        ajustado = metricas_curtas(resumir(y_te, previsoes_ajustadas[NOMES[tipo]], proba_te))
        for rotulo, met in [("decisão padrão", padrao), ("pesos escolhidos no treino", ajustado)]:
            linhas.append({"modelo": NOMES[tipo], "regra": rotulo, "pesos": json.dumps(pesos),
                           "f1_macro_fora_da_amostra": f1_oof, **met})
        print(f"  {NOMES[tipo]:34s} pesos={pesos}  teste: F1 macro {padrao['f1_macro']:.4f} -> "
              f"{ajustado['f1_macro']:.4f} | revocação fatal {ajustado['recall_fatal']:.3f} | "
              f"precisão fatal {ajustado['precisao_fatal']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        # Curva de ponto de operação (só descritiva, nada é escolhido com ela): quanto mais peso
        # damos à classe FATAL, mais fatais o modelo encontra, com mais alertas falsos.
        for w_fatal in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            p = {"COM_FERIDOS": 1.0, "FATAL": w_fatal, "SEM_VITIMAS": 1.0}
            met = metricas_curtas(resumir(y_te, prever_com_pesos(proba_te, p), proba_te))
            curva.append({"modelo": NOMES[tipo], "peso_fatal": w_fatal, **met})
    salvar_csv(linhas, "decisao.csv")
    salvar_csv(curva, "decisao_curva.csv")
    # Intervalo de 95% do F1 macro com a regra ajustada (e das diferenças entre os modelos)
    ic = intervalos_bootstrap(y_te, previsoes_ajustadas, semente=SEMENTE)
    ic.to_csv(PASTA_RESULTADOS / "decisao_bootstrap.csv", index=False, encoding="utf-8")
    print(ic.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ----------------------------------------------------------------------------------------------
# Etapa 5: ablação (quanto o desempenho depende da informação disponível)
# ----------------------------------------------------------------------------------------------
def etapa_ablacao(df, cat, num):
    """Mesma divisão 80/20, três conjuntos de variáveis: completo, sem localização e com causa/tipo.

    - Sem localização: retira latitude, longitude, UF e BR. Mostra quanto os modelos dependem de
      "conhecer o lugar" (ligado ao resultado da validação espacial).
    - Com causa e tipo do acidente: acrescenta duas variáveis que descrevem a dinâmica do sinistro
      (ex.: colisão frontal, atropelamento). Elas só são conhecidas DEPOIS do sinistro, por isso não
      entram no modelo principal; servem para estimar quanto uma informação mais próxima da
      gravidade melhora a previsão.
    """
    print("\n[ablacao] completo x sem localização x com causa e tipo do acidente", flush=True)
    localizacao = ["latitude", "longitude", "uf", "br"]
    X, y, _, _ = criar_features(df)
    X_ct, _, cat_ct, num_ct = criar_features(df, incluir_causa_tipo=True)
    configuracoes = {
        "completo (26 atributos)": (X, cat, num),
        "sem localização": (X, [c for c in cat if c not in localizacao], [c for c in num if c not in localizacao]),
        "com causa e tipo do acidente": (X_ct, cat_ct, num_ct),
    }
    linhas = []
    for rotulo, (Xc, cat_c, num_c) in configuracoes.items():
        X_tr, X_te, y_tr, y_te = dividir_treino_teste(Xc[cat_c + num_c], y, teste=0.2, semente=SEMENTE)
        for tipo in MODELOS_TREINADOS:
            modelo = montar_modelo(tipo, cat_c, num_c).fit(X_tr, y_tr)
            met = metricas_curtas(avaliar(modelo, X_te, y_te)[0])
            linhas.append({"configuracao": rotulo, "n_atributos": len(cat_c) + len(num_c),
                           "modelo": NOMES[tipo], **met})
            print(f"  {rotulo:30s} {NOMES[tipo]:34s} F1 macro={met['f1_macro']:.4f}  "
                  f"AUC={met['roc_auc_ovr_macro']:.4f}  AP fatal={met['ap_fatal']:.4f}", flush=True)
    salvar_csv(linhas, "ablacao.csv")


# ----------------------------------------------------------------------------------------------
def main():
    etapas = ["temporal", "espacial", "ajuste", "decisao", "ablacao"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etapas", nargs="+", default=etapas, choices=etapas, help="etapas a executar")
    ap.add_argument("--n-candidatos", type=int, default=9,
                    help="combinações sorteadas na busca de hiperparâmetros (além do padrão)")
    args = ap.parse_args()
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    print("Carregando a base...")
    df = carregar_base()
    X, y, cat, num = criar_features(df)
    anos, uf = df["ano"], df["uf"]  # a base tem o ano de cada sinistro; ele NÃO é variável de entrada
    # A mesma divisão 80/20 (mesma semente) do treinamento principal, para os números serem comparáveis
    X_tr, X_te, y_tr, y_te = dividir_treino_teste(X, y, teste=0.2, semente=SEMENTE)

    if "temporal" in args.etapas:
        etapa_temporal(X, y, anos, cat, num)
    if "espacial" in args.etapas:
        etapa_espacial(X, y, uf, cat, num)
    if "ajuste" in args.etapas:
        etapa_ajuste(X_tr, y_tr, X_te, y_te, cat, num, args.n_candidatos)
    if "decisao" in args.etapas:
        etapa_decisao(X_tr, y_tr, X_te, y_te, cat, num)
    if "ablacao" in args.etapas:
        etapa_ablacao(df, cat, num)
    print("\nPronto. Resultados em results/.")


if __name__ == "__main__":
    main()
