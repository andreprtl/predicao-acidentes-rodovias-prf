"""Análise descritiva enxuta: proporção de sinistros fatais por condição.

Gera os números da caracterização da base no artigo (tabela de letalidade) e uma figura com
três painéis (hora do dia, tipo de pista e UF). A análise exploratória completa, com os
gráficos 01 a 05, continua no notebook 02_analise_exploratoria_eda.ipynb.

Uso (a partir da raiz do projeto):
    python -m src.descritiva

Saídas: results/descritiva_letalidade.csv e figures/09_letalidade_por_condicao.png
"""

import pandas as pd

from src.data_loader import RAIZ_PROJETO, carregar_base
from src.visualization import plotar_letalidade

PASTA_RESULTADOS = RAIZ_PROJETO / "results"


def pct_fatal(serie_alvo):
    """% de sinistros fatais em um conjunto de sinistros."""
    return float((serie_alvo == "FATAL").mean() * 100)


def main():
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    df = carregar_base()
    alvo = df["alvo_severidade"]
    media = pct_fatal(alvo)
    linhas = [("Total", "Todos os sinistros", len(df), media)]

    # Horário: madrugada = das 0h às 5h59
    madrugada = df["hora"].between(0, 5)
    linhas += [("Horário", "Madrugada (0h a 5h59)", int(madrugada.sum()), pct_fatal(alvo[madrugada])),
               ("Horário", "Demais horários", int((~madrugada).sum()), pct_fatal(alvo[~madrugada]))]

    # Tipo de pista
    for pista in ["SIMPLES", "DUPLA", "MÚLTIPLA"]:
        m = df["tipo_pista"] == pista
        linhas.append(("Tipo de pista", pista.capitalize(), int(m.sum()), pct_fatal(alvo[m])))

    # Traçado: combinações mais frequentes, como aparecem no registro (ex.: "RETA;DECLIVE")
    tracado = df["tracado_via"].fillna("")
    for valor, rotulo in [("RETA", "Reta"), ("CURVA", "Curva"), ("RETA;DECLIVE", "Reta em declive"),
                          ("RETA;ACLIVE", "Reta em aclive"), ("INTERSEÇÃO DE VIAS", "Interseção de vias")]:
        m = tracado == valor
        linhas.append(("Traçado", rotulo, int(m.sum()), pct_fatal(alvo[m])))

    # UF: as cinco com maior índice médio de severidade (pesos 1, 5 e 13; só descritivo)
    por_uf = df.groupby("uf").agg(n=("alvo_severidade", "size"),
                                  pct_fatal=("alvo_severidade", pct_fatal),
                                  indice_medio=("indice_severidade", "mean"),
                                  pct_pista_simples=("tipo_pista", lambda s: (s == "SIMPLES").mean() * 100))
    top = por_uf.sort_values("indice_medio", ascending=False).head(5)
    for uf, r in top.iterrows():
        linhas.append(("UF", f"{uf} (índice médio {r['indice_medio']:.2f}; pista simples "
                             f"{r['pct_pista_simples']:.1f}%)", int(r["n"]), r["pct_fatal"]))
    linhas.append(("UF", f"Brasil (pista simples {(df['tipo_pista'] == 'SIMPLES').mean() * 100:.1f}%)",
                   len(df), media))

    tabela = pd.DataFrame(linhas, columns=["grupo", "condicao", "n_sinistros", "pct_fatal"])
    tabela.to_csv(PASTA_RESULTADOS / "descritiva_letalidade.csv", index=False, encoding="utf-8")
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # Figura com três painéis
    hora = df.groupby("hora")["alvo_severidade"].apply(pct_fatal)
    pista = pd.Series({p.capitalize(): pct_fatal(alvo[df["tipo_pista"] == p])
                       for p in ["SIMPLES", "DUPLA", "MÚLTIPLA"]})
    ufs = por_uf.sort_values("pct_fatal", ascending=False).head(5)["pct_fatal"]
    ufs["Brasil"] = media
    plotar_letalidade(hora, pista, ufs, media)
    print("Pronto. Resultados em results/ e figures/.")


if __name__ == "__main__":
    main()
