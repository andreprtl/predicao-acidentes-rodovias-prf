"""Figuras dos resultados (matrizes de confusão, curva precisão-revocação, importância e descritiva)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de janela
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_FIGURAS = RAIZ_PROJETO / "figures"

# Uma cor fixa por modelo, igual em todas as figuras (a cor identifica o modelo, não a posição)
CORES_MODELOS = {
    "Regressão Logística Multinomial": "#2a78d6",  # azul
    "Gradient Boosting": "#eb6834",                # laranja
    "Random Forest": "#1baf7a",                    # verde-água
}
COR = "#2a78d6"            # um único tom de azul quando a figura mostra só magnitude
COR_TEXTO = "#222222"
COR_SECUNDARIA = "#6b6a66"  # textos auxiliares, linhas de referência

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9893",
    "axes.labelcolor": COR_TEXTO, "xtick.color": COR_TEXTO, "ytick.color": COR_TEXTO,
    "legend.frameon": False,
})


# Números dos eixos com vírgula decimal (padrão brasileiro)
VIRGULA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def _salvar(fig, caminho):
    caminho = Path(caminho)
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


def plotar_matrizes_confusao(matrizes: dict, classes, caminho=None):
    """matrizes: {nome_do_modelo: matriz (linhas = real, colunas = previsto)}.

    Cada linha é normalizada (soma 100%), então a diagonal é a revocação (recall) da classe.
    Como ler: a célula (linha FATAL, coluna FATAL) é a % de sinistros fatais que o modelo
    acertou; as outras células da linha mostram para qual classe ele errou.
    """
    caminho = caminho or PASTA_FIGURAS / "06_matrizes_confusao.png"
    rotulos = {"SEM_VITIMAS": "Sem vítimas", "COM_FERIDOS": "Com feridos", "FATAL": "Fatal"}
    nomes_eixo = [rotulos.get(c, c) for c in classes]
    fig, eixos = plt.subplots(1, len(matrizes), figsize=(4.6 * len(matrizes), 4.1), squeeze=False)
    for k, (ax, (nome, cm)) in enumerate(zip(eixos[0], matrizes.items())):
        pct = cm / cm.sum(axis=1, keepdims=True) * 100  # % por linha (classe real)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(classes)):
            for j in range(len(classes)):
                cor = "white" if pct[i, j] > 55 else COR_TEXTO  # contraste com o fundo
                contagem = f"{cm[i, j]:,}".replace(",", ".")  # separador de milhar com ponto
                ax.text(j, i, f"{pct[i, j]:.1f}%".replace(".", ",") + f"\n({contagem})",
                        ha="center", va="center", color=cor, fontsize=9)
        ax.set_xticks(range(len(classes)), nomes_eixo)
        ax.set_yticks(range(len(classes)), nomes_eixo if k == 0 else [""] * len(classes))
        ax.set_xlabel("Classe prevista")
        if k == 0:
            ax.set_ylabel("Classe real")
        ax.set_title(nome)
        ax.spines[:].set_visible(False)
    fig.tight_layout()
    return _salvar(fig, caminho)


def plotar_curvas_fatal(curvas: dict, pontos: dict, prevalencia, caminho=None):
    """Curva precisão-revocação da classe FATAL para cada modelo.

    curvas: {modelo: (precisao, revocacao)}; pontos: {modelo: (revocacao, precisao)} na decisão
    padrão (classe de maior probabilidade). Quanto mais alta a curva, melhor o modelo separa os
    fatais; o ponto marca onde cada modelo opera sem ajuste da regra de decisão.
    """
    caminho = caminho or PASTA_FIGURAS / "10_curva_precisao_revocacao_fatal.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for nome, (prec, rev) in curvas.items():
        cor = CORES_MODELOS.get(nome, COR)
        ax.plot(rev, prec, color=cor, lw=2, label=nome)
        r, p = pontos[nome]
        ax.scatter([r], [p], s=60, color=cor, edgecolor="white", linewidth=2, zorder=5)
    ax.axhline(prevalencia, color=COR_SECUNDARIA, lw=1, ls="--")
    ax.text(0.01, prevalencia - 0.006, f"taxa de base ({prevalencia:.1%})".replace(".", ","),
            ha="left", va="top", color=COR_SECUNDARIA, fontsize=9)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.yaxis.set_major_formatter(VIRGULA)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Revocação dos fatais (fração dos fatais detectados)")
    ax.set_ylabel("Precisão dos fatais")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", title="● = decisão padrão de cada modelo", title_fontsize=9)
    ax.set_title("Curva precisão-revocação da classe Fatal (conjunto de teste)")
    return _salvar(fig, caminho)


def plotar_importancia_modelos(tabela, caminho=None, rotular=2):
    """Importância por permutação de grupos (queda do F1 macro) para vários modelos.

    tabela: DataFrame com uma linha por grupo e uma coluna por modelo. Os grupos ficam na ordem
    da última coluna (Random Forest) e só os `rotular` maiores recebem o valor escrito.
    """
    caminho = caminho or PASTA_FIGURAS / "07_importancia_grupos.png"
    tabela = tabela.sort_values(tabela.columns[-1], ascending=True)
    modelos = list(tabela.columns)
    n, m = len(tabela), len(modelos)
    altura = 0.8 / m
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * n + 1.4))
    y = np.arange(n)
    for k, nome in enumerate(modelos):
        pos = y + (k - (m - 1) / 2) * altura
        ax.barh(pos, tabela[nome].values, height=altura * 0.85, color=CORES_MODELOS.get(nome, COR),
                label=nome)
        for i in range(n - rotular, n):  # rótulos só nos grupos mais importantes
            v = tabela[nome].values[i]
            ax.text(v + 0.0006, pos[i], f"{v:.3f}".replace(".", ","), va="center", fontsize=8,
                    color=COR_TEXTO)
    ax.set_yticks(y, tabela.index)
    ax.axvline(0, color="#9a9893", lw=0.8)
    ax.set_xlabel("Queda do F1 macro ao embaralhar o grupo")
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right")
    ax.set_title("Importância por permutação de grupos de variáveis")
    return _salvar(fig, caminho)


def plotar_importancia(tabela, coluna, titulo, caminho, topo=15):
    """Barras horizontais das variáveis mais importantes de um único modelo."""
    dados = tabela[coluna].sort_values(ascending=False).head(topo)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(dados) + 1.2))
    ax.barh(dados.index, dados.values, color=COR, height=0.6)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.set_title(titulo)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    return _salvar(fig, caminho)


def plotar_letalidade(por_hora, por_pista, por_uf, media, caminho=None):
    """Três painéis com a % de sinistros fatais por hora, tipo de pista e UF (análise descritiva)."""
    caminho = caminho or PASTA_FIGURAS / "09_letalidade_por_condicao.png"
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [2.2, 1, 1.6]},
                              sharey=True)
    ref = dict(color=COR_SECUNDARIA, lw=1, ls="--")
    ax = eixos[0]
    ax.plot(por_hora.index, por_hora.values, color=COR, lw=2, marker="o", ms=3)
    ax.axhline(media, **ref)
    ax.axvspan(-0.5, 5.5, color="#cde2fb", alpha=0.5, lw=0)
    ax.text(2.5, 0.6, "madrugada\n(0h–5h59)", ha="center", va="bottom", fontsize=9, color=COR_TEXTO)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("% de sinistros fatais")
    ax.set_title("Por hora do dia")
    for ax, serie, titulo in [(eixos[1], por_pista, "Por tipo de pista"), (eixos[2], por_uf, "UFs com maior % e Brasil")]:
        cores = [COR_SECUNDARIA if r == "Brasil" else COR for r in serie.index]
        ax.bar(serie.index, serie.values, color=cores, width=0.6)
        ax.axhline(media, **ref)
        for i, v in enumerate(serie.values):
            ax.text(i, v + 0.3, f"{v:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=8)
        ax.set_title(titulo)
    eixos[0].text(23.4, media + 0.25, f"média {media:.1f}%".replace(".", ","), ha="right",
                  va="bottom", fontsize=8, color=COR_SECUNDARIA)
    for ax in eixos:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return _salvar(fig, caminho)
