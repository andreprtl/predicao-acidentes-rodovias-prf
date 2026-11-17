"""Treina e avalia os modelos descritos na metodologia do artigo.

O que este script faz, em ordem:
    1. Carrega a base tratada da PRF e monta as variáveis de entrada (sem vazamento de dados).
    2. Divide os dados em treino (80%) e teste (20%), mantendo a proporção das classes.
    3. Treina quatro modelos e compara o desempenho de cada um no teste, com intervalo de
       confiança de 95% (bootstrap) para o F1 macro e para as diferenças entre modelos.
    4. Repete a comparação com validação cruzada de 5 partes (só nos dados de treino).
    5. Mede quais grupos de variáveis mais pesam na previsão de cada modelo.
    6. Salva métricas, tabelas, figuras e o modelo treinado.

Uso (a partir da raiz do projeto):
    python -m src.train                       # configuração do artigo (cerca de 10 minutos)
    python -m src.train --folds 0             # pula a validação cruzada (mais rápido)
    python -m src.train --n-arvores 100       # Random Forest menor, para testar
    python -m src.train --incluir-causa-tipo  # experimento extra com causa e tipo do acidente

Experimentos complementares (validação temporal e espacial, ajuste de hiperparâmetros, ajuste
da regra de decisão e ablação) ficam em src/experimentos.py e não alteram os números daqui.

Modelos comparados (mesmos dados, mesma divisão 80/20; definidos em src/modelos.py):
    0. Baseline: sempre prevê a classe mais comum (referência mínima)
    1. Regressão Logística Multinomial (class_weight="balanced")
    2. Gradient Boosting (HistGradientBoosting, class_weight="balanced")
    3. Random Forest Classifier (class_weight="balanced")
"""

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

# Funções dos outros arquivos da pasta src/
from src.avaliacao import (avaliar, curva_fatal, importancia_por_grupos, intervalos_bootstrap,
                           precisao_na_revocacao)
from src.data_loader import RAIZ_PROJETO, carregar_base
from src.modelos import NOMES, ORDEM, PARAMS_RF, SEMENTE, montar_modelo  # noqa: F401 (PARAMS_RF reexportado)
from src.preprocessing import CLASSES, criar_features, dividir_treino_teste, grupos_de_variaveis
from src.visualization import (PASTA_FIGURAS, plotar_curvas_fatal, plotar_importancia,
                               plotar_importancia_modelos, plotar_matrizes_confusao)

PASTA_RESULTADOS = RAIZ_PROJETO / "results"
PASTA_SAIDAS = RAIZ_PROJETO / "data" / "outputs"  # ignorada pelo git (arquivos grandes)


def validar_cruzado(modelos, X_treino, y_treino, folds):
    """Validação cruzada estratificada: mede o quanto o resultado varia entre divisões diferentes.

    Os dados de TREINO são divididos em `folds` partes (mantendo a proporção das classes).
    Cada parte serve uma vez de "teste" enquanto o modelo é treinado nas demais. Assim vemos a
    média e o desvio padrão de cada métrica, e não só o valor de uma única divisão.
    O conjunto de teste (20%) não é usado aqui: continua reservado para a avaliação final.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEMENTE)
    # nome que usamos -> nome da métrica no scikit-learn
    metricas = {"f1_macro": "f1_macro", "acuracia": "accuracy", "roc_auc_ovr_macro": "roc_auc_ovr"}
    linhas = []
    for nome, modelo in modelos.items():
        t0 = time.time()
        # cross_validate treina uma cópia nova do modelo em cada divisão
        res = cross_validate(modelo, X_treino, y_treino, cv=cv, scoring=metricas, n_jobs=1)
        linha = {"modelo": nome, "folds": folds}
        for chave in metricas:
            valores = res["test_" + chave]
            linha[f"{chave}_media"] = float(valores.mean())
            linha[f"{chave}_desvio"] = float(valores.std(ddof=1))  # desvio padrão amostral
        # F1 macro de cada parte, para ver se a ordem entre os modelos se mantém em todas elas
        linha["f1_macro_por_parte"] = ";".join(f"{v:.4f}" for v in res["test_f1_macro"])
        linha["segundos"] = round(time.time() - t0, 1)
        linhas.append(linha)
        print(f"  {nome:34s} F1 macro = {linha['f1_macro_media']:.3f} ± {linha['f1_macro_desvio']:.3f}  "
              f"| AUC = {linha['roc_auc_ovr_macro_media']:.3f} ± {linha['roc_auc_ovr_macro_desvio']:.3f}",
              flush=True)
    return pd.DataFrame(linhas)


def main():
    # --- opções de linha de comando (todas opcionais) ---
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incluir-causa-tipo", action="store_true",
                    help="inclui causa_acidente e tipo_acidente como variáveis (fora do modelo principal)")
    ap.add_argument("--n-arvores", type=int, default=200, help="árvores do Random Forest")
    ap.add_argument("--amostra-permutacao", type=int, default=20000,
                    help="linhas do teste usadas na importância por permutação")
    ap.add_argument("--folds", type=int, default=5,
                    help="partes da validação cruzada nos dados de treino (0 = não executa)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="reamostragens do intervalo de confiança")
    args = ap.parse_args()

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    PASTA_SAIDAS.mkdir(parents=True, exist_ok=True)
    # No experimento extra, os arquivos ganham um sufixo para não sobrescrever os do artigo
    sufixo = "_com_causa_tipo" if args.incluir_causa_tipo else ""

    # --- 1. dados ---
    print("Carregando a base...")
    df = carregar_base()
    # X = variáveis de entrada, y = gravidade (o que queremos prever)
    X, y, cat, num = criar_features(df, incluir_causa_tipo=args.incluir_causa_tipo)
    # Treino (80%): o modelo aprende. Teste (20%): dados que ele nunca viu, para avaliar de verdade.
    X_tr, X_te, y_tr, y_te = dividir_treino_teste(X, y, teste=0.2, semente=SEMENTE)
    print(f"Variáveis: {X.shape[1]} | treino: {len(X_tr):,} | teste: {len(X_te):,}".replace(",", "."))

    # --- 2. modelos (definidos em src/modelos.py) ---
    modelos = {NOMES[t]: montar_modelo(t, cat, num, n_arvores=args.n_arvores) for t in ORDEM}

    # --- 3. treino e avaliação de cada modelo no teste ---
    todas, matrizes, relatorios, previsoes, probas = [], {}, [], {}, {}
    for nome, modelo in modelos.items():
        t0 = time.time()
        modelo.fit(X_tr, y_tr)  # treino: o modelo aprende com os dados de treino
        met, cm, rel, prev, proba = avaliar(modelo, X_te, y_te)  # avaliação nos dados de teste
        todas.append({"modelo": nome, **met, "segundos_treino_e_avaliacao": round(time.time() - t0, 1)})
        relatorios.append(f"=== {nome} ===\n{rel}")
        previsoes[nome], probas[nome] = prev, proba
        if nome != NOMES["baseline"]:  # o baseline só preveria uma classe
            matrizes[nome] = cm
        print(f"{nome:34s} acurácia={met['acuracia']:.3f}  F1 macro={met['f1_macro']:.3f}  "
              f"AUC={met['roc_auc_ovr_macro']:.3f}  AP fatal={met['ap_fatal']:.3f}  "
              f"revocação fatal={met['recall_fatal']:.3f}  precisão fatal={met['precisao_fatal']:.3f}",
              flush=True)

    # Intervalo de confiança de 95% do F1 macro (e das diferenças entre modelos) por bootstrap
    print(f"Bootstrap ({args.bootstrap} reamostragens do teste)...")
    ic = intervalos_bootstrap(y_te, previsoes, n=args.bootstrap, semente=SEMENTE)
    print(ic.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Precisão dos fatais em níveis fixos de revocação (compara os modelos no MESMO ponto)
    nao_baseline = [n for n in modelos if n != NOMES["baseline"]]
    tabela_rev = pd.DataFrame({
        nome: {f"precisao_com_revocacao_{int(r * 100)}": precisao_na_revocacao(y_te, probas[nome], r)
               for r in (0.3, 0.5, 0.7)} for nome in nao_baseline}).T
    print(tabela_rev.to_string(float_format=lambda v: f"{v:.3f}"))

    # --- 4. validação cruzada (robustez da comparação entre os modelos) ---
    # Roda depois da avaliação no teste e só usa o treino, então não altera os números acima.
    cv_tabela = None
    if args.folds >= 2:
        print(f"Validação cruzada estratificada ({args.folds} partes) nos dados de treino...")
        cv_tabela = validar_cruzado({n: montar_modelo(t, cat, num, n_arvores=args.n_arvores)
                                     for t, n in NOMES.items()}, X_tr, y_tr, args.folds)

    # --- 5. importância das variáveis ---
    rf = modelos[NOMES["rf"]]
    # (a) Feature Importance clássica do Random Forest (por impureza), uma linha por variável.
    #     Atenção: tende a favorecer variáveis contínuas ou com muitas categorias (lat, lon, mês).
    nomes_colunas = list(rf.named_steps["pre"].get_feature_names_out())
    tabela_impureza = pd.DataFrame(
        {"importancia_impureza": rf.named_steps["clf"].feature_importances_}, index=nomes_colunas
    ).sort_values("importancia_impureza", ascending=False)

    # (b) Importância por permutação, por GRUPO de variáveis, para cada modelo: quanto o F1 macro
    #     cai no teste quando o grupo inteiro é embaralhado. Usa uma amostra do teste (mais rápido).
    print("Calculando importância por permutação de grupos...")
    grupos = grupos_de_variaveis(list(X.columns))
    amostra = X_te.sample(min(args.amostra_permutacao, len(X_te)), random_state=SEMENTE)
    y_amostra = y_te.loc[amostra.index]
    ordem_imp = [NOMES["lr"], NOMES["hgb"], NOMES["rf"]]
    tabela_grupos = pd.DataFrame({nome: importancia_por_grupos(modelos[nome], amostra, y_amostra, grupos,
                                                               semente=SEMENTE) for nome in ordem_imp})
    tabela_grupos = tabela_grupos.sort_values(NOMES["rf"], ascending=False)
    print(tabela_grupos.to_string(float_format=lambda v: f"{v:.4f}"))

    # --- 6. salvar tudo ---
    (PASTA_RESULTADOS / f"metricas{sufixo}.json").write_text(
        json.dumps(todas, indent=2, ensure_ascii=False), encoding="utf-8")
    (PASTA_RESULTADOS / f"relatorio_classificacao{sufixo}.txt").write_text(
        "\n\n".join(relatorios), encoding="utf-8")
    ic.to_csv(PASTA_RESULTADOS / f"intervalos_bootstrap{sufixo}.csv", index=False, encoding="utf-8")
    tabela_rev.to_csv(PASTA_RESULTADOS / f"precisao_por_revocacao{sufixo}.csv", encoding="utf-8")
    if cv_tabela is not None:
        cv_tabela.to_csv(PASTA_RESULTADOS / f"validacao_cruzada{sufixo}.csv", index=False, encoding="utf-8")
    tabela_impureza.to_csv(PASTA_RESULTADOS / f"importancia_variaveis{sufixo}.csv", encoding="utf-8")
    tabela_grupos.to_csv(PASTA_RESULTADOS / f"importancia_grupos{sufixo}.csv", encoding="utf-8")

    # Figuras do artigo
    plotar_matrizes_confusao(matrizes, CLASSES, PASTA_FIGURAS / f"06_matrizes_confusao{sufixo}.png")
    plotar_importancia_modelos(tabela_grupos, PASTA_FIGURAS / f"07_importancia_grupos{sufixo}.png")
    plotar_importancia(tabela_impureza, "importancia_impureza",
                       "Importância por impureza (Random Forest)",
                       PASTA_FIGURAS / f"08_importancia_impureza{sufixo}.png")
    prevalencia = float((y_te == "FATAL").mean())
    curvas = {nome: curva_fatal(y_te, probas[nome]) for nome in ordem_imp}
    pontos = {nome: (m["recall_fatal"], m["precisao_fatal"]) for nome, m in
              ((t["modelo"], t) for t in todas) if nome in ordem_imp}
    plotar_curvas_fatal(curvas, pontos, prevalencia,
                        PASTA_FIGURAS / f"10_curva_precisao_revocacao_fatal{sufixo}.png")

    # Modelo treinado, para reutilizar depois sem treinar de novo (não vai para o GitHub)
    joblib.dump(rf, PASTA_SAIDAS / f"random_forest{sufixo}.joblib", compress=3)
    print("Pronto. Resultados em results/ e figures/.")


# Só roda main() quando o arquivo é executado direto (python -m src.train),
# e não quando é apenas importado por outro arquivo.
if __name__ == "__main__":
    main()
