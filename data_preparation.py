"""
data_preparation.py

Pré-processamento dos dados do Airbnb de Hong Kong para o Projeto 1 de ML.
Cada função trata de uma parte específica do pipeline.
"""
import os
import json
import numpy as np
import pandas as pd

# 1. CARREGAR DADOS
def carregar_dados(caminho="data/listings.csv"):
    """Carrega o CSV, converte o preço, remove linhas sem preço e colunas vazias."""
    df = pd.read_csv(caminho, low_memory=False)
    df["price"] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float)
    df = df.dropna(subset=["price"]).copy()
    df = df.dropna(axis=1, how="all")   # remove colunas 100% vazias
    return df



# 2. FEATURES DE DURAÇÃO

def criar_features_duracao(df):
    """Cria 'noites' e 'long_stay' a partir das datas de check-in/check-out."""
    df = df.copy()
    checkin = pd.to_datetime(df["price_quote_checkin_date"])
    checkout = pd.to_datetime(df["price_quote_checkout_date"])
    df["noites"] = (checkout - checkin).dt.days
    df["long_stay"] = (df["noites"] >= 28).astype(int)
    return df


# 3. AMENITIES (top 15 binárias)

def criar_features_amenities(df, top_n=15):
    """Cria 'n_amenities' e uma coluna binária por cada uma das top_n amenities."""
    df = df.copy()
    df["amenities_lista"] = df["amenities"].apply(json.loads)
    df["n_amenities"] = df["amenities_lista"].apply(len)

    todas = [a for lista in df["amenities_lista"] for a in lista]
    top = pd.Series(todas).value_counts().head(top_n).index.tolist()

    for amenity in top:
        nome = "tem_" + amenity.lower().replace(" ", "_")
        df[nome] = df["amenities_lista"].apply(lambda x: int(amenity in x))

    return df



# 4. BATHROOMS a partir do texto

def extrair_bathrooms(df):
    """Preenche valores em falta de 'bathrooms' com o número extraído de 'bathrooms_text'."""
    df = df.copy()
    numero = df["bathrooms_text"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
    df["bathrooms"] = df["bathrooms"].fillna(numero)
    return df



# 5. SPLIT TREINO / VALIDAÇÃO / TESTE
def split_treino_val_teste(df, seed=42):
    """Divide em 70% treino, 15% validação, 15% teste de forma reprodutível."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_treino = int(0.70 * n)
    n_val = int(0.15 * n)
    treino = df.iloc[:n_treino]
    val = df.iloc[n_treino:n_treino + n_val]
    teste = df.iloc[n_treino + n_val:]
    return treino, val, teste


# 6. TRATAMENTO DE MISSING VALUES
def tratar_missing(treino, outros, estrategia="simples"):
    """
    Imputa missing values usando uma das duas políticas:
    - 'simples': mediana global, aprendida só do treino.
    - 'agrupada': mediana por grupo (room_type, accommodates), aprendida só do treino.

    'outros' é uma lista de dataframes (validação, teste) para aplicar os mesmos valores.
    """
    treino = treino.copy()
    outros = [d.copy() for d in outros]

    # Indicador de 'tem_reviews' (ausência com significado)
    for d in [treino] + outros:
        d["tem_reviews"] = d["review_scores_rating"].notna().astype(int)

    cols_reviews = [c for c in treino.columns if c.startswith("review_scores_")]
    cols_num = ["bedrooms", "bathrooms", "beds"]

    # Reviews: mediana dos que têm reviews (sempre do treino)
    for col in cols_reviews:
        med = treino.loc[treino["tem_reviews"] == 1, col].median()
        for d in [treino] + outros:
            d[col] = d[col].fillna(med)

    # bedrooms, bathrooms, beds: duas estratégias possíveis
    if estrategia == "simples":
        for col in cols_num:
            med = treino[col].median()
            for d in [treino] + outros:
                d[col] = d[col].fillna(med)

    elif estrategia == "agrupada":
        for col in cols_num:
            medianas = treino.groupby(["room_type", "accommodates"])[col].median()
            medianas = medianas.dropna()        # grupos sem dados caem no fallback
            med_geral = treino[col].median()    # fallback
            for d in [treino] + outros:
                d[col] = d.apply(
                    lambda r: r[col] if pd.notna(r[col])
                    else medianas.get((r["room_type"], r["accommodates"]), med_geral),
                    axis=1
                )
    else:
        raise ValueError("estrategia tem de ser 'simples' ou 'agrupada'")

    return treino, outros



# 7. TRATAMENTO DE OUTLIERS
def tratar_outliers(treino, outros, estrategia="nenhuma"):
    """
    Trata outliers no preço com uma de três estratégias:
    - 'nenhuma': não faz nada
    - 'capping': aplica limites do IQR (sobre log-preço) ao preço de todos os conjuntos
    - 'filtragem': remove linhas do TREINO marcadas por IQR ou MAD (união)
    """
    treino = treino.copy()
    outros = [d.copy() for d in outros]

    if estrategia == "nenhuma":
        return treino, outros

    log_p = np.log(treino["price"])

    if estrategia == "capping":
        q1, q3 = log_p.quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf = np.exp(q1 - 1.5 * iqr)
        lim_sup = np.exp(q3 + 1.5 * iqr)
        # Só o treino é alterado: a validação e o teste mantêm os preços reais,
        # para que as métricas meçam o erro contra a realidade.
        treino["price"] = treino["price"].clip(lower=lim_inf, upper=lim_sup)

    elif estrategia == "filtragem":
        # IQR
        q1, q3 = log_p.quantile([0.25, 0.75])
        iqr = q3 - q1
        out_iqr = (log_p < q1 - 1.5 * iqr) | (log_p > q3 + 1.5 * iqr)
        # MAD
        med = log_p.median()
        mad = (log_p - med).abs().median()
        z = (log_p - med).abs() / (1.4826 * mad)
        out_mad = z > 3
        # União (só se remove do treino, nunca do teste)
        manter = ~(out_iqr | out_mad)
        treino = treino[manter].copy()
    else:
        raise ValueError("estrategia tem de ser 'nenhuma', 'capping' ou 'filtragem'")

    return treino, outros


# 8. MATRIZ FINAL (one-hot + escala)
def preparar_matriz_final(treino, outros, features_num, features_cat,
                          polinomiais=False, interacoes=False):
    """
    One-hot das categóricas, termos polinomiais/interações (antes de escalar),
    e standardização com média/std do treino.
    """
    def transformar(d):
        X_num = d[features_num].copy()

        # Polinomiais ANTES de escalar (nos valores originais)
        if polinomiais:
            if "accommodates" in X_num.columns:
                X_num["accommodates_sq"] = X_num["accommodates"] ** 2
            if "n_amenities" in X_num.columns:
                X_num["n_amenities_sq"] = X_num["n_amenities"] ** 2

        X_cat = pd.get_dummies(d[features_cat], drop_first=True).astype(int)
        X = pd.concat([X_num, X_cat], axis=1)

        # Interações: long_stay x room_type (one-hot já criado)
        if interacoes and "long_stay" in X.columns:
            for c in [col for col in X.columns if col.startswith("room_type_")]:
                X[f"long_stay_x_{c}"] = X["long_stay"] * X[c]

        return X

    X_treino = transformar(treino)
    Xs_outros = [transformar(d) for d in outros]

    # Alinhar colunas
    for X in Xs_outros:
        X_treino, _ = X_treino.align(X, join="outer", axis=1, fill_value=0)
    Xs_outros = [X.reindex(columns=X_treino.columns, fill_value=0) for X in Xs_outros]

        # Standardizar as features numéricas e os polinomiais (one-hot fica em 0/1)
    cols_escalar = [c for c in features_num if c in X_treino.columns]
    if polinomiais:
        cols_escalar += [c for c in ["accommodates_sq", "n_amenities_sq"]
                         if c in X_treino.columns]
    media = X_treino[cols_escalar].mean()
    desvio = X_treino[cols_escalar].std().replace(0, 1)
    X_treino[cols_escalar] = (X_treino[cols_escalar] - media) / desvio
    for X in Xs_outros:
        X[cols_escalar] = (X[cols_escalar] - media) / desvio

    y_treino = np.log(treino["price"])
    ys_outros = [np.log(d["price"]) for d in outros]

    return X_treino, Xs_outros, y_treino, ys_outros



# 9. LISTAS DE FEATURES
def listar_features(df):
    """
    Devolve as listas de features usadas pelos modelos.
    Centralizada aqui para que o pipeline e o k-fold usem sempre as mesmas.
    """
    features_num = ["accommodates", "bedrooms", "bathrooms", "beds",
                    "minimum_nights", "noites", "n_amenities", "long_stay",
                    "review_scores_rating", "number_of_reviews"]
    features_cat = ["room_type", "neighbourhood_cleansed"]
    features_bin = [c for c in df.columns if c.startswith("tem_")]
    return features_num + features_bin, features_cat


# 10. GUARDAR MATRIZ FINAL EM DISCO
def guardar_matriz(X_treino, Xs_outros, y_treino, ys_outros,
                   pasta="data/processado"):
    """Guarda as matrizes de features e os vetores alvo em ficheiros CSV."""
    os.makedirs(pasta, exist_ok=True)

    X_treino.to_csv(f"{pasta}/X_treino.csv", index=False)
    Xs_outros[0].to_csv(f"{pasta}/X_val.csv", index=False)
    Xs_outros[1].to_csv(f"{pasta}/X_teste.csv", index=False)

    y_treino.to_csv(f"{pasta}/y_treino.csv", index=False)
    ys_outros[0].to_csv(f"{pasta}/y_val.csv", index=False)
    ys_outros[1].to_csv(f"{pasta}/y_teste.csv", index=False)

    print(f"Matrizes guardadas em {pasta}/")



# 11. PIPELINE COMPLETO
def pipeline_completo(caminho="data/listings.csv",
                      estrategia_missing="simples",
                      estrategia_outliers="nenhuma",
                      polinomiais=False,
                      interacoes=False,
                      seed=42,
                      guardar=False):
    """Corre o pipeline todo com as estratégias escolhidas."""
    df = carregar_dados(caminho)
    df = criar_features_duracao(df)
    df = extrair_bathrooms(df)
    df = criar_features_amenities(df)

    treino, val, teste = split_treino_val_teste(df, seed=seed)
    treino, (val, teste) = tratar_missing(treino, [val, teste], estrategia_missing)
    treino, (val, teste) = tratar_outliers(treino, [val, teste], estrategia_outliers)

    features_num, features_cat = listar_features(treino)

    X_treino, Xs_outros, y_treino, ys_outros = preparar_matriz_final(
        treino, [val, teste], features_num, features_cat,
        polinomiais=polinomiais, interacoes=interacoes
    )

    if guardar:
        guardar_matriz(X_treino, Xs_outros, y_treino, ys_outros)

    return X_treino, Xs_outros, y_treino, ys_outros




# 12. K-FOLD CROSS-VALIDATION (sem leakage)
def kfold_indices(n, k=5, seed=42):
    """Divide n amostras em k folds. Devolve lista de (indices_treino, indices_val)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)
    resultado = []
    for i in range(k):
        val_idx = folds[i]
        treino_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        resultado.append((treino_idx, val_idx))
    return resultado


def preparar_fold(df, treino_idx, val_idx,
                  estrategia_missing="simples",
                  estrategia_outliers="nenhuma",
                  polinomiais=False, interacoes=False):
    """Prepara um fold, aprendendo tudo apenas no treino deste fold."""
    treino = df.iloc[treino_idx].copy()
    val = df.iloc[val_idx].copy()

    treino, (val,) = tratar_missing(treino, [val], estrategia_missing)
    treino, (val,) = tratar_outliers(treino, [val], estrategia_outliers)

    features_num, features_cat = listar_features(treino)

    return preparar_matriz_final(treino, [val], features_num, features_cat,
                                  polinomiais=polinomiais, interacoes=interacoes)