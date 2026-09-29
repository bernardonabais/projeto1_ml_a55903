import json
import numpy as np
import pandas as pd

def carregar_dados(caminho="data/listings.csv"):
    """
    Carrega o ficheiro do Inside Airbnb e faz as limpezas iniciais:
    - converte o preço de texto para número
    - remove linhas sem preço
    - remove colunas 100% vazias e outras sem valor preditívo

    Devolve o dataframe limpo, pronto para o resto do pipeline.
    """
    df = pd.read_csv(caminho, low_memory=False)

    # Converter o preço de "$X,YZ.WW" para float
    df["price"] = (
        df["price"].str.replace(r"[$,]", "", regex=True).astype(float))

    # Remover linhas sem preço (o alvo não se imputa)
    df = df.dropna(subset=["price"]).copy()

    # Remover colunas 100% vazias
    colunas_vazias = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=colunas_vazias)

    # Remover colunas sem valor preditivo direto
    colunas_a_remover = [
        "host_about", "host_location", "description", "picture_url",
        "listing_url", "scrape_id", "last_scraped", "source",
        "name", "host_url", "host_profile_id", "host_profile_url",
        "host_name", "host_thumbnail_url", "host_picture_url",
        "calendar_last_scraped",
        # Fugas de informação: derivadas do preço
        "price_quote_total_price", "price_quote_price_per_night",
        "price_quote_raw", "estimated_revenue_l365d",
    ]
    colunas_existentes = [c for c in colunas_a_remover if c in df.columns]
    df = df.drop(columns=colunas_existentes)

    return df

def criar_features_duracao(df):
    """
    Cria duas features a partir das datas da estadia simulada:
    - `noites`: número de noites (checkout - checkin)
    - `long_stay`: 1 se noites >= 28, 0 caso contrário

    As colunas originais das datas são removidas depois.
    """
    df = df.copy()

    df["noites"] = (
        pd.to_datetime(df["price_quote_checkout_date"])
        - pd.to_datetime(df["price_quote_checkin_date"])
    ).dt.days

    df["long_stay"] = (df["noites"] >= 28).astype(int)

    df = df.drop(columns=["price_quote_checkin_date",
                          "price_quote_checkout_date"])

    return df

def tratar_datas_reviews(df):
    """
    Converte `first_review` e `last_review` (texto com datas) em duas
    features numéricas:
    - `dias_desde_primeira_review`: idade do anúncio no Airbnb
    - `dias_desde_ultima_review`: quão recente é a última atividade

    A data de referência é a data mais recente presente no dataset (uma
    aproximação da data do scrape).
    Se um alojamento não tem reviews, ambas ficam a NaN (serão imputadas
    depois com a mediana do treino).
    """
    df = df.copy()

    first = pd.to_datetime(df["first_review"], errors="coerce")
    last = pd.to_datetime(df["last_review"], errors="coerce")

    # Data de referência: a mais recente entre todas as reviews
    referencia = last.max()

    df["dias_desde_primeira_review"] = (referencia - first).dt.days
    df["dias_desde_ultima_review"] = (referencia - last).dt.days

    df = df.drop(columns=["first_review", "last_review"])
    return df

# amenities


# Categorias temáticas (ordem importa: primeira que "encaixa" ganha)
CATEGORIAS_AMENITIES = {
    "luxo": ["pool", "hot tub", "sauna", "gym", "doorman", "concierge",
             "bbq grill", "fireplace", "piano", "bidet"],
    "vista_exterior": ["balcony", "patio", "garden", "backyard",
                       "view", "beach access"],
    "cozinha_completa": ["oven", "stove", "dishwasher", "coffee maker",
                         "toaster", "blender", "rice maker", "wine glass",
                         "barware"],
    "familia": ["crib", "high chair", "children", "baby", "board game"],
    "seguranca": ["smoke alarm", "carbon monoxide", "fire extinguisher",
                  "first aid", "lock on bedroom", "safe", "security camera"],
    "acessibilidade": ["elevator", "step-free", "wide entrance",
                       "ground floor", "accessible"],
    "estacionamento": ["parking", "ev charger"],
    "servicos_host": ["self check-in", "lockbox", "keypad", "smart lock",
                      "building staff", "cleaning available",
                      "long term stays", "luggage dropoff"],
    "tecnologia": ["tv", "hdtv", "cable", "sound system", "game console",
                   "books and reading", "ethernet", "workspace"],
    "basicas": ["wifi", "air conditioning", "heating", "hot water",
                "kitchen", "refrigerator", "washer", "dryer", "iron",
                "hangers", "bed linens", "essentials", "shampoo",
                "hair dryer", "shower gel", "cooking basics",
                "dishes", "microwave", "kettle", "toilet paper", "towel"],
}

BINARIAS_LUXO = {
    "tem_piscina": ["pool"],
    "tem_jacuzzi": ["hot tub"],
    "tem_gym": ["gym"],
    "tem_doorman": ["doorman"],
    "tem_vista_mar": ["sea view", "ocean view"],
    "tem_vista_cidade": ["city skyline"],
}


def _classificar_amenity(amenity):
    """Devolve o nome da categoria a que uma amenity pertence."""
    a = amenity.lower()
    for cat, palavras in CATEGORIAS_AMENITIES.items():
        for palavra in palavras:
            if palavra in a:
                return cat
    return "outras"


def _contar_por_categoria(lista_amenities):
    """Recebe uma lista de amenities e devolve um dicionário com as contagens."""
    contagens = {cat: 0 for cat in CATEGORIAS_AMENITIES}
    contagens["outras"] = 0
    for a in lista_amenities:
        contagens[_classificar_amenity(a)] += 1
    return contagens


def _tem_amenity(lista_amenities, palavras):
    """Devolve 1 se alguma amenity da lista contém alguma das palavras dadas."""
    lista_lower = [a.lower() for a in lista_amenities]
    return int(any(p in a for a in lista_lower for p in palavras))


def tratar_amenities(df):
    """
    Converte a coluna `amenities` (texto JSON) em várias features numéricas:
    - `n_amenities`: total por alojamento
    - 11 colunas `amen_<categoria>`: contagem por categoria
    - 6 colunas binárias de luxo (tem_piscina, tem_jacuzzi, ...)

    A coluna original `amenities` é removida.
    """
    df = df.copy()

    # Converter texto JSON em lista
    listas = df["amenities"].apply(json.loads)

    # Feature: número total de amenities
    df["n_amenities"] = listas.apply(len)

    # Features de contagem por categoria
    contagens = listas.apply(_contar_por_categoria).apply(pd.Series)
    contagens.columns = ["amen_" + c for c in contagens.columns]
    df = pd.concat([df, contagens], axis=1)

    # Binárias de luxo
    for nome, palavras in BINARIAS_LUXO.items():
        df[nome] = listas.apply(lambda x: _tem_amenity(x, palavras))

    df = df.drop(columns=["amenities"])
    return df


#EXTRAIR bathrooms A PARTIR DE bathrooms_text


def tratar_bathrooms_text(df):
    """
    Muitos valores de `bathrooms` estão em falta, mas `bathrooms_text`
    contém a mesma informação em texto (por exemplo "1 bath", "1.5 baths",
    "Half-bath", "Shared 2 baths"). Esta função extrai o número quando
    `bathrooms` está em falta e depois remove a coluna de texto.
    """
    import re

    df = df.copy()

    def extrair(texto):
        if not isinstance(texto, str):
            return np.nan
        t = texto.lower().strip()
        if "half" in t:
            return 0.5
        m = re.search(r"(\d+(?:\.\d+)?)", t)
        return float(m.group(1)) if m else np.nan

    extraido = df["bathrooms_text"].apply(extrair)
    df["bathrooms"] = df["bathrooms"].fillna(extraido)

    df = df.drop(columns=["bathrooms_text"])
    return df


# 5. SPLIT TREINO / VALIDAÇÃO / TESTE


def split_treino_val_teste(df, prop_treino=0.7, prop_val=0.15,
                            prop_teste=0.15, seed=42):
    """
    Divide o dataframe em três conjuntos disjuntos:
    - treino (70% por defeito): usado para aprender parâmetros
      (médias para imputação, escalas, limites de capping, modelo)
    - validação (15%): usado para escolher hiperparâmetros
    - teste (15%): usado só uma vez, no fim, para reportar

    A divisão é aleatória mas reprodutível (seed fixa).
    Devolve três dataframes.
    """
    assert abs(prop_treino + prop_val + prop_teste - 1.0) < 1e-9, \
        "As proporções têm de somar 1"

    df_baralhado = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_baralhado)
    n_treino = int(n * prop_treino)
    n_val = int(n * prop_val)

    df_treino = df_baralhado.iloc[:n_treino].copy()
    df_val    = df_baralhado.iloc[n_treino:n_treino + n_val].copy()
    df_teste  = df_baralhado.iloc[n_treino + n_val:].copy()

    return df_treino, df_val, df_teste


# ---------------------------------------------------------------------------
# 6. IMPUTAÇÃO DE MISSING VALUES
# ---------------------------------------------------------------------------

COLUNAS_REVIEWS = [
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value", "reviews_per_month",
]

COLUNAS_A_IMPUTAR = ["bedrooms", "bathrooms", "beds",
                      "dias_desde_primeira_review",
                      "dias_desde_ultima_review"]


def treinar_imputadores(df_treino, estrategia="simples"):
    """
    Aprende os valores de imputação a partir do conjunto de treino.

    estrategia:
        "simples"  -> mediana global por coluna
        "agrupada" -> mediana por (room_type, accommodates) para bedrooms,
                      bathrooms e beds; scores continuam com mediana global
                      dos alojamentos que têm reviews.

    Devolve um dicionário com os valores aprendidos, para depois aplicar
    com `aplicar_imputacao`.
    """
    imp = {"estrategia": estrategia}

    # ---- Colunas de reviews (política igual nas duas estratégias) ----
    # Preencher com a mediana dos alojamentos que TÊM reviews.
    tem_reviews_mask = df_treino["review_scores_rating"].notna()
    imp["reviews"] = {
        col: df_treino.loc[tem_reviews_mask, col].median()
        for col in COLUNAS_REVIEWS if col in df_treino.columns
    }
    imp["reviews_per_month_zero"] = 0.0  # para os que não têm reviews

    # ---- Colunas numéricas (bedrooms, bathrooms, beds) ----
    if estrategia == "simples":
        imp["numericas"] = {
            col: df_treino[col].median()
            for col in COLUNAS_A_IMPUTAR if col in df_treino.columns
        }

    elif estrategia == "agrupada":
        imp["numericas_grupo"] = {}
        for col in COLUNAS_A_IMPUTAR:
            if col not in df_treino.columns:
                continue
            # Mediana por grupo (room_type x accommodates)
            medianas = (df_treino
                        .groupby(["room_type", "accommodates"])[col]
                        .median())
            imp["numericas_grupo"][col] = medianas
        # Fallback: mediana global, para grupos que não aparecem no treino
        imp["numericas_fallback"] = {
            col: df_treino[col].median()
            for col in COLUNAS_A_IMPUTAR if col in df_treino.columns
        }

    else:
        raise ValueError(f"Estratégia desconhecida: {estrategia}")

    return imp


def aplicar_imputacao(df, imputadores):
    """
    Aplica os valores aprendidos por `treinar_imputadores` a qualquer conjunto.
    Também cria a feature `tem_reviews` (1 se tem reviews, 0 caso contrário).
    """
    df = df.copy()

    # ---- Feature "tem reviews" ----
    df["tem_reviews"] = df["review_scores_rating"].notna().astype(int)

    # ---- Preencher scores de reviews ----
    for col, valor in imputadores["reviews"].items():
        if col in df.columns:
            df[col] = df[col].fillna(valor)
    if "reviews_per_month" in df.columns:
        # reviews_per_month faz sentido a 0 para quem não tem reviews
        df["reviews_per_month"] = df["reviews_per_month"].fillna(
            imputadores["reviews_per_month_zero"])

    # ---- Preencher colunas numéricas ----
    if imputadores["estrategia"] == "simples":
        for col, valor in imputadores["numericas"].items():
            if col in df.columns:
                df[col] = df[col].fillna(valor)

    elif imputadores["estrategia"] == "agrupada":
        for col, medianas in imputadores["numericas_grupo"].items():
            if col not in df.columns:
                continue
            # Para cada linha com missing, procurar a mediana do seu grupo
            def preencher(row):
                if pd.notna(row[col]):
                    return row[col]
                chave = (row["room_type"], row["accommodates"])
                if chave in medianas.index:
                    valor = medianas.loc[chave]
                    if pd.notna(valor):
                        return valor
                # Fallback: mediana global (chave não existe ou grupo todo NaN)
                return imputadores["numericas_fallback"][col]
            df[col] = df.apply(preencher, axis=1)

    return df


# 7. TRATAMENTO DE OUTLIERS DE PREÇO


def treinar_outliers(df_treino, estrategia="nenhum"):
    """
    Aprende os limites de outliers a partir do treino.

    estrategia:
        "nenhum"    -> não faz nada
        "capping"   -> limites do IQR sobre log-preço
        "filtragem" -> união de IQR e MAD sobre log-preço

    Devolve um dicionário com os limites aprendidos, para aplicar depois.
    """
    out = {"estrategia": estrategia}

    if estrategia == "nenhum":
        return out

    log_preco = np.log(df_treino["price"])

    if estrategia == "capping":
        # Limites do IQR
        q1, q3 = log_preco.quantile([0.25, 0.75])
        iqr = q3 - q1
        out["lim_inf"] = float(np.exp(q1 - 1.5 * iqr))
        out["lim_sup"] = float(np.exp(q3 + 1.5 * iqr))

    elif estrategia == "filtragem":
        # Critério IQR
        q1, q3 = log_preco.quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf_iqr = q1 - 1.5 * iqr
        lim_sup_iqr = q3 + 1.5 * iqr
        # Critério MAD
        mediana = log_preco.median()
        mad = (log_preco - mediana).abs().median()
        lim_inf_mad = mediana - 3 * 1.4826 * mad
        lim_sup_mad = mediana + 3 * 1.4826 * mad
        # Guardar tudo (para aplicar união depois)
        out["lim_inf_iqr"] = float(np.exp(lim_inf_iqr))
        out["lim_sup_iqr"] = float(np.exp(lim_sup_iqr))
        out["lim_inf_mad"] = float(np.exp(lim_inf_mad))
        out["lim_sup_mad"] = float(np.exp(lim_sup_mad))

    else:
        raise ValueError(f"Estratégia desconhecida: {estrategia}")

    return out


def aplicar_outliers(df, outliers, aplicar_ao_treino=True):
    """
    Aplica o tratamento de outliers ao dataframe.

    Se estrategia=="filtragem", só filtra se `aplicar_ao_treino=True`.
    Nos conjuntos de validação/teste NUNCA se removem linhas (queremos
    ver a performance no mundo real, com outliers e tudo).
    """
    df = df.copy()

    if outliers["estrategia"] == "nenhum":
        return df

    if outliers["estrategia"] == "capping":
        # Cortar preços fora dos limites (nos três conjuntos)
        df["price"] = df["price"].clip(lower=outliers["lim_inf"],
                                        upper=outliers["lim_sup"])
        return df

    if outliers["estrategia"] == "filtragem":
        if not aplicar_ao_treino:
            return df
        # União dos dois critérios
        preco = df["price"]
        mask_iqr = (preco < outliers["lim_inf_iqr"]) | (preco > outliers["lim_sup_iqr"])
        mask_mad = (preco < outliers["lim_inf_mad"]) | (preco > outliers["lim_sup_mad"])
        mask = mask_iqr | mask_mad
        return df.loc[~mask].copy()

    return df


# 8. CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS


CATEGORICAS = [
    "room_type", "property_type", "neighbourhood_cleansed",
    "host_is_superhost", "host_has_profile_pic",
    "host_identity_verified", "has_availability",
]

def treinar_codificador(df_treino, min_freq=30):
    """
    Aprende, a partir do treino, quais são as categorias válidas de cada
    variável categórica.

    Categorias com menos de `min_freq` ocorrências no treino são agrupadas
    num rótulo "Outros", para não criar colunas one-hot quase vazias.

    Devolve um dicionário { coluna: [categorias válidas] }.
    """
    cod = {"min_freq": min_freq, "categorias": {}}
    for col in CATEGORICAS:
        if col not in df_treino.columns:
            continue
        contagens = df_treino[col].value_counts(dropna=False)
        validas = contagens[contagens >= min_freq].index.tolist()
        cod["categorias"][col] = validas
    return cod


def aplicar_codificador(df, codificador):
    """
    Aplica one-hot encoding usando as categorias aprendidas no treino.
    Categorias raras ou desconhecidas ficam como "Outros".
    """
    df = df.copy()

    for col, validas in codificador["categorias"].items():
        if col not in df.columns:
            continue
        # Substituir valores raros/desconhecidos por "Outros"
        df[col] = df[col].where(df[col].isin(validas), other="Outros")
        # Converter para string, para evitar problemas com booleanos, etc.
        df[col] = df[col].astype(str)

    # One-hot encoding só das categóricas listadas
    colunas_a_codificar = [c for c in codificador["categorias"] if c in df.columns]
    df = pd.get_dummies(df, columns=colunas_a_codificar,
                         prefix=colunas_a_codificar, drop_first=False)

    # Converter as colunas dummy (bool) para int (0/1)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    return df


# ESCALAR VARIÁVEIS NUMÉRICAS

# Colunas que NÃO devem ser escaladas (alvo, ids, binárias, dummies)
COLUNAS_NAO_ESCALAR = {"price", "id", "host_id"}


def _colunas_a_escalar(df):
    """Devolve as colunas numéricas que devem ser padronizadas."""
    a_escalar = []
    for c in df.columns:
        if c in COLUNAS_NAO_ESCALAR:
            continue
        if df[c].dtype not in ("int64", "float64", "int32", "float32"):
            continue
        # Colunas binárias (0/1) não precisam de escalar
        valores_unicos = df[c].dropna().unique()
        if set(valores_unicos).issubset({0, 1}):
            continue
        a_escalar.append(c)
    return a_escalar


def treinar_escalador(df_treino):
    """
    Aprende, a partir do treino, a média e o desvio-padrão de cada coluna
    numérica não-binária. Devolve um dicionário para aplicar depois.
    """
    a_escalar = _colunas_a_escalar(df_treino)
    esc = {
        "colunas": a_escalar,
        "media": {c: float(df_treino[c].mean()) for c in a_escalar},
        "desvio": {c: float(df_treino[c].std()) for c in a_escalar},
    }
    return esc


def aplicar_escalador(df, escalador):
    """
    Padroniza as colunas: (x - media) / desvio.
    Usa os valores aprendidos no treino, para evitar leakage.
    """
    df = df.copy()
    for c in escalador["colunas"]:
        if c not in df.columns:
            continue
        desvio = escalador["desvio"][c]
        if desvio == 0:
            df[c] = 0.0  # coluna constante, não dá para escalar
        else:
            df[c] = (df[c] - escalador["media"][c]) / desvio
    return df

#feature engineering

def adicionar_polinomiais(df, colunas, grau=2):
    """
    Adiciona termos polinomiais para as colunas dadas.
    Exemplo: para `accommodates` com grau=2, cria `accommodates_pot2`.
    """
    df = df.copy()
    for col in colunas:
        if col not in df.columns:
            continue
        for g in range(2, grau + 1):
            df[f"{col}_pot{g}"] = df[col] ** g
    return df


def adicionar_interacoes(df, pares):
    """
    Adiciona termos de interação (multiplicação) entre pares de colunas.

    `pares` é uma lista de tuplos (col_a, col_b). Se col_b for um prefixo,
    interage col_a com todas as colunas one-hot que começam por col_b.
    """
    df = df.copy()
    for col_a, col_b in pares:
        if col_a not in df.columns:
            continue
        # Se col_b for um prefixo, procurar todas as colunas one-hot
        colunas_b = [c for c in df.columns if c.startswith(col_b + "_")]
        if not colunas_b and col_b in df.columns:
            colunas_b = [col_b]
        for cb in colunas_b:
            nome = f"{col_a}_x_{cb}"
            df[nome] = df[col_a] * df[cb]
    return df
# 10. PIPELINE COMPLETO


def preparar_dados(caminho="data/listings.csv",
                    estrategia_missing="simples",
                    estrategia_outliers="nenhum",
                    log_preco=True,
                    usar_feature_engineering=False,
                    seed=42):
    """
    Pipeline completo: carrega os dados, aplica todas as transformações
    e devolve os três conjuntos prontos para treinar modelos.

    Parâmetros:
        caminho             - onde está o CSV
        estrategia_missing  - "simples" ou "agrupada"
        estrategia_outliers - "nenhum", "capping" ou "filtragem"
        log_preco           - se True, o alvo é log(price) em vez de price
        seed                - para reprodutibilidade do split

    Devolve um dicionário com:
        X_treino, y_treino, X_val, y_val, X_teste, y_teste
        (todos como DataFrames/Series do pandas)
    e os "transformers" aprendidos, caso queiras reutilizar.
    """
    # 1-4. Carregar e criar features
    df = carregar_dados(caminho)
    df = criar_features_duracao(df)
    df = tratar_datas_reviews(df)    
    df = tratar_amenities(df)
    df = tratar_bathrooms_text(df)

    # 5. Split
    df_treino, df_val, df_teste = split_treino_val_teste(df, seed=seed)

    # 6. Imputação (aprendida no treino, aplicada em todos)
    imp = treinar_imputadores(df_treino, estrategia=estrategia_missing)
    df_treino = aplicar_imputacao(df_treino, imp)
    df_val    = aplicar_imputacao(df_val, imp)
    df_teste  = aplicar_imputacao(df_teste, imp)

    # 7. Outliers (só no treino se for filtragem)
    out = treinar_outliers(df_treino, estrategia=estrategia_outliers)
    df_treino = aplicar_outliers(df_treino, out, aplicar_ao_treino=True)
    df_val    = aplicar_outliers(df_val,    out, aplicar_ao_treino=False)
    df_teste  = aplicar_outliers(df_teste,  out, aplicar_ao_treino=False)

    # 8. Codificar categóricas
    cod = treinar_codificador(df_treino)
    df_treino = aplicar_codificador(df_treino, cod)
    df_val    = aplicar_codificador(df_val, cod)
    df_teste  = aplicar_codificador(df_teste, cod)

    # Garantir mesmas colunas nos três conjuntos
    todas_colunas = df_treino.columns
    for outro in (df_val, df_teste):
        for c in todas_colunas:
            if c not in outro.columns:
                outro[c] = 0
    df_val   = df_val[todas_colunas]
    df_teste = df_teste[todas_colunas]

        # 8b. Feature engineering (opcional)
    if usar_feature_engineering:
        colunas_pol = ["accommodates", "bedrooms", "bathrooms"]
        df_treino = adicionar_polinomiais(df_treino, colunas_pol, grau=2)
        df_val    = adicionar_polinomiais(df_val,    colunas_pol, grau=2)
        df_teste  = adicionar_polinomiais(df_teste,  colunas_pol, grau=2)

        pares = [("long_stay", "room_type"),
                 ("accommodates", "room_type")]
        df_treino = adicionar_interacoes(df_treino, pares)
        df_val    = adicionar_interacoes(df_val,    pares)
        df_teste  = adicionar_interacoes(df_teste,  pares)

        # Re-alinhar colunas (as interações podem gerar colunas diferentes)
        todas = df_treino.columns
        for outro in (df_val, df_teste):
            for c in todas:
                if c not in outro.columns:
                    outro[c] = 0
        df_val   = df_val[todas]
        df_teste = df_teste[todas]

    # 9. Escalar numéricas
    esc = treinar_escalador(df_treino)
    df_treino = aplicar_escalador(df_treino, esc)
    df_val    = aplicar_escalador(df_val, esc)
    df_teste  = aplicar_escalador(df_teste, esc)

    # Separar alvo (y) das features (X)
    def separar(d):
        y = np.log(d["price"]) if log_preco else d["price"].copy()
        X = d.drop(columns=["price"])
        # Remover ids (não são features)
        for c in ("id", "host_id"):
            if c in X.columns:
                X = X.drop(columns=[c])
        return X, y

    X_treino, y_treino = separar(df_treino)
    X_val,    y_val    = separar(df_val)
    X_teste,  y_teste  = separar(df_teste)

    return {
        "X_treino": X_treino, "y_treino": y_treino,
        "X_val":    X_val,    "y_val":    y_val,
        "X_teste":  X_teste,  "y_teste":  y_teste,
        "imputador":   imp,
        "outliers":    out,
        "codificador": cod,
        "escalador":   esc,
    }