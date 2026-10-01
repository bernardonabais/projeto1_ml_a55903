"""
logistic_regression.py

Regressão logística binária implementada do zero com gradient descent (batch).
Classe positiva: top 20% de preço dentro de cada grupo (room_type × bairro).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. DEFINIR CLASSE POSITIVA
# ---------------------------------------------------------------------------
def definir_classe_positiva(df, coluna_preco="price",
                             grupos=("room_type", "neighbourhood_cleansed"),
                             percentil=0.80):
    """
    Marca como classe positiva (1) os alojamentos cujo preço está acima
    do percentil especificado DENTRO do seu grupo (room_type × bairro).

    Devolve um vetor 0/1 do tamanho de df.
    """
    df = df.copy()
    limiares = df.groupby(list(grupos))[coluna_preco].quantile(percentil)
    y = []
    for _, linha in df.iterrows():
        chave = tuple(linha[g] for g in grupos)
        limite = limiares.get(chave, df[coluna_preco].quantile(percentil))
        y.append(int(linha[coluna_preco] > limite))
    return np.array(y)


# ---------------------------------------------------------------------------
# 2. SIGMOID
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Função sigmoide: transforma qualquer número em probabilidade entre 0 e 1."""
    z = np.clip(z, -500, 500)   # evita overflow numérico
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 3. TREINO (GRADIENT DESCENT)
# ---------------------------------------------------------------------------
def treinar(X, y, taxa=0.01, n_iters=1000, l2=0.0):
    """
    Treina regressão logística com batch gradient descent.
    Devolve pesos, bias e histórico da perda (log-loss).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    pesos = np.zeros(d)
    bias = 0.0
    historico = []

    for _ in range(n_iters):
        z = X @ pesos + bias
        p = sigmoid(z)

        # Log-loss (com segurança para log(0))
        p_safe = np.clip(p, 1e-10, 1 - 1e-10)
        perda = -(y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe)).mean()
        perda += l2 * (pesos ** 2).sum()
        historico.append(perda)

        # Gradientes
        erro = p - y
        grad_pesos = (X.T @ erro) / n + 2 * l2 * pesos
        grad_bias = erro.mean()

        pesos -= taxa * grad_pesos
        bias -= taxa * grad_bias

    return pesos, bias, historico


# ---------------------------------------------------------------------------
# 4. PREVISÃO
# ---------------------------------------------------------------------------
def prever_prob(X, pesos, bias):
    """Devolve as probabilidades previstas (0 a 1)."""
    X = np.asarray(X, dtype=float)
    return sigmoid(X @ pesos + bias)


def prever_classe(X, pesos, bias, threshold=0.5):
    """Devolve as classes (0 ou 1) usando o threshold especificado."""
    return (prever_prob(X, pesos, bias) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 5. MÉTRICAS
# ---------------------------------------------------------------------------
def metricas(y_real, y_pred, y_prob=None):
    """
    Calcula accuracy, precision, recall, F1, matriz de confusão.
    Se y_prob for passado, calcula também ROC-AUC.
    """
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    tp = int(((y_pred == 1) & (y_real == 1)).sum())
    tn = int(((y_pred == 0) & (y_real == 0)).sum())
    fp = int(((y_pred == 1) & (y_real == 0)).sum())
    fn = int(((y_pred == 0) & (y_real == 1)).sum())

    accuracy = (tp + tn) / len(y_real)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    resultados = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matriz_confusao": np.array([[tn, fp], [fn, tp]]),
    }

    if y_prob is not None:
        resultados["roc_auc"] = roc_auc(y_real, y_prob)

    return resultados


def roc_auc(y_real, y_prob):
    """Calcula a AUC sob a curva ROC pelo método dos pares."""
    y_real = np.asarray(y_real)
    y_prob = np.asarray(y_prob)
    pos = y_prob[y_real == 1]
    neg = y_prob[y_real == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Para cada par (positivo, negativo), conta quantos positivos têm prob > negativo
    maior = (pos[:, None] > neg[None, :]).sum()
    igual = (pos[:, None] == neg[None, :]).sum()
    return (maior + 0.5 * igual) / (len(pos) * len(neg))