"""
linear_regression.py

Regressão linear implementada do zero com gradient descent (batch).
Suporta regularização L2 (ridge) como extensão.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. TREINO: GRADIENT DESCENT (BATCH)
# ---------------------------------------------------------------------------
def treinar(X, y, taxa=0.01, n_iters=1000, l2=0.0):
    """
    Treina uma regressão linear com batch gradient descent.

    Parâmetros:
      X     : matriz de features (n_amostras x n_features)
      y     : vetor alvo (n_amostras)
      taxa  : learning rate
      n_iters : número de iterações
      l2    : força da regularização L2 (0 = sem regularização)

    Devolve:
      pesos    : vetor com os coeficientes aprendidos
      bias     : intercept (termo independente)
      historico: lista com o valor da perda em cada iteração
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    pesos = np.zeros(d)
    bias = 0.0
    historico = []

    for _ in range(n_iters):
        # Previsão atual
        y_pred = X @ pesos + bias

        # Erro
        erro = y_pred - y

        # Perda (MSE + L2)
        perda = (erro ** 2).mean() + l2 * (pesos ** 2).sum()
        historico.append(perda)

        # Gradientes
        grad_pesos = (2 / n) * (X.T @ erro) + 2 * l2 * pesos
        grad_bias = (2 / n) * erro.sum()

        # Atualização
        pesos -= taxa * grad_pesos
        bias -= taxa * grad_bias

    return pesos, bias, historico


# ---------------------------------------------------------------------------
# 2. PREVISÃO
# ---------------------------------------------------------------------------
def prever(X, pesos, bias):
    """Devolve as previsões para X com os pesos e o bias aprendidos."""
    X = np.asarray(X, dtype=float)
    return X @ pesos + bias


# ---------------------------------------------------------------------------
# 3. MÉTRICAS
# ---------------------------------------------------------------------------
def metricas(y_real, y_pred):
    """Calcula MAE, RMSE e R² entre valores reais e previstos."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.abs(y_real - y_pred).mean()
    rmse = np.sqrt(((y_real - y_pred) ** 2).mean())
    ss_res = ((y_real - y_pred) ** 2).sum()
    ss_tot = ((y_real - y_real.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# 4. GRÁFICO DE CONVERGÊNCIA
# ---------------------------------------------------------------------------
def plot_convergencia(historico, titulo="Convergência do gradient descent",
                       guardar=None):
    """Desenha a perda ao longo das iterações."""
    plt.figure(figsize=(8, 4))
    plt.plot(historico)
    plt.xlabel("Iteração")
    plt.ylabel("Perda (MSE)")
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    if guardar:
        plt.savefig(guardar, dpi=150, bbox_inches="tight")
    plt.show()