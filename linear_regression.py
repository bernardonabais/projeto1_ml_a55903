"""
linear_regression.py

Implementação de regressão linear com gradient descent (batch e mini-batch)
para o projeto de Machine Learning. Suporta regularização L2 como extensão.
"""

import numpy as np


class LinearRegressionGD:
    """
    Regressão linear treinada por gradient descent.

    Parâmetros:
        learning_rate    - passo de atualização dos pesos (default 0.01)
        n_iter           - número máximo de iterações (default 1000)
        batch_size       - tamanho do mini-batch. Se None, usa batch completo.
        l2_lambda        - força da regularização L2. 0 = sem regularização.
        tol              - critério de paragem: se a mudança da loss for
                            menor que isto durante `paciencia` iterações,
                            para. None = não usar early stopping.
        paciencia        - iterações consecutivas com mudança < tol para parar.
        seed             - para reprodutibilidade do baralhamento no mini-batch.
        verbose          - se True, imprime a loss de X em X iterações.
    """

    def __init__(self, learning_rate=0.01, n_iter=1000,
                 batch_size=None, l2_lambda=0.0,
                 tol=None, paciencia=10, seed=42, verbose=False):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.tol = tol
        self.paciencia = paciencia
        self.seed = seed
        self.verbose = verbose

        # Serão preenchidos em fit()
        self.pesos_ = None          # vetor de pesos aprendidos
        self.historico_loss_ = []   # loss ao longo do treino


    def _adicionar_bias(self, X):
        """Adiciona uma coluna de 1s à esquerda de X, para o termo w_0."""
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)), X])

    def _calcular_loss(self, X, y):
        """Calcula o MSE (com termo de regularização L2 se ativado)."""
        m = X.shape[0]
        erro = X @ self.pesos_ - y
        loss = (erro ** 2).sum() / (2 * m)
        if self.l2_lambda > 0:
            # Excluir o bias (w_0) da regularização
            loss += (self.l2_lambda / (2 * m)) * (self.pesos_[1:] ** 2).sum()
        return loss

    def _calcular_gradiente(self, X, y):
        """Calcula o gradiente do MSE (com termo L2 se ativado)."""
        m = X.shape[0]
        erro = X @ self.pesos_ - y
        gradiente = (X.T @ erro) / m
        if self.l2_lambda > 0:
            # Não regularizar o bias: pomos 0 no primeiro elemento
            reg = np.concatenate([[0], self.pesos_[1:]])
            gradiente += (self.l2_lambda / m) * reg
        return gradiente

    def fit(self, X, y):
        """Treina o modelo com gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        X_b = self._adicionar_bias(X)
        m, n_features = X_b.shape

        # Inicializar pesos a zero
        self.pesos_ = np.zeros(n_features)
        self.historico_loss_ = []

        rng = np.random.default_rng(self.seed)
        iters_sem_melhoria = 0

        for it in range(self.n_iter):
            # Escolher que amostras usar nesta iteração
            if self.batch_size is None or self.batch_size >= m:
                X_batch, y_batch = X_b, y
            else:
                idx = rng.choice(m, size=self.batch_size, replace=False)
                X_batch, y_batch = X_b[idx], y[idx]

            # Atualizar pesos
            gradiente = self._calcular_gradiente(X_batch, y_batch)
            self.pesos_ -= self.learning_rate * gradiente

            # Guardar loss (sobre o dataset todo, para monitorização)
            loss = self._calcular_loss(X_b, y)
            self.historico_loss_.append(loss)

            # Verbose
            if self.verbose and it % 100 == 0:
                print(f"  Iter {it:4d}  loss = {loss:.6f}")

            # Early stopping
            if self.tol is not None and it > 0:
                delta = abs(self.historico_loss_[-2] - loss)
                if delta < self.tol:
                    iters_sem_melhoria += 1
                    if iters_sem_melhoria >= self.paciencia:
                        if self.verbose:
                            print(f"  Convergiu na iteração {it}")
                        break
                else:
                    iters_sem_melhoria = 0

        return self

    def predict(self, X):
        """Aplica os pesos aprendidos para prever valores novos."""
        if self.pesos_ is None:
            raise RuntimeError("O modelo tem de ser treinado antes de prever.")
        X = np.asarray(X, dtype=float)
        X_b = self._adicionar_bias(X)
        return X_b @ self.pesos_

    
# Métricas de regressão


def calcular_metricas(y_verdadeiro, y_previsto):
    """
    Devolve MAE, RMSE e R2 num dicionário.
    Todos os cálculos são feitos no mesmo espaço em que y foi treinado
    (por exemplo, log-preço). Se quiseres em HKD, aplica np.exp antes.
    """
    y_verdadeiro = np.asarray(y_verdadeiro, dtype=float)
    y_previsto = np.asarray(y_previsto, dtype=float)

    erro = y_previsto - y_verdadeiro
    mae = np.mean(np.abs(erro))
    rmse = np.sqrt(np.mean(erro ** 2))

    # R2 = 1 - SS_res / SS_tot
    ss_res = np.sum(erro ** 2)
    ss_tot = np.sum((y_verdadeiro - y_verdadeiro.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}



# Gráfico de convergência


def plot_convergencia(modelos, nomes, caminho=None, log=True):
    """
    Desenha a curva de convergência (loss ao longo das iterações) de um
    ou vários modelos. Aceita listas para comparar configurações.

    Parâmetros:
        modelos  - um LinearRegressionGD ou lista deles
        nomes    - nome (str) ou lista de nomes para a legenda
        caminho  - se for uma string, guarda a figura nesse caminho
        log      - se True, usa escala logarítmica no eixo y (recomendado)
    """
    import matplotlib.pyplot as plt

    if not isinstance(modelos, list):
        modelos = [modelos]
        nomes = [nomes]

    fig, ax = plt.subplots(figsize=(9, 5))
    for modelo, nome in zip(modelos, nomes):
        ax.plot(modelo.historico_loss_, label=nome, linewidth=1.5)

    ax.set_xlabel("Iteração")
    ax.set_ylabel("Loss (MSE / 2)")
    ax.set_title("Convergência do gradient descent")
    if log:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if caminho:
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.show()