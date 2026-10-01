# Projeto Prático 1 - Machine Learning: Aprendizagem Supervisionada

**Autor:** Bernardo Nabais
**Cadeira:** Machine Learning
**Curso:** Inteligência Artificial e Ciência de Dados, UBI
**Ano letivo:** 2026/2027

---

## 1. Objetivo

Pipeline completo de aprendizagem supervisionada sobre listings do Airbnb, com
dois objetivos:

- **Regressão:** prever o preço por noite a partir das características do alojamento.
- **Classificação:** identificar alojamentos no top 20% de preço dentro de grupos comparáveis.

Todos os modelos são implementados de raiz com gradient descent, sem bibliotecas
de machine learning. O foco está na justificação das decisões de pré-processamento
e na interpretação dos modelos.

---

## 2. Dataset

| | |
|---|---|
| **Cidade** | Hong Kong |
| **Fonte** | [Inside Airbnb - Get the Data](https://insideairbnb.com/get-the-data/) |
| **Datas de scraping** | 2026-06-28, 2026-06-29, 2026-07-03 |
| **Moeda** | HKD (dólares de Hong Kong) |
| **Dimensões originais** | 6 734 linhas × 90 colunas |
| **Após limpeza** | 6 046 linhas × 51 features |

O ficheiro descarregado (`listings.csv.gz`) foi descomprimido e colocado em
`data/listings.csv`. O código lê diretamente esse caminho; `pd.read_csv` aceita
qualquer das duas formas, bastando ajustar o nome.

---

## 3. Estrutura do projeto
'projeto1_ml_a55903/
├── data/
│ ├── listings.csv # Dataset original (descomprimido)
│ └── processado/ # Matrizes finais geradas pelo pipeline
├── figures/ # Figuras geradas pelo notebook
├── report/
│ ├── ablacoes.csv # Tabela comparativa das estratégias
│ ├── coeficientes_linear.csv
│ └── coeficientes_logistica.csv
├── exploracao.ipynb # Análise exploratória e experiências
├── data_preparation.py # Pipeline de pré-processamento
├── linear_regression.py # Regressão linear (gradient descent + L2)
├── logistic_regression.py # Regressão logística (gradient descent)
└── README.md'

---

## 4. Como reproduzir

**Requisitos:** Python 3.11+, com `pandas`, `numpy` e `matplotlib`.
Nenhuma biblioteca de machine learning é usada.

```bash
pip install pandas numpy matplotlib jupyter
```

**Passos:**

1. Descarregar `listings.csv.gz` de Hong Kong em
   [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/),
   descomprimir e colocar em `data/listings.csv`.
2. Abrir `exploracao.ipynb` e executar todas as células por ordem
   (*Restart & Run All*).
3. As figuras são escritas em `figures/`, as tabelas em `report/` e as matrizes
   finais em `data/processado/`.

**Reprodutibilidade:** todas as divisões aleatórias usam `seed=42`
(split treino/validação/teste e k-fold). Os resultados são determinísticos.

**Usar o pipeline diretamente:**

```python
from data_preparation import pipeline_completo

X_treino, (X_val, X_teste), y_treino, (y_val, y_teste) = pipeline_completo(
    estrategia_missing="simples",      # ou "agrupada"
    estrategia_outliers="nenhuma",     # ou "capping" / "filtragem"
    polinomiais=True,
    interacoes=True,
    guardar=True,
)
```

---

## 5. Decisões de pré-processamento

### 5.1 Alvo

- Coluna `price`, convertida de texto (`"$1,181.02"`) para número.
- Confirmado como preço por noite (igual a `price_quote_price_per_night`).
- **688 linhas sem preço removidas** (10,2%) — o alvo nunca é imputado.
- Alvo dos modelos: **log(price)**, por a distribuição ser fortemente
  assimétrica (mediana 424 HKD, máximo 171 234 HKD).

### 5.2 Fugas de informação

Excluídas das features por serem derivadas do alvo:
`price_quote_total_price`, `price_quote_price_per_night`,
`estimated_revenue_l365d`, `price_quote_raw`.

### 5.3 Duração da estadia simulada

- **Decisão:** manter todas as durações e usar a duração como feature.
- **Justificação:** a mediana cai de 545 HKD (1 noite) para 221 HKD (28-31
  noites). Em Hong Kong, estadias com menos de 28 noites exigem licença, o que
  explica a concentração de anúncios no grupo 28-31 (2 535 alojamentos).
- **Features criadas:** `noites` (numérica) e `long_stay` (binária, ≥ 28).
- **Validação da decisão:** o viés dos resíduos é praticamente nulo em todos os
  grupos de duração (entre −0.004 e +0.059 para 1 a 31 noites).

### 5.4 Missing values

**Ausência com significado:**
- Colunas de reviews (~45%, correlação 1.00 entre si — falham nas mesmas
  linhas): criada feature `tem_reviews`; `review_scores_rating` imputado com a
  mediana dos alojamentos que têm reviews.

**Valor desconhecido — duas políticas comparadas:**
- `bedrooms` (44,8%), `bathrooms` (11%), `beds` (4,9%).
- Política **simples:** mediana global do treino.
- Política **agrupada:** mediana por grupo `room_type × accommodates`.
- `bathrooms_text` (0,3% missing) usada para recuperar valores de `bathrooms`
  antes de qualquer imputação.

**Removidas:**
- 14 colunas 100% vazias, incluindo `host_response_rate`, `host_response_time`,
  `host_acceptance_rate`, `neighbourhood`, `license` e `instant_bookable`.
  Limitação do ficheiro desta versão.
- `property_type` (55 categorias), removida por multicolinearidade total com
  `room_type` — ver secção 5.7.

### 5.5 Outliers

Dois critérios robustos aplicados sobre o log-preço:
- **IQR de Tukey (1.5 × IQR):** 40 outliers (0,66%), limites [29, 6 823] HKD
- **MAD (z robusto > 3):** 28 outliers (0,46%), limites [21, 8 485] HKD

Três estratégias comparadas: sem tratamento, capping (winsorização nos limites
do IQR) e filtragem robusta (remoção da união IQR ∪ MAD).

**`minimum_nights` e `accommodates`** foram analisadas com os mesmos critérios.
No caso de `minimum_nights`, o MAD assinala 43% das observações por efeito da
distribuição bimodal (a premissa de unimodalidade não se verifica); ambas as
variáveis foram mantidas sem tratamento.

**Nota metodológica:** o tratamento de outliers é aplicado **apenas ao conjunto
de treino**. Winsorizar o alvo na validação e no teste produziria métricas
otimistas e não comparáveis entre estratégias.

### 5.6 Amenities

871 valores distintos, convertidos em:
- `n_amenities`: contagem total por alojamento.
- Uma **binária por cada uma das 15 amenities mais frequentes**.

A seleção das top 15 é feita sobre o dataset completo, antes do split. Como não
usa o preço, o risco de leakage é desprezável.

### 5.7 Multicolinearidade

Uma versão inicial incluía `room_type` e `property_type` em simultâneo. Uma
tabela de contingência mostrou redundância total (cada categoria de
`property_type` corresponde exclusivamente a um `room_type`), e os coeficientes
tinham sinais contraditórios com a análise exploratória. `property_type` foi
removida, reduzindo a matriz de 109 para 51 features.

### 5.8 Split, validação e prevenção de leakage

- Split **70% treino / 15% validação / 15% teste**, com `seed=42`.
- **K-fold cross-validation com 5 folds** para comparar modelos e selecionar
  hiperparâmetros.
- Todo o pré-processamento que aprende parâmetros a partir dos dados (medianas
  de imputação, limites de outliers, média e desvio para standardização,
  limiares da classe positiva) é calculado **apenas no treino** e aplicado aos
  restantes conjuntos. Nos k-folds, essa aprendizagem acontece dentro de cada
  fold.
- O conjunto de teste é usado **apenas para o reporte final**, nunca para tomar
  decisões.

### 5.9 Features finais (51)

- **Numéricas:** `accommodates`, `bedrooms`, `bathrooms`, `beds`,
  `minimum_nights`, `noites`, `n_amenities`, `long_stay`,
  `review_scores_rating`, `number_of_reviews`
- **Polinomiais:** `accommodates²`, `n_amenities²` (construídos **antes** da
  standardização)
- **Binárias:** `tem_reviews` + 15 amenities
- **Categóricas (one-hot):** `room_type`, `neighbourhood_cleansed`
- **Interações:** `long_stay × room_type`

---

## 6. Resultados

### Regressão

Configuração final: log-preço, polinomiais e interações, imputação simples, sem
tratamento de outliers, L2 = 0.01.

| Conjunto | MAE | RMSE | R² |
|---|---|---|---|
| Treino | 0.3441 | 0.5014 | 0.6808 |
| Validação | 0.3560 | 0.5107 | 0.6908 |
| **Teste** | **0.3609** | **0.4989** | **0.6746** |

Comparação de alvos e features (média de 5 folds):

| Modelo | R² |
|---|---|
| Preço bruto (HKD) | 0.188 |
| Log-preço | 0.660 |
| Log + polinomiais | 0.667 |
| Log + polinomiais + interações | 0.672 |

Regularização L2 (curva em U, ótimo em 0.01):

| λ | 0.000 | 0.001 | **0.010** | 0.100 | 1.000 |
|---|---|---|---|---|---|
| R² | 0.672 | 0.673 | **0.677** | 0.671 | 0.585 |

### Classificação

Classe positiva: top 20% de preço dentro de `room_type × bairro`, com limiares
aprendidos no treino.

| Métrica | Threshold 0.5 | Threshold 0.273 |
|---|---|---|
| Accuracy | 0.8282 | 0.7676 |
| Precision | 0.7632 | 0.4262 |
| Recall | 0.1648 | **0.5739** |
| F1 | 0.2710 | **0.4891** |
| ROC-AUC | 0.8043 | 0.8043 |

O threshold 0.273 foi selecionado por maximização do F1 na validação e aplicado
ao teste sem novo ajuste.

### Conclusões principais

**Decisões com maior impacto:**
1. Log-preço (R² de 0.188 para 0.660, e treino muito mais estável).
2. Remoção de `property_type` (multicolinearidade).
3. Duração como feature em vez de filtragem do dataset.
4. Ajuste do threshold de classificação (recall de 0.165 para 0.574).

**Decisões com pouco impacto:**
1. Tratamento de outliers — nenhuma das três estratégias produziu diferença
   relevante, provavelmente porque o log-preço já comprime os extremos.
2. Política de imputação — simples e agrupada são equivalentes, por as
   variáveis imputadas serem redundantes face a `accommodates`.

---

## 7. Limitações

- Várias colunas previstas no enunciado estão 100% vazias nesta versão do
  ficheiro (`host_response_rate`, `host_response_time`, `host_acceptance_rate`).
- Categorias com poucas observações (Hotel room: 15, Shared room: 116)
  apresentam erro elevado e viés sistemático; o modelo está calibrado para as
  categorias dominantes.
- As coordenadas geográficas não são usadas diretamente — o efeito espacial é
  captado apenas ao nível do bairro.
- Os coeficientes descrevem associações, não relações causais. O caso de
  `number_of_reviews` (coeficiente negativo em ambos os modelos) ilustra-o:
  a leitura plausível é que alojamentos mais baratos têm maior ocupação e
  acumulam mais reviews.
