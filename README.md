# Projeto 1 - Machine Learning: Aprendizagem Supervisionada

**Autor:** Bernardo Nabais
**Cadeira:** Machine Learning
**Curso:** Inteligência Artificial e Ciência de Dados, UBI
**Ano letivo:** 2026/2027

# 1. Objetivo

Construir um pipeline completo de aprendizagem supervisionada sobre listings do Airbnb, com dois objetivos:

- **Regressão:** prever o preço por noite a partir das características do alojamento.
- **Classificação:** identificar alojamentos "caros" (top 20% do preço dentro de grupos comparáveis).

O foco não é minimizar o erro, mas justificar cada decisão de pré-processamento e mostrar como diferentes estratégias afetam os resultados e a interpretação dos modelos.

## 2. Dataset
- Cidade: Hong Kong
- Fonte: https://insideairbnb.com/pt/get-the-data/
- Ficheiro: listings.csv.gz 
- Data do ficheiro: junho-julho de 2026 (last_scraped: 2026-06-28, 2026-06-29, 2026-07-03)
- Moeda do preço: HKD(dólares de Hong Kong)
- Nº de linhas e colunas: 6734 x 90

## 3. Estrutura do projeto
(a árvore de pastas, com uma linha a explicar cada ficheiro)

## 4. Como reproduzir


## 5. Decisões de pré-processamento

### 5.1 Alvo

- Alvo da regressão: coluna `price`, convertida de texto (`"$X.YZ"`) para número.
- Confirmado como preço por noite (`price = price_quote_price_per_night`).
- **Removidas 688 linhas sem preço** (10,2%), ficando com 6 046 alojamentos.

### 5.2 Duração da estadia simulada

- Decisão: **manter todas as durações** e usar a duração como feature.
- Justificação: a mediana varia fortemente entre grupos de duração (545 HKD em 1 noite vs 221 HKD em 28-31 noites), efeito do desconto por *long stay* e da regulação de Hong Kong (que exige licença para estadias com menos de 28 noites).
- Features criadas: `noites` (numérica) e `long_stay` (binária, `noites >= 28`).

### 5.3 Outliers

- Aplicados dois critérios robustos sobre o log-preço:
  - **IQR (Tukey, 1,5 × IQR):** 40 outliers (0,7%)
  - **MAD (z robusto > 3):** 28 outliers (0,46%)
- **Filtragem robusta:** união dos dois critérios.
- **Capping/winsorização:** limites do IQR, [29 HKD, 6 823 HKD].
- `minimum_nights` não tratada como outlier (máximo 365 é plausível; a distribuição bimodal reflete o mercado real).
- `accommodates` sem outliers detetáveis.

### 5.4 Missing values

Distinção entre valores desconhecidos e ausência com significado:

**Ausência com significado:**
- Colunas de reviews (~45%): criada feature `tem_reviews`; scores nulos preenchidos com a mediana dos alojamentos com reviews.

**Valor desconhecido - duas políticas comparadas:**
- `bedrooms` (44,8%): mediana global (simples) vs mediana por grupo `room_type × accommodates` (agrupada).
- `bathrooms` (11%): valor extraído primeiro de `bathrooms_text` (0,3% de missing); resto imputado com as duas políticas.
- `beds` (4,9%): mesmas duas políticas.

**Eliminadas:**
- 14 colunas 100% vazias, incluindo `host_response_rate`, `host_response_time`, `host_acceptance_rate`, `neighbourhood`, `license` e `instant_bookable` (limitação do ficheiro desta versão).
- `host_about`, `host_location`, `description`, `picture_url` (sem valor preditivo direto para o preço do alojamento).

### 5.5 Amenities

- 871 amenities distintas convertidas em:
  - **11 features de contagem por categoria** (categorias exclusivas, prioridade para luxo): `luxo`, `vista_exterior`, `cozinha_completa`, `familia`, `seguranca`, `acessibilidade`, `estacionamento`, `servicos_host`, `tecnologia`, `basicas`, `outras`.
  - **6 binárias de luxo específicas:** `tem_piscina`, `tem_jacuzzi`, `tem_gym`, `tem_doorman`, `tem_vista_mar`, `tem_vista_cidade`.
- Feature-resumo: `n_amenities` (total por alojamento).

### 5.6 Fugas de informação

Excluídas das features por serem derivadas do alvo:
- `price_quote_total_price`, `price_quote_price_per_night`, `estimated_revenue_l365d`.

### 5.7 Split e validação

- Split treino/validação/teste com seed fixa (a definir na implementação).
- Toda a aprendizagem de parâmetros (médias para imputação, escalas, limites de capping) é feita **apenas no conjunto de treino** e aplicada aos restantes, para evitar *data leakage*.
- K-fold cross-validation para escolha de hiperparâmetros.

---


## 6. Resultados principais
(tabelas e conclusões finais, remetendo para o relatório)

## 7. Limitações

- O ficheiro desta versão de Hong Kong tem várias colunas 100% vazias que o enunciado esperava utilizar (`host_response_rate`, `host_response_time`, `neighbourhood_group_cleansed`, entre outras).
- A concentração de anúncios em poucos bairros (~80% em 3 bairros) e a raridade de algumas categorias (Hotel room com 15 anúncios, Shared room com 116) limitam o que o modelo pode aprender sobre essas subpopulações.
