# Engenharia de Features Causais para Trading Quantitativo

**Arsenal de Features para Modelos ML Tabulares em Séries Temporais Financeiras OHLCV**

*Research Assistant em Quantitative Trading*  
*Janeiro 2026*

---

## Sumário

1. [Introdução e Fundamentos](#1-introdução-e-fundamentos)
2. [Framework de Validação de Causalidade](#2-framework-de-validação-de-causalidade)
3. [Features de Estatística e Econometria](#3-features-de-estatística-e-econometria)
4. [Features de Física Estatística e Sistemas Dinâmicos](#4-features-de-física-estatística-e-sistemas-dinâmicos)
5. [Features de Análise de Séries Temporais](#5-features-de-análise-de-séries-temporais)
6. [Features de Microestrutura de Mercado](#6-features-de-microestrutura-de-mercado)
7. [Implementações Python](#7-implementações-python)
8. [Rankings e Recomendações](#8-rankings-e-recomendações)
9. [Considerações de Deploy](#9-considerações-de-deploy)
10. [Referências](#referências)

---

## 1. Introdução e Fundamentos

Este documento apresenta um arsenal completo de **features causais** para modelos de machine learning tabulares aplicados a trading quantitativo. Todas as features são estritamente causais, calculadas apenas com informações disponíveis até o tempo *t*, eliminando completamente vazamentos temporais (lookahead bias), repaint e normalizações globais problemáticas.

### 1.1 Princípios Fundamentais

> **Restrições Obrigatórias:**
> - **Causalidade estrita:** Features calculadas apenas com dados até tempo t
> - **Proibição de repaint:** Valores passados não podem ser alterados
> - **Sem lookahead bias:** Nenhuma informação futura pode ser utilizada
> - **Sem normalizações globais:** Estatísticas devem ser rolling/expanding
> - **Adequação para ML tabular:** Features vetoriais, não sequenciais

### 1.2 Tipos de Dados

As features são projetadas para operar sobre:
- **OHLCV:** Open, High, Low, Close, Volume
- **Timeframes arbitrários:** Intraday ou diário
- **Retornos e preços:** Transformações permitidas desde que causais

---

## 2. Framework de Validação de Causalidade

Antes de apresentar as features, estabelecemos um framework rigoroso para validação de causalidade[1].

### 2.1 Teste de Causalidade

#### Critérios de Validação

Para cada feature, verificamos:
1. **Dependência temporal:** A feature em t depende apenas de dados em t' ≤ t?
2. **Estabilidade histórica:** Valores passados permanecem constantes?
3. **Janela de cálculo:** Usa janela rolling/expanding, nunca janela futura?
4. **Normalização:** Estatísticas de normalização são causais?

### 2.2 Exemplos de Violação

> **⚠️ Proibido (Viola Causalidade):**
> ```python
> z_score = (x - mean(x)) / std(x)  # usa toda a série
> rank = pd.rank()                   # requer conhecimento global
> percentile = percentileofscore()   # depende de dados futuros
> ema = EMA(close, 20)               # com cálculo recursivo infinito
> ```

> **✅ Permitido (Causal):**
> ```python
> z_score = (x - rolling_mean(x, 60)) / rolling_std(x, 60)
> rank = rolling_rank(x, 60)         # rank dentro da janela
> ema = EMA(close, 20)               # com inicialização finita
> ```

---

## 3. Features de Estatística e Econometria

### 3.1 Momentos Estatísticos (Classe S)

#### 3.1.1 Realized Volatility (RV) — **RANK S**

**O que mede:** Volatilidade realizada intraday, estimador não-viesado da variância integrada[2].

**Por que funciona:** Volatilidade clusterizada (efeito ARCH). Períodos de alta volatilidade tendem a persistir. Preditor clássico de risco e retornos futuros.

**Regimes:** Funciona melhor em regimes de alta volatilidade, crises, e mercados emergentes. Perde poder em mercados muito eficientes.

##### Formulação Matemática

$$RV_t = \sum_{i=1}^{M} r_{t,i}^2$$

Onde $r_{t,i} = \ln(P_{t,i}) - \ln(P_{t,i-1})$ são os retornos intraday.

##### Validação de Causalidade
- ✅ **Causal:** Usa apenas retornos do dia t
- ✅ **Sem repaint:** Valor não muda após cálculo
- ✅ **Sem lookahead:** Janela fixa, não expande para futuro
- ⚠️ **Efeitos de borda:** Nenhum para dados intraday completos

---

#### 3.1.2 Realized Skewness e Kurtosis — **RANK A**

**O que mede:** Assimetria (skewness) e caudas (kurtosis) da distribuição de retornos intraday[3].

$$RS_t = \frac{\sqrt{M} \sum_{i=1}^{M} r_{t,i}^3}{RV_t^{3/2}}$$

$$RK_t = \frac{M \sum_{i=1}^{M} r_{t,i}^4}{RV_t^2}$$

**Sinal preditivo:** Skewness negativa prediz retornos futuros mais altos (premiação por risco de crash). Kurtosis alta indica risco de eventos extremos.

---

### 3.2 Features de Distribuição (Classe A)

#### 3.2.1 Percentis de Retorno — **RANK A**

**O que mede:** Posição relativa do retorno atual dentro da distribuição histórica recente.

$$Pct_t = \frac{\sum_{i=t-N}^{t} \mathbb{1}_{[r_i < r_t]}}{N} \times 100$$

**Intuição:** Retornos extremos (percentis 0-10 ou 90-100) tendem a reverter (mean-reversion) ou acelerar (momentum), dependendo do timeframe.

---

#### 3.2.2 Value at Risk (VaR) Rolling — **RANK B**

$$VaR_{t,\alpha} = -\text{percentile}(\{r_{t-N}, ..., r_t\}, \alpha \times 100)$$

> **Nota:** VaR histórico é causal. VaR paramétrico que estima parâmetros da distribuição em toda a série não é.

---

### 3.3 Dependência Temporal (Classe A)

#### 3.3.1 Autocorrelação de Retornos — **RANK A**

**O que mede:** Correlação serial dos retornos, indicador de previsibilidade[4].

$$\rho_{t,k} = \frac{\sum_{i=t-N+k}^{t} (r_i - \bar{r})(r_{i-k} - \bar{r})}{\sum_{i=t-N}^{t} (r_i - \bar{r})^2}$$

**Regimes:** Autocorrelação positiva em curtos prazos (momentum intraday). Negativa em longos prazos (mean-reversion).

---

#### 3.3.2 Autocorrelação de Retornos ao Quadrado — **RANK S**

**O que mede:** Persistência da volatilidade (efeito ARCH/GARCH).

$$\rho_{t,k}^{(2)} = \text{Corr}(r_t^2, r_{t-k}^2)$$

> **Por que é Rank S:** Volatilidade clusterizada é um dos fenômenos mais robustos em finanças. Alta autocorrelação de quadrados prediz continuação da volatilidade.

---

## 4. Features de Física Estatística e Sistemas Dinâmicos

### 4.1 Entropia e Informação (Classe S)

#### 4.1.1 Shannon Entropy de Retornos — **RANK S**

**O que mede:** Desordem/incerteza na distribuição de retornos. Quanto maior a entropia, mais "aleatória" a série[5].

$$H_t = -\sum_{j=1}^{B} p_{j,t} \log_2(p_{j,t})$$

Onde $p_{j,t}$ é a probabilidade empírica do retorno cair no bin $j$ na janela $[t-N, t]$.

**Sinal preditivo:** Baixa entropia indica regimes previsíveis (tendência forte). Alta entropia indica regime de ruído (range-bound).

---

#### 4.1.2 Permutation Entropy — **RANK A**

**O que mede:** Complexidade da série temporal baseada em padrões ordinais[6].

$$PE_t = -\sum_{\pi} p(\pi) \log_2 p(\pi)$$

Onde $\pi$ representa padrões ordinais de ordem $m$ (embedding dimension).

**Vantagem:** Não requer binning, é robusta a outliers, captura dinâmicas não-lineares.

---

#### 4.1.3 Transfer Entropy (Causalidade de Granger Não-Linear) — **RANK B**

$$TE_{Y \rightarrow X} = \sum p(x_{t+1}, x_t^{(k)}, y_t^{(l)}) \log \frac{p(x_{t+1}|x_t^{(k)}, y_t^{(l)})}{p(x_{t+1}|x_t^{(k)})}$$

**Aplicação:** Mede fluxo de informação entre ativos (leading-lag relationships).

---

### 4.2 Exponentes de Hurst e Memória Longa (Classe A)

#### 4.2.1 Hurst Exponent (R/S Analysis) — **RANK A**

**O que mede:** Grau de persistência ou mean-reversion na série temporal[7].

$$E[R_n/S_n] = C \times n^H$$

Onde $R_n$ é o range e $S_n$ é o desvio padrão da série.

| Valor de H | Interpretação | Regime de Trading |
|------------|---------------|-------------------|
| H < 0.5 | Mean-reversion (anti-persistência) | Reversão à média |
| H = 0.5 | Random walk (mercado eficiente) | Nenhuma vantagem |
| H > 0.5 | Tendência (persistência) | Momentum/Trend-following |

---

#### 4.2.2 Detrended Fluctuation Analysis (DFA) — **RANK A**

**Vantagem sobre R/S:** Mais robusto a tendências determinísticas na série.

$$F(n) = \sqrt{\frac{1}{N} \sum_{t=1}^{N} [y(t) - y_n(t)]^2} \sim n^\alpha$$

---

### 4.3 Dimensão Fractal (Classe B)

#### 4.3.1 Fractal Dimension (Box-Counting) — **RANK B**

**O que mede:** "Rugosidade" da série temporal. Dimensão próxima de 1 indica tendência suave. Próxima de 2 indica comportamento aleatório[8].

---

## 5. Features de Análise de Séries Temporais

### 5.1 Medidas de Persistência (Classe S)

#### 5.1.1 Variance Ratio — **RANK S**

**O que mede:** Testa a hipótese de random walk. VR < 1 indica mean-reversion, VR > 1 indica momentum.

$$VR(k) = \frac{Var(r_t^{(k)})}{k \times Var(r_t^{(1)})}$$

Onde $r_t^{(k)} = \sum_{i=0}^{k-1} r_{t-i}$ é o retorno de k períodos.

---

#### 5.1.2 Ljung-Box Estatística (Rolling) — **RANK A**

$$Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}$$

**Uso:** Q alto indica autocorrelação significativa (previsibilidade).

---

### 5.2 Features de Mudança de Regime (Classe A)

#### 5.2.1 Rolling Z-Score de Volatilidade — **RANK A**

$$Z_t^{(vol)} = \frac{RV_t - \overline{RV}_{t-N:t}}{\sigma_{RV,t-N:t}}$$

**Interpretação:** Z > 2 indica regime de alta volatilidade. Z < -2 indica regime de baixa volatilidade.

---

#### 5.2.2 Chamfer Distance (Detecção de Anomalias) — **RANK B**

**O que mede:** Distância entre o padrão recente e padrões históricos conhecidos (crashes, rallies).

---

### 5.3 Decomposição de Sinais (Classe B)

#### 5.3.1 Hodrick-Prescott Filter (Rolling) — **RANK B**

**O que mede:** Decompõe a série em tendência (ciclo de longo prazo) e ciclo (componente de curto prazo).

> **⚠️ Atenção:** HP filter tradicional usa toda a série (não-causal). Versão rolling usa apenas janela recente.

---

## 6. Features de Microestrutura de Mercado

### 6.1 Features de Volume (Classe S)

#### 6.1.1 Volume-Weighted Average Price (VWAP) Deviation — **RANK S**

**O que mede:** Desvio do preço atual em relação ao preço médio ponderado por volume[9].

$$VWAP_t = \frac{\sum_{i=t-N}^{t} P_i \times V_i}{\sum_{i=t-N}^{t} V_i}$$

$$Dev_t = \frac{P_t - VWAP_t}{VWAP_t} \times 100$$

**Sinal:** Preço acima do VWAP indica pressão compradora. Desvios extremos tendem a reverter.

---

#### 6.1.2 On-Balance Volume (OBV) Momentum — **RANK A**

$$OBV_t = OBV_{t-1} + \text{sgn}(P_t - P_{t-1}) \times V_t$$

**Feature:** ROC de OBV em janela N.

---

### 6.2 Features de Ordem e Liquidez (Classe A)

#### 6.2.1 Amihud Illiquidity Ratio — **RANK A**

**O que mede:** Impacto de preço por unidade de volume. Proxy para liquidez[10].

$$ILLIQ_t = \frac{1}{N} \sum_{i=t-N}^{t} \frac{|r_i|}{V_i \times P_i}$$

**Interpretação:** Valor alto = ilíquido. Valor baixo = líquido.

---

#### 6.2.2 Kyle's Lambda (Proxy) — **RANK B**

**O que mede:** Sensibilidade do preço ao fluxo de ordens.

$$\lambda_t = \frac{|\Delta P_t|}{V_t}$$

---

### 6.3 Features de Pressão de Ordem (Classe S)

#### 6.3.1 Order Flow Imbalance (Proxy) — **RANK S**

**O que mede:** Desequilíbrio entre pressão compradora e vendedora[11].

$$OFI_t = \frac{(Close_t - Open_t) \times Volume_t}{High_t - Low_t}$$

**Intuição:** Usa a posição do close no range do dia como proxy para direção do fluxo.

---

#### 6.3.2 VPIN (Volume-Synchronized Probability of Informed Trading) — **RANK A**

**O que mede:** Probabilidade de trading informado, preditor de toxicidade do fluxo[12].

$$VPIN_t = \frac{\sum_{i=t-N}^{t} |V_i^B - V_i^S|}{\sum_{i=t-N}^{t} (V_i^B + V_i^S)}$$

**Proxy:** $V^B - V^S \approx \text{sgn}(Close - Open) \times Volume$ (classificação de tick).

---

## 7. Implementações Python

### 7.1 Classe Base de Features

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple

class CausalFeatureEngine:
    """
    Motor de features causais para trading quantitativo.

    Todas as features são estritamente causais:
    - Calculadas apenas com dados até tempo t
    - Sem lookahead bias
    - Sem repaint
    """

    def __init__(self, max_lookback: int = 252):
        self.max_lookback = max_lookback

    def _validate_causal(self, series: pd.Series, window: int) -> bool:
        """Valida se a janela é causal."""
        return window <= len(series) and window <= self.max_lookback
```

---

### 7.2 Features de Volatilidade

```python
def realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Volatilidade realizada (soma de quadrados).

    Args:
        returns: Série de retornos
        window: Janela de cálculo

    Returns:
        Série de volatilidade realizada
    """
    return returns.rolling(window=window).apply(
        lambda x: np.sqrt(np.sum(x**2)), raw=True
    )

def realized_skewness(self, returns: pd.Series, window: int = 20) -> pd.Series:
    """Skewness realizada intraday."""
    rv = self.realized_volatility(returns, window)
    m3 = returns.rolling(window=window).apply(
        lambda x: np.sum(x**3), raw=True
    )
    return np.sqrt(window) * m3 / (rv**1.5)

def realized_kurtosis(self, returns: pd.Series, window: int = 20) -> pd.Series:
    """Kurtosis realizada intraday."""
    rv = self.realized_volatility(returns, window)
    m4 = returns.rolling(window=window).apply(
        lambda x: np.sum(x**4), raw=True
    )
    return window * m4 / (rv**2)

def parkinson_volatility(self, high: pd.Series, low: pd.Series, 
                        window: int = 20) -> pd.Series:
    """
    Estimador de Parkinson (usa high-low).
    Mais eficiente que close-to-close.
    """
    log_hl = np.log(high / low)
    return np.sqrt(
        log_hl.rolling(window=window).mean() / (4 * np.log(2))
    )

def garman_klass_volatility(self, open_: pd.Series, high: pd.Series,
                             low: pd.Series, close: pd.Series,
                             window: int = 20) -> pd.Series:
    """
    Estimador de Garman-Klass (usa OHLC).
    Estimador eficiente de variância.
    """
    log_hl = np.log(high / low)**2
    log_co = np.log(close / open_)**2

    return np.sqrt(
        0.5 * log_hl.rolling(window=window).mean() - 
        (2 * np.log(2) - 1) * log_co.rolling(window=window).mean()
    )
```

---

### 7.3 Features de Entropia

```python
def shannon_entropy(self, returns: pd.Series, window: int = 60,
                   bins: int = 10) -> pd.Series:
    """
    Entropia de Shannon rolling.

    Args:
        returns: Série de retornos
        window: Janela de cálculo
        bins: Número de bins para discretização
    """
    def calc_entropy(x):
        hist, _ = np.histogram(x, bins=bins, density=True)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs))

    return returns.rolling(window=window).apply(calc_entropy, raw=True)

def permutation_entropy(self, series: pd.Series, window: int = 60,
                       order: int = 3) -> pd.Series:
    """
    Permutation Entropy (Bandt & Pompe).

    Captura complexidade da série temporal.
    """
    def calc_pe(x):
        # Cria padrões ordinais
        patterns = []
        for i in range(len(x) - order + 1):
            pattern = tuple(np.argsort(x[i:i+order]))
            patterns.append(pattern)

        # Conta frequências
        unique, counts = np.unique(patterns, return_counts=True)
        probs = counts / len(patterns)

        return -np.sum(probs * np.log2(probs))

    return series.rolling(window=window).apply(calc_pe, raw=True)
```

---

### 7.4 Features de Hurst e Memória Longa

```python
def hurst_exponent(self, series: pd.Series, window: int = 100,
                  max_lag: int = 20) -> pd.Series:
    """
    Hurst Exponent via R/S Analysis (Rolling).

    H < 0.5: Mean-reversion
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    def calc_hurst(x):
        lags = range(2, min(max_lag, len(x)//4))
        rs_values = []

        for lag in lags:
            # Divide em chunks
            n_chunks = len(x) // lag
            rs_chunks = []

            for i in range(n_chunks):
                chunk = x[i*lag:(i+1)*lag]
                mean_chunk = np.mean(chunk)
                std_chunk = np.std(chunk)

                if std_chunk == 0:
                    continue

                # Calcula R/S
                cumdev = np.cumsum(chunk - mean_chunk)
                r = np.max(cumdev) - np.min(cumdev)
                rs_chunks.append(r / std_chunk)

            if rs_chunks:
                rs_values.append(np.mean(rs_chunks))

        # Regressão log-log
        if len(rs_values) < 3:
            return 0.5

        log_lags = np.log(list(lags)[:len(rs_values)])
        log_rs = np.log(rs_values)

        slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
        return slope

    return series.rolling(window=window).apply(calc_hurst, raw=True)

def dfa_alpha(self, series: pd.Series, window: int = 100,
              scales: Optional[list] = None) -> pd.Series:
    """
    Detrended Fluctuation Analysis (Rolling).

    Mais robusto que R/S para séries com tendência.
    """
    if scales is None:
        scales = [4, 8, 16, 32]

    def calc_dfa(x):
        fluctuations = []

        for scale in scales:
            if scale >= len(x):
                continue

            n_segments = len(x) // scale
            f_scale = []

            for i in range(n_segments):
                segment = x[i*scale:(i+1)*scale]

                # Ajusta tendência linear
                t = np.arange(len(segment))
                slope, intercept, _, _, _ = stats.linregress(t, segment)
                trend = intercept + slope * t

                # Calcula desvio
                f_scale.append(np.sqrt(np.mean((segment - trend)**2)))

            if f_scale:
                fluctuations.append(np.mean(f_scale))

        # Regressão log-log
        if len(fluctuations) < 2:
            return 0.5

        log_scales = np.log([s for s in scales if s < len(x)])[:len(fluctuations)]
        log_f = np.log(fluctuations)

        slope, _, _, _, _ = stats.linregress(log_scales, log_f)
        return slope

    return series.rolling(window=window).apply(calc_dfa, raw=True)
```

---

### 7.5 Features de Microestrutura

```python
def vwap_deviation(self, close: pd.Series, volume: pd.Series,
                  window: int = 20) -> pd.Series:
    """
    Desvio percentual do VWAP.
    """
    tp = close  # Typical price simplificado
    vwap = (tp * volume).rolling(window=window).sum() / \
           volume.rolling(window=window).sum()

    return (close - vwap) / vwap * 100

def amihud_illiquidity(self, returns: pd.Series, close: pd.Series,
                      volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Ratio de Amihud (proxy para illiquidez).
    """
    return (np.abs(returns) / (volume * close)).rolling(window=window).mean()

def order_flow_imbalance(self, open_: pd.Series, high: pd.Series,
                        low: pd.Series, close: pd.Series,
                        volume: pd.Series) -> pd.Series:
    """
    Proxy para order flow imbalance.

    Usa posição do close no range como proxy para direção.
    """
    # Classificação de tick proxy
    buy_volume = volume * (close >= open_).astype(float)
    sell_volume = volume * (close < open_).astype(float)

    return (buy_volume - sell_volume) / (buy_volume + sell_volume)

def vpin_proxy(self, open_: pd.Series, close: pd.Series,
              volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Proxy para VPIN (Volume-Synchronized PIN).
    """
    signed_volume = volume * np.sign(close - open_)
    return signed_volume.abs().rolling(window=window).sum() / \
           volume.rolling(window=window).sum()
```

---

### 7.6 Features de Autocorrelação

```python
def rolling_autocorr(self, series: pd.Series, lag: int = 1,
                    window: int = 60) -> pd.Series:
    """
    Autocorrelação rolling.
    """
    return series.rolling(window=window).apply(
        lambda x: x.autocorr(lag=lag), raw=False
    )

def volatility_clustering(self, returns: pd.Series, lag: int = 1,
                         window: int = 60) -> pd.Series:
    """
    Autocorrelação de retornos ao quadrado (efeito ARCH).
    """
    return self.rolling_autocorr(returns**2, lag=lag, window=window)

def variance_ratio(self, returns: pd.Series, k: int = 5,
                  window: int = 60) -> pd.Series:
    """
    Variance Ratio (Lo & MacKinlay).

    VR < 1: Mean-reversion
    VR = 1: Random walk
    VR > 1: Momentum
    """
    var_1 = returns.rolling(window=window).var()

    # Retornos de k períodos
    returns_k = returns.rolling(window=k).sum()
    var_k = returns_k.rolling(window=window-k+1).var()

    return var_k / (k * var_1)
```

---

### 7.7 Features Técnicas Clássicas (Causais)

```python
def rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (causal).
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(self, close: pd.Series, fast: int = 12, slow: int = 26,
        signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (causal com inicialização exponencial).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def bollinger_bands(self, close: pd.Series, window: int = 20,
                   num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bandas de Bollinger (rolling, causais).
    """
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()

    upper = middle + num_std * std
    lower = middle - num_std * std

    return upper, middle, lower

def rolling_percentile(self, series: pd.Series, window: int = 60) -> pd.Series:
    """
    Percentil rolling do valor atual na janela.
    """
    def calc_percentile(x):
        return stats.percentileofscore(x[:-1], x[-1])

    return series.rolling(window=window).apply(calc_percentile, raw=True)
```

---

## 8. Rankings e Recomendações

### 8.1 Ranking Geral por Classe

| Rank | Feature | Categoria | Justificativa |
|------|---------|-----------|---------------|
| **S** | Realized Volatility | Econometria | Predictor robusto de risco, efeito ARCH comprovado |
| **S** | Volatility Clustering (ρ²) | Econometria | Fenômeno mais robusto em finanças |
| **S** | Shannon Entropy | Física Estatística | Captura regime de mercado, não-linear |
| **S** | VWAP Deviation | Microestrutura | Proxy robusto para pressão de ordem |
| **S** | Variance Ratio | Séries Temporais | Teste direto de previsibilidade |
| **A** | Realized Skewness | Econometria | Premiação por risco de crash |
| **A** | Hurst Exponent | Física Estatística | Distingue momentum vs mean-reversion |
| **A** | Permutation Entropy | Física Estatística | Robusta a outliers, captura não-linearidades |
| **A** | Order Flow Imbalance | Microestrutura | Leading indicator de pressão |
| **A** | Amihud Illiquidity | Microestrutura | Proxy clássico de liquidez |
| **B** | Realized Kurtosis | Econometria | Útil mas ruidosa |
| **B** | Fractal Dimension | Física Estatística | Interessante mas computacionalmente custosa |
| **B** | HP Filter (Rolling) | Séries Temporais | Útil para decomposição |

---

### 8.2 Matriz de Regimes

| Regime de Mercado | Features Prioritárias | Estratégia |
|-------------------|----------------------|------------|
| Alta volatilidade (crise) | RV, Entropia, VPIN | Reduzir exposição, proteção |
| Baixa volatilidade (calm) | Variance Ratio, Hurst | Carry, mean-reversion |
| Tendência forte | Hurst > 0.5, VWAP, OBV | Momentum, trend-following |
| Range-bound | Hurst < 0.5, Bollinger | Mean-reversion |
| Alta entropia | Evitar sinais, reduzir tamanho | Esperar clareza |
| Baixa entropia | Seguir tendência identificada | Aumentar convicção |

---

### 8.3 Recomendações de Feature Engineering

> **Pipeline Recomendado:**
> 1. **Base:** RV, VWAP Deviation, Rolling Autocorr (S)
> 2. **Expansão:** Adicionar Entropia, Hurst, Variance Ratio (A)
> 3. **Especialização:** Adicionar features específicas do ativo/mercado
> 4. **Seleção:** Usar importância de feature (SHAP, permutation) para remover redundantes

---

## 9. Considerações de Deploy

### 9.1 Validação Temporal

> **⚠️ Regras de Validação:**
> - Usar **Time Series Split** (nunca shuffle)
> - Embargo entre treino e teste (evitar overlap de labels)
> - Purging de eventos importantes (evitar vazamento de informação)
> - Validação em múltiplos regimes de mercado

---

### 9.2 Checklist de Produção

- ✅ Todas as features calculadas com dados até t-1 (lag explícito)
- ✅ Janelas de rolling definidas e documentadas
- ✅ Tratamento de valores iniciais (warm-up period)
- ✅ Testes de causalidade automatizados
- ✅ Monitoramento de drift de features
- ✅ Fallback para valores missing (forward fill apropriado)

---

### 9.3 Efeitos de Borda e Mitigação

| Problema | Mitigação |
|----------|-----------|
| Janela inicial insuficiente | Período de warm-up (não operar) |
| Lookahead em EMA | Usar adjust=False, inicialização com SMA |
| Binning em entropia | Usar bins fixos (não adaptativos) |
| Estabilidade de Hurst | Janela mínima de 100 observações |

---

## Referências

1. Oliveira, D. C., et al. (2024). Causality-inspired models for financial time series forecasting. *arXiv preprint arXiv:2408.09960*.

2. Caporin, M. (2023). The role of jumps in realized volatility modeling and forecasting. *Journal of Financial Econometrics*, 21(4), 1143-1180.

3. Chen, X., Li, B., & Worthington, A. C. (2021). Higher moments and US industry returns: realized skewness and kurtosis. *Review of Accounting and Finance*, 20(4), 500-520.

4. Schadner, W. (2022). Expected Return Auto-Correlation: Believes, Efficiency and Meltdowns. *SSRN Electronic Journal*.

5. Liu, A., Chen, J., Yang, S. Y., & Hawkes, A. G. (2020). The flow of information in trading: An entropy approach to market regimes. *Entropy*, 22(9), 1064.

6. Gupta, R., Gupta, S., Singh, J., & Kais, S. (2025). Entropy-Assisted Quality Pattern Identification in Finance. *Entropy*, 27(4), 430.

7. Bui, Q., & Slepaczuk, R. (2022). Applying Hurst Exponent in pair trading strategies on Nasdaq 100 index. *Physica A: Statistical Mechanics and its Applications*, 589, 126574.

8. Dawi, N. B. M., Matejicek, M., & Maresova, P. (2025). Unraveling financial market dynamics: The application of fractal theory in financial time series analysis. *Fractals*, 33(1).

9. Lehalle, C. A., & Laruelle, S. (2018). *Market microstructure in practice*. World Scientific.

10. Kerr, J., Sadka, G., & Sadka, R. (2020). Illiquidity and price informativeness. *Management Science*, 66(8), 3412-3428.

11. Ju, G., Kim, K. K., & Lim, D. Y. (2019). Learning multi-market microstructure from order book data. *Quantitative Finance*, 19(9), 1497-1513.

12. Easley, D., de Prado, M. L., & O'Hara, M. (2019). The exchange of flow toxicity. *Journal of Trading*, 14(2), 21-37.

---

*Documento gerado em Janeiro 2026*
