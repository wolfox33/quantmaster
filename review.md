# Review do Projeto Quantmaster

## 1. Visão Geral do Projeto

**Quantmaster** é uma biblioteca Python de features quantitativas para adicionar colunas em `pandas.DataFrame` com dados OHLCV. A biblioteca segue um padrão funcional onde cada feature é uma função pura que:

- Recebe `pd.DataFrame` (OHLCV) ou `pd.Series`
- Retorna `pd.Series` ou `pd.DataFrame` alinhado ao índice original
- Não modifica dados in-place
- Usa type hints e segue PEP-8

### Estrutura Atual

```
src/quantmaster/features/
├── __init__.py          # Exports públicos
├── momentum.py          # RSI
├── volatility.py        # HAR-RV, Yang-Zhang, Realized Variance
├── statistical.py       # Fracdiff, Hurst DFA, Ornstein-Uhlenbeck
├── volume.py            # RVOL (Relative Volume)
└── utils.py             # Helpers de validação
```

---

## 2. Features Existentes

| Feature | Categoria | Descrição |
|---------|-----------|-----------|
| `rsi` | Momentum | Relative Strength Index com EWM |
| `realized_variance` | Volatility | Variância realizada (retornos²) |
| `har_rv` | Volatility | HAR-RV com componentes diário/semanal/mensal |
| `har_rv_forecast` | Volatility | Previsão HAR-RV com regressão rolling |
| `yang_zhang_volatility` | Volatility | Estimador Yang-Zhang (OHLC) |
| `fracdiff` | Statistical | Diferenciação fracionária (preserva memória) |
| `hurst_dfa` | Statistical | Expoente de Hurst via DFA |
| `ornstein_uhlenbeck` | Statistical | Parâmetros OU para mean-reversion |
| `rvol` | Volume | Volume relativo (log do ratio) |

### Pontos Fortes

1. **Foco em features acadêmicas sólidas** - HAR-RV, Hurst DFA, OU são bem fundamentados na literatura
2. **Diferenciação fracionária** - Técnica de Marcos Lopez de Prado para preservar memória
3. **Yang-Zhang** - Estimador de volatilidade eficiente usando OHLC
4. **API consistente** - Todas as funções seguem o mesmo padrão

### Lacunas Identificadas

- Poucos estimadores de volatilidade baseados em range (Parkinson, Garman-Klass, Rogers-Satchell)
- Sem features de microestrutura/liquidez
- Sem detecção de jumps/descontinuidades
- Sem medidas de entropia/complexidade
- Sem decomposição de variância (semivariance, signed jumps)
- Sem features de regime/structural breaks

---

## 3. Hipóteses de Novas Features

Baseado em pesquisa de artigos acadêmicos, literatura quant (WorldQuant, Marcos Lopez de Prado, papers de microestrutura) e práticas de hedge funds, apresento features que:

- **Não existem em TA-Lib/ta/pandas-ta**
- **São sólidas historicamente na literatura**
- **Podem ser calculadas a partir de OHLCV**
- **Se enquadram no estilo da biblioteca**

---

### 3.1 Volatility (Range-Based Estimators)

#### 3.1.1 `parkinson_volatility`
**Referência:** Parkinson (1980) - "The Extreme Value Method for Estimating the Variance of the Rate of Return"

Estimador de volatilidade usando apenas High-Low, 5x mais eficiente que close-to-close.

```python
def parkinson_volatility(
    data: pd.DataFrame,
    *,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    # σ² = (1/4ln2) * E[(ln(H/L))²]
```

**Por que incluir:** Complementa Yang-Zhang; base para outros estimadores; usado em produção por quants.

---

#### 3.1.2 `garman_klass_volatility`
**Referência:** Garman & Klass (1980) - "On the Estimation of Security Price Volatilities from Historical Data"

Combina OHLC para estimativa mais eficiente que Parkinson.

```python
def garman_klass_volatility(
    data: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.Series:
    # σ² = 0.5*(ln(H/L))² - (2ln2-1)*(ln(C/O))²
```

**Por que incluir:** 8x mais eficiente que close-to-close; amplamente citado na literatura.

---

#### 3.1.3 `rogers_satchell_volatility`
**Referência:** Rogers & Satchell (1991) - "Estimating Variance From High, Low and Closing Prices"

Único estimador range-based que funciona com drift (tendência).

```python
def rogers_satchell_volatility(
    data: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.Series:
    # σ² = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
```

**Por que incluir:** Robusto a drift; componente do Yang-Zhang; usado quando há tendência clara.

---

### 3.2 Microstructure & Liquidity

#### 3.2.1 `amihud_illiquidity`
**Referência:** Amihud (2002) - "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects"

Proxy de price impact: |return| / dollar_volume.

```python
def amihud_illiquidity(
    data: pd.DataFrame,
    *,
    window: int = 20,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    # ILLIQ = mean(|r_t| / DVOL_t)
```

**Por que incluir:** Um dos proxies de liquidez mais citados (>10k citações); prediz retornos futuros; calculável com OHLCV.

---

#### 3.2.2 `roll_spread`
**Referência:** Roll (1984) - "A Simple Implicit Measure of the Effective Bid-Ask Spread"

Estima bid-ask spread implícito via autocovariância negativa de retornos.

```python
def roll_spread(
    data: pd.DataFrame,
    *,
    window: int = 20,
    price_col: str = "close",
) -> pd.Series:
    # spread = 2 * sqrt(-cov(r_t, r_{t-1})) se cov < 0, else 0
```

**Por que incluir:** Estimador clássico de microestrutura; base para variantes modernas; usado quando não há dados de bid-ask.

---

#### 3.2.3 `corwin_schultz_spread`
**Referência:** Corwin & Schultz (2012) - "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices"

Estima spread usando High-Low de 1 e 2 dias consecutivos.

```python
def corwin_schultz_spread(
    data: pd.DataFrame,
    *,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    # S = 2*(exp(α) - 1) / (1 + exp(α))
    # α baseado em ratio de high-low de 1 e 2 dias
```

**Por que incluir:** Superior ao Roll em mercados menos líquidos; usa apenas OHLC; publicado em top journal (RFS).

---

### 3.3 Jump Detection & Decomposition

#### 3.3.1 `bipower_variation`
**Referência:** Barndorff-Nielsen & Shephard (2004) - "Power and Bipower Variation"

Estimador de variância contínua robusto a jumps.

```python
def bipower_variation(
    data: pd.DataFrame,
    *,
    price_col: str = "close",
) -> pd.Series:
    # BV = (π/2) * sum(|r_t| * |r_{t-1}|)
```

**Por que incluir:** Separa variação contínua de jumps; fundamental para modelos de volatilidade modernos; base do modelo HAR-RV-J.

---

#### 3.3.2 `jump_variation`
**Referência:** Andersen, Bollerslev & Diebold (2007)

Componente de jump: max(RV - BV, 0).

```python
def jump_variation(
    data: pd.DataFrame,
    *,
    price_col: str = "close",
) -> pd.Series:
    # JV = max(RV - BV, 0)
```

**Por que incluir:** Jumps têm poder preditivo diferente da volatilidade contínua; usado em HAR-RV-J.

---

#### 3.3.3 `realized_semivariance`
**Referência:** Barndorff-Nielsen, Kinnebrock & Shephard (2010) - "Measuring Downside Risk"

Decompõe variância em componentes positivos e negativos.

```python
def realized_semivariance(
    data: pd.DataFrame,
    *,
    price_col: str = "close",
) -> pd.DataFrame:
    # RSV+ = sum(r_t² * I(r_t > 0))
    # RSV- = sum(r_t² * I(r_t < 0))
```

**Por que incluir:** RSV- prediz volatilidade futura melhor que RV total (assimetria de Patton-Sheppard); captura "bad volatility".

---

#### 3.3.4 `signed_jump_variation`
**Referência:** Patton & Sheppard (2015) - "Good Volatility, Bad Volatility"

Decompõe jumps em positivos e negativos para capturar assimetria.

```python
def signed_jump_variation(
    data: pd.DataFrame,
    *,
    price_col: str = "close",
) -> pd.DataFrame:
    # ΔJ+ = (RSV+ - BV/2)+
    # ΔJ- = (RSV- - BV/2)+
```

**Por que incluir:** Jumps negativos têm maior persistência que positivos; melhora previsão de volatilidade.

---

### 3.4 Entropy & Complexity

#### 3.4.1 `sample_entropy`
**Referência:** Richman & Moorman (2000)

Mede regularidade/previsibilidade de uma série temporal.

```python
def sample_entropy(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 100,
    m: int = 2,
    r: float = 0.2,
    price_col: str = "close",
) -> pd.Series:
    # SampEn = -ln(A/B) onde A e B são contagens de padrões similares
```

**Por que incluir:** Detecta mudanças de regime; baixa entropia indica mercado previsível; usado em análise de séries financeiras.

---

#### 3.4.2 `permutation_entropy`
**Referência:** Bandt & Pompe (2002)

Entropia baseada em padrões ordinais (ranking de valores consecutivos).

```python
def permutation_entropy(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 100,
    order: int = 3,
    delay: int = 1,
    price_col: str = "close",
) -> pd.Series:
    # PE = -sum(p_i * ln(p_i)) para distribuição de padrões ordinais
```

**Por que incluir:** Robusto a ruído; computacionalmente eficiente; detecta transições de regime em mercados.

---

### 3.5 Regime & Structural Breaks

#### 3.5.1 `cusum_statistic`
**Referência:** Brown, Durbin & Evans (1975) - CUSUM test

Detecta mudanças estruturais acumulando desvios da média.

```python
def cusum_statistic(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # CUSUM_t = sum_{i=1}^{t}(r_i - μ) / σ
```

**Por que incluir:** Detecta breaks em tempo real; usado em controle de qualidade e finanças; base para estratégias de regime.

---

#### 3.5.2 `variance_ratio`
**Referência:** Lo & MacKinlay (1988) - "Stock Market Prices Do Not Follow Random Walks"

Testa eficiência de mercado comparando variâncias de diferentes horizontes.

```python
def variance_ratio(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 120,
    holding_period: int = 5,
    price_col: str = "close",
) -> pd.Series:
    # VR(q) = Var(r_t(q)) / (q * Var(r_t))
```

**Por que incluir:** VR > 1 indica momentum, VR < 1 indica mean-reversion; paper seminal em market microstructure.

---

#### 3.5.3 `trend_intensity`
**Referência:** Inspired by ADX but different calculation

Mede força de tendência via ratio de retornos direcionais.

```python
def trend_intensity(
    data: pd.DataFrame,
    *,
    window: int = 20,
    price_col: str = "close",
) -> pd.Series:
    # TI = |sum(r_i)| / sum(|r_i|)
```

**Por que incluir:** Normalizado [0,1]; 1 = tendência perfeita, 0 = sem tendência; útil para regime detection.

---

### 3.6 Higher Moments & Risk

#### 3.6.1 `realized_skewness`
**Referência:** Amaya et al. (2015) - "Does Realized Skewness Predict the Cross-Section of Equity Returns?"

Terceiro momento realizado para capturar assimetria.

```python
def realized_skewness(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    price_col: str = "close",
) -> pd.Series:
    # RSkew = (√n * sum(r³)) / RV^(3/2)
```

**Por que incluir:** Prediz retornos cross-section (paper em JFE); ações com skew positivo têm retornos menores.

---

#### 3.6.2 `realized_kurtosis`
**Referência:** Amaya et al. (2015)

Quarto momento realizado para capturar fat tails.

```python
def realized_kurtosis(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    price_col: str = "close",
) -> pd.Series:
    # RKurt = (n * sum(r⁴)) / RV²
```

**Por que incluir:** Complementa skewness; kurtosis alta indica risco de tail events.

---

### 3.7 Volume & Price Interaction

#### 3.7.1 `price_volume_correlation`
**Referência:** Karpoff (1987) - "The Relation Between Price Changes and Trading Volume"

Correlação rolling entre variação de preço e volume.

```python
def price_volume_correlation(
    data: pd.DataFrame,
    *,
    window: int = 20,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    # corr(|r_t|, V_t) over window
```

**Por que incluir:** Proxy de information flow; alta correlação indica trades informativos.

---

#### 3.7.2 `volume_volatility_ratio`
**Referência:** Llorente et al. (2002) - "Dynamic Volume-Return Relation"

Ratio de volume normalizado por volatilidade.

```python
def volume_volatility_ratio(
    data: pd.DataFrame,
    *,
    window: int = 20,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    # VVR = (V / V_mean) / (σ / σ_mean)
```

**Por que incluir:** Normaliza volume pela volatilidade; picos indicam eventos de liquidez.

---

#### 3.7.3 `close_location_value`
**Referência:** Arms (1989) - Equivolume charting

Onde o close está dentro do range high-low.

```python
def close_location_value(
    data: pd.DataFrame,
    *,
    window: int = 1,
) -> pd.Series:
    # CLV = (2*C - H - L) / (H - L)
```

**Por que incluir:** Proxy de buy/sell pressure; base para Chaikin Money Flow; range [-1, 1].

---

### 3.8 Autocorrelation & Memory

#### 3.8.1 `return_autocorrelation`
**Referência:** Campbell, Lo & MacKinlay (1997)

Autocorrelação de retornos em diferentes lags.

```python
def return_autocorrelation(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    lag: int = 1,
    price_col: str = "close",
) -> pd.Series:
    # AC(lag) = corr(r_t, r_{t-lag})
```

**Por que incluir:** Detecta momentum/reversal; base para variance ratio; usado em market efficiency tests.

---

#### 3.8.2 `absolute_return_autocorrelation`
**Referência:** Ding, Granger & Engle (1993) - "Long Memory Property of Stock Returns"

Autocorrelação de retornos absolutos (detecta clustering de volatilidade).

```python
def absolute_return_autocorrelation(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    lag: int = 1,
    price_col: str = "close",
) -> pd.Series:
    # AC_abs(lag) = corr(|r_t|, |r_{t-lag}|)
```

**Por que incluir:** Retornos absolutos têm memória longa (decay lento); proxy de volatility clustering.

---

### 3.9 Range-Based Features

#### 3.9.1 `intraday_range`
**Referência:** Alizadeh, Brandt & Diebold (2002)

Log range como proxy de volatilidade.

```python
def intraday_range(
    data: pd.DataFrame,
    *,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    # IR = rolling_mean(ln(H) - ln(L))
```

**Por que incluir:** Mais robusto a outliers que volatilidade; base para estimadores range-based.

---

#### 3.9.2 `overnight_gap`
**Referência:** Análise de gaps overnight

Diferença entre open atual e close anterior.

```python
def overnight_gap(
    data: pd.DataFrame,
    *,
    open_col: str = "open",
    close_col: str = "close",
) -> pd.Series:
    # gap = ln(O_t / C_{t-1})
```

**Por que incluir:** Captura news overnight; componente importante de volatilidade total; usado em estratégias de gap.

---

#### 3.9.3 `intraday_return`
**Referência:** Decomposição de retornos

Retorno intraday (open-to-close).

```python
def intraday_return(
    data: pd.DataFrame,
    *,
    open_col: str = "open",
    close_col: str = "close",
) -> pd.Series:
    # r_intra = ln(C_t / O_t)
```

**Por que incluir:** Complementa overnight gap; diferentes dinâmicas intraday vs overnight.

---

## 4. Priorização Sugerida

### Tier 1 - Alta Prioridade (Fundamento Acadêmico Sólido)
1. `parkinson_volatility` - Complementa Yang-Zhang
2. `garman_klass_volatility` - Muito citado
3. `bipower_variation` - Base para decomposição de jumps
4. `realized_semivariance` - Paper de Patton-Sheppard
5. `amihud_illiquidity` - Proxy de liquidez mais usado

### Tier 2 - Média Prioridade (Útil para ML)
6. `variance_ratio` - Test de eficiência
7. `realized_skewness` - Prediz retornos
8. `realized_kurtosis` - Risco de tail
9. `corwin_schultz_spread` - Spread sem bid-ask
10. `permutation_entropy` - Detecta regimes

### Tier 3 - Menor Prioridade (Nice to Have)
11. `roll_spread` - Clássico mas menos preciso
12. `rogers_satchell_volatility` - Com drift
13. `jump_variation` - Requer bipower
14. `sample_entropy` - Computacionalmente intensivo
15. Outras features de range e autocorrelação

---

## 5. Padrão de Implementação

Cada feature deve seguir o template em `new_feature.md`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from quantmaster.features.utils import get_price_series, validate_positive_int, validate_columns


def feature_name(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    # outros parâmetros keyword-only
) -> pd.Series:
    """Docstring com descrição, referência acadêmica, e exemplo."""
    window = validate_positive_int(window, name="window")
    
    # Cálculo
    
    out.name = f"feature_name_{window}"
    return out
```

---

## 6. Conclusão

O **Quantmaster** tem uma base sólida de features estatísticas avançadas (HAR-RV, Hurst, OU, fracdiff). As hipóteses apresentadas expandem a biblioteca em direções complementares:

1. **Estimadores de volatilidade range-based** - Mais eficientes que close-to-close
2. **Microestrutura/liquidez** - Amihud, Roll, Corwin-Schultz
3. **Decomposição de variância** - Semivariance, jumps, bipower
4. **Entropia e complexidade** - Sample/Permutation entropy
5. **Higher moments** - Skewness, kurtosis realizados

Todas as features propostas:
- Têm base acadêmica publicada em journals de primeira linha
- Podem ser calculadas a partir de OHLCV
- Seguem o estilo funcional da biblioteca
- Não duplicam TA-Lib/ta/pandas-ta
- São usadas em produção por quants e hedge funds

---

# PARTE II: Features Avançadas 2015-2025

## 7. Pesquisa Recente: Avanços em Features Quantitativas (2015-2025)

A última década trouxe avanços significativos em três frentes principais:
1. **Rough Volatility & Path Signatures** - Nova teoria matemática para modelar volatilidade
2. **Machine Learning Features** - Features engineered para modelos de ML
3. **Estimadores Jump-Robust Avançados** - MedRV, MinRV, HARQ
4. **Microstructure Moderna** - VPIN, Order Flow Imbalance
5. **Decomposições Avançadas** - Semivariance HAR, Good/Bad Volatility

---

## 8. Novas Hipóteses de Features (2015-2025)

### 8.1 Rough Volatility & Realized Roughness

#### 8.1.1 `realized_roughness`
**Referência:** Gatheral, Jaisson & Rosenbaum (2018) - "Volatility is Rough"

Estima o expoente de Hurst da volatilidade realizada, detectando "roughness" (H < 0.5).

```python
def realized_roughness(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    lags: list[int] = [1, 2, 5, 10],
    price_col: str = "close",
) -> pd.Series:
    # H estimado via regressão log-log de E[|log(RV_t) - log(RV_{t-lag})|^2] vs lag
    # H < 0.5 indica rough volatility (anti-persistência)
```

**Por que incluir:** Paper seminal de 2018 com >1000 citações; revolucionou modelagem de volatilidade; H tipicamente ~0.1 para índices; detecta regime de volatilidade.

**Evidência:** Gatheral et al. (2018) demonstram que a volatilidade de índices de ações exibe roughness com H ≈ 0.1, muito menor que 0.5 esperado para Brownian motion.

---

#### 8.1.2 `log_volatility_increment`
**Referência:** Gatheral et al. (2018), Bennedsen et al. (2022)

Incrementos de log-volatilidade para análise de roughness.

```python
def log_volatility_increment(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    lag: int = 1,
    price_col: str = "close",
) -> pd.Series:
    # Δlog(RV) = log(RV_t) - log(RV_{t-lag})
```

**Por que incluir:** Base para estimação de roughness; distribuição tem propriedades específicas para rough vol.

---

### 8.2 Jump-Robust Estimators Modernos

#### 8.2.1 `medrv` (Median Realized Variance)
**Referência:** Andersen, Dobrev & Schaumburg (2012) - "Jump-Robust Volatility Estimation using Nearest Neighbor Truncation"

Estimador de variância usando mediana de retornos adjacentes.

```python
def medrv(
    data: pd.DataFrame | pd.Series,
    *,
    price_col: str = "close",
) -> pd.Series:
    # MedRV = (π / (6 - 4√3 + π)) * sum(med(|r_{t-1}|, |r_t|, |r_{t+1}|)²)
```

**Por que incluir:** Mais robusto a jumps que bipower variation; melhor eficiência em amostras finitas; publicado em Journal of Econometrics.

---

#### 8.2.2 `minrv` (Minimum Realized Variance)
**Referência:** Andersen, Dobrev & Schaumburg (2012)

Estimador usando mínimo de retornos adjacentes.

```python
def minrv(
    data: pd.DataFrame | pd.Series,
    *,
    price_col: str = "close",
) -> pd.Series:
    # MinRV = (π / (π - 2)) * sum(min(|r_{t-1}|, |r_t|)²)
```

**Por que incluir:** Complementa MedRV; mais conservador; útil para detecção de jumps extremos.

---

#### 8.2.3 `realized_quarticity`
**Referência:** Barndorff-Nielsen & Shephard (2002), Bollerslev, Patton & Quaedvlieg (2016)

Quarto momento realizado para estimar variância da variância.

```python
def realized_quarticity(
    data: pd.DataFrame | pd.Series,
    *,
    price_col: str = "close",
) -> pd.Series:
    # RQ = (n/3) * sum(r_t^4)
```

**Por que incluir:** Base do modelo HARQ (Bollerslev 2016); captura measurement error em RV; melhora forecasts de volatilidade.

---

#### 8.2.4 `harq_adjustment`
**Referência:** Bollerslev, Patton & Quaedvlieg (2016) - "Exploiting the Errors"

Correção de atenuação usando quarticity para melhorar HAR-RV.

```python
def harq_adjustment(
    data: pd.DataFrame | pd.Series,
    *,
    price_col: str = "close",
    window: int = 22,
) -> pd.Series:
    # HARQ usa RQ para ajustar coeficientes do HAR dinamicamente
    # Retorna fator de ajuste baseado em RQ/RV²
```

**Por que incluir:** Paper em JoE 2016; melhora 5-15% em forecasts de RV; lida com heteroscedasticidade do erro de medição.

---

### 8.3 Semivariance & Asymmetric Volatility Avançada

#### 8.3.1 `shar_components`
**Referência:** Patton & Sheppard (2015) - "Good Volatility, Bad Volatility"

HAR com semivariances separadas (extensão do HAR-RV).

```python
def shar_components(
    data: pd.DataFrame | pd.Series,
    *,
    price_col: str = "close",
    weekly_window: int = 5,
    monthly_window: int = 22,
) -> pd.DataFrame:
    # Retorna: RSV+_d, RSV-_d, RSV+_w, RSV-_w, RSV+_m, RSV-_m
```

**Por que incluir:** RSV- tem 2x mais poder preditivo que RSV+; assimetria é chave para risk management; paper altamente citado.

---

#### 8.3.2 `leverage_effect_measure`
**Referência:** Bollerslev, Litvinova & Tauchen (2006), Carr & Wu (2017)

Mede correlação negativa entre retornos e volatilidade futura.

```python
def leverage_effect_measure(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    horizon: int = 5,
    price_col: str = "close",
) -> pd.Series:
    # LE = corr(r_t, RV_{t+horizon} - RV_t)
```

**Por que incluir:** Quantifica leverage effect; varia ao longo do tempo; útil para timing de estratégias de volatilidade.

---

### 8.4 Microstructure Moderna (2015-2024)

#### 8.4.1 `vpin` (Volume-Synchronized Probability of Informed Trading)
**Referência:** Easley, López de Prado & O'Hara (2012) - "The Volume Clock"

Probabilidade de trading informado sincronizada por volume.

```python
def vpin(
    data: pd.DataFrame,
    *,
    volume_bucket_size: int = 50,
    n_buckets: int = 50,
    volume_col: str = "volume",
    price_col: str = "close",
) -> pd.Series:
    # VPIN = |V_buy - V_sell| / (V_buy + V_sell)
    # Usa bulk volume classification para estimar buy/sell
```

**Por que incluir:** Co-autor Marcos Lopez de Prado; detecta informed trading sem dados de ordem; usado para flash crash prediction.

**Nota:** Versão simplificada usando bulk volume classification (sem tick data).

---

#### 8.4.2 `order_flow_imbalance`
**Referência:** Chordia, Roll & Subrahmanyam (2002), Cont, Kukanov & Stoikov (2014)

Proxy de imbalance buy/sell usando OHLCV.

```python
def order_flow_imbalance(
    data: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.Series:
    # OFI = (V * sign(C - O)) ou usando close location value
    # Versão OHLCV: OFI = V * CLV
```

**Por que incluir:** Prediz retornos de curto prazo; base para market making; calculável com OHLCV.

---

#### 8.4.3 `relative_spread_proxy`
**Referência:** Abdi & Ranaldo (2017) - "A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices"

Proxy de spread usando close, high, low.

```python
def relative_spread_proxy(
    data: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.Series:
    # Spread ≈ 2 * sqrt(max(0, E[(H-C)(H-O) + (L-C)(L-O)]))
```

**Por que incluir:** Publicado em RFS 2017; mais preciso que Corwin-Schultz para ações líquidas; usa apenas OHLC.

---

### 8.5 Time-Series Momentum & Trend Features

#### 8.5.1 `time_series_momentum`
**Referência:** Moskowitz, Ooi & Pedersen (2012) - "Time Series Momentum"

Momentum baseado em retorno próprio do ativo (não cross-sectional).

```python
def time_series_momentum(
    data: pd.DataFrame | pd.Series,
    *,
    lookback: int = 252,
    volatility_window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # TSMOM = r_{t-lookback:t} / σ_{t-vol_window:t}
```

**Por que incluir:** Paper seminal em JFE; base para CTAs e managed futures; volatility-scaled.

---

#### 8.5.2 `trend_strength_indicator`
**Referência:** Baltas & Kosowski (2020) - "Demystifying Time-Series Momentum Strategies"

Força de tendência normalizada.

```python
def trend_strength_indicator(
    data: pd.DataFrame | pd.Series,
    *,
    windows: list[int] = [21, 63, 126, 252],
    price_col: str = "close",
) -> pd.Series:
    # TSI = média dos sinais de momentum em múltiplos horizontes
    # Cada sinal = sign(r) * min(|r|/σ, 2)
```

**Por que incluir:** Combina múltiplos horizontes; clipping reduz outliers; usado por CTAs.

---

#### 8.5.3 `price_acceleration`
**Referência:** Inspired by physics analogy in quant finance

Segunda derivada do preço (momentum do momentum).

```python
def price_acceleration(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    price_col: str = "close",
) -> pd.Series:
    # Accel = d²P/dt² ≈ (r_t - r_{t-window}) / window
```

**Por que incluir:** Detecta mudanças de regime antes do momentum; leading indicator para reversões.

---

### 8.6 Machine Learning-Ready Features

#### 8.6.1 `rolling_beta`
**Referência:** Fama & French (1992), Frazzini & Pedersen (2014) - "Betting Against Beta"

Beta CAPM rolling em relação a benchmark.

```python
def rolling_beta(
    data: pd.DataFrame,
    benchmark: pd.Series,
    *,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # β = Cov(r_asset, r_benchmark) / Var(r_benchmark)
```

**Por que incluir:** Feature fundamental para modelos de risco; varia significativamente ao longo do tempo.

---

#### 8.6.2 `downside_beta`
**Referência:** Ang, Chen & Xing (2006) - "Downside Risk"

Beta calculado apenas em dias de queda do mercado.

```python
def downside_beta(
    data: pd.DataFrame,
    benchmark: pd.Series,
    *,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # β- = Cov(r_asset, r_benchmark | r_benchmark < 0) / Var(r_benchmark | r_benchmark < 0)
```

**Por que incluir:** Captura risco assimétrico; premium por downside beta é significativo.

---

#### 8.6.3 `information_discreteness`
**Referência:** Da, Gurun & Warachka (2014) - "Frog in the Pan"

Mede se informação chega de forma discreta ou contínua.

```python
def information_discreteness(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    price_col: str = "close",
) -> pd.Series:
    # ID = sign(r_total) * (% dias com mesmo sinal - % dias com sinal oposto)
```

**Por que incluir:** Paper em RFS; retornos "discretos" (poucos dias grandes) revertem mais.

---

#### 8.6.4 `max_drawdown_duration`
**Referência:** Magdon-Ismail & Atiya (2004)

Duração do maior drawdown em janela rolling.

```python
def max_drawdown_duration(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 252,
    price_col: str = "close",
) -> pd.Series:
    # MDD_duration = número de períodos do maior drawdown na janela
```

**Por que incluir:** Complementa max drawdown; duração é tão importante quanto magnitude.

---

### 8.7 Tail Risk & Extreme Events

#### 8.7.1 `tail_risk_measure`
**Referência:** Kelly & Jiang (2014) - "Tail Risk and Asset Prices"

Medida de risco de cauda baseada em cross-section.

```python
def tail_risk_measure(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    quantile: float = 0.05,
    price_col: str = "close",
) -> pd.Series:
    # λ = média dos retornos abaixo do quantile / quantile esperado
```

**Por que incluir:** Tail risk tem premium; paper em JFE; calculável para ativo individual.

---

#### 8.7.2 `value_at_risk_historical`
**Referência:** Jorion (2006) - "Value at Risk"

VaR histórico rolling.

```python
def value_at_risk_historical(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 252,
    confidence: float = 0.95,
    price_col: str = "close",
) -> pd.Series:
    # VaR = quantile(r, 1-confidence) over window
```

**Por que incluir:** Métrica de risco padrão; base para Expected Shortfall; exigida por reguladores.

---

#### 8.7.3 `expected_shortfall`
**Referência:** Acerbi & Tasche (2002) - "On the Coherence of Expected Shortfall"

CVaR/ES - média das perdas além do VaR.

```python
def expected_shortfall(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 252,
    confidence: float = 0.95,
    price_col: str = "close",
) -> pd.Series:
    # ES = E[r | r < VaR]
```

**Por que incluir:** Substitui VaR em Basel III; coerente; captura magnitude de perdas extremas.

---

### 8.8 Efficiency & Predictability

#### 8.8.1 `market_efficiency_index`
**Referência:** Inspired by Lo (2004) - "Adaptive Markets Hypothesis"

Índice composto de eficiência de mercado.

```python
def market_efficiency_index(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # MEI = 1 - |AC(1)| - 0.5*|VR - 1|
    # Combina autocorrelação e variance ratio
```

**Por que incluir:** Detecta períodos de ineficiência; base para timing de estratégias.

---

#### 8.8.2 `runs_test_statistic`
**Referência:** Wald & Wolfowitz (1940), modernizado para finanças

Testa aleatoriedade via contagem de runs.

```python
def runs_test_statistic(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # Z = (R - μ_R) / σ_R onde R = número de runs
```

**Por que incluir:** Detecta clustering (poucos runs) ou reversão (muitos runs); complementa autocorrelação.

---

### 8.9 Cointegration & Relative Value

#### 8.9.1 `spread_zscore`
**Referência:** Inspired by pairs trading literature

Z-score de spread entre ativo e benchmark.

```python
def spread_zscore(
    data: pd.DataFrame,
    benchmark: pd.Series,
    *,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # z = (spread - mean(spread)) / std(spread)
    # spread = log(P_asset) - β*log(P_benchmark)
```

**Por que incluir:** Base para pairs trading; usado em arbitragem estatística.

---

### 8.10 Path Signatures (Cutting-Edge 2020+)

#### 8.10.1 `path_signature_features`
**Referência:** Lyons (1998), Kidger & Lyons (2020) - "Signatory: Differentiable Computations of the Signature"

Features de signature para time series.

```python
def path_signature_features(
    data: pd.DataFrame,
    *,
    depth: int = 2,
    window: int = 20,
) -> pd.DataFrame:
    # Signature = (1, ∫dX, ∫∫dX⊗dX, ...) truncada em depth
    # Para OHLCV: path 5-dimensional
```

**Por que incluir:** State-of-the-art em ML para time series; captura dependências temporais de forma única; usado em quant funds modernos.

**Nota:** Requer implementação cuidadosa ou dependência de biblioteca (signatory, esig).

---

## 9. Priorização Atualizada (2015-2025)

### Tier 1+ (Cutting-Edge, Alto Impacto)
1. `realized_roughness` - Rough volatility (Gatheral 2018)
2. `medrv` / `minrv` - Jump-robust (Andersen 2012)
3. `realized_quarticity` + `harq_adjustment` - HARQ (Bollerslev 2016)
4. `shar_components` - Semivariance HAR (Patton 2015)

### Tier 1 (Sólido, Bem Documentado)
5. `vpin` - Microstructure (Lopez de Prado 2012)
6. `time_series_momentum` - TSMOM (Moskowitz 2012)
7. `leverage_effect_measure` - Assimetria
8. `expected_shortfall` - Tail risk (Basel III)

### Tier 2 (Útil para ML/Backtesting)
9. `rolling_beta` / `downside_beta`
10. `information_discreteness`
11. `order_flow_imbalance`
12. `relative_spread_proxy`
13. `trend_strength_indicator`

### Tier 3 (Especializado/Avançado)
14. `path_signature_features` - Requer dependência externa
15. `spread_zscore` - Pairs trading
16. `market_efficiency_index`
17. `runs_test_statistic`

---

## 10. Comparativo: Features Clássicas vs. Modernas

| Aspecto | Clássicas (pré-2015) | Modernas (2015-2025) |
|---------|---------------------|----------------------|
| **Volatilidade** | RV, HAR-RV | Rough vol, HARQ, MedRV |
| **Jumps** | Bipower Variation | Signed jumps, Good/Bad vol |
| **Liquidez** | Amihud, Roll | VPIN, Order flow imbalance |
| **Momentum** | Simple returns | TSMOM, Multi-horizon |
| **Risco** | VaR | Expected Shortfall, Tail risk |
| **Complexidade** | Hurst, Entropy | Path signatures |
| **ML-ready** | Features isoladas | Decomposições, multi-output |

---

## 11. Considerações de Implementação (2015-2025)

### 11.1 Dependências Opcionais
Algumas features modernas podem se beneficiar de:
- `signatory` / `esig` para path signatures
- `numba` para aceleração de cálculos rolling

### 11.2 Multi-Output Features
Muitas features modernas retornam múltiplas colunas:
- `shar_components` → 6 colunas (RSV+/- em 3 horizontes)
- `path_signature_features` → múltiplas features de signature

### 11.3 Parâmetros Sensíveis
- Roughness estimation requer cuidado com número de lags
- VPIN depende de escolha de bucket size
- HARQ requer janela suficiente para RQ estável

---

## 12. Conclusão Ampliada

A pesquisa de 2015-2025 trouxe avanços significativos:

1. **Rough Volatility** - Novo paradigma: volatilidade é "rough" (H ≈ 0.1)
2. **HARQ e Quarticity** - Correção de atenuação melhora forecasts
3. **Semivariance Assimétrica** - "Bad volatility" prediz melhor
4. **Microstructure VPIN** - Detecta informed trading sem tick data
5. **Path Signatures** - ML state-of-the-art para time series

O **Quantmaster** pode se diferenciar ao implementar estas features modernas que **não existem em TA-Lib, ta, ou pandas-ta**, posicionando a biblioteca como referência para quants que precisam de features academicamente sólidas e cutting-edge.

---

## Referências Principais

1. Amihud, Y. (2002). "Illiquidity and Stock Returns"
2. Barndorff-Nielsen & Shephard (2004). "Power and Bipower Variation"
3. Corwin & Schultz (2012). "A Simple Way to Estimate Bid-Ask Spreads"
4. Garman & Klass (1980). "On the Estimation of Security Price Volatilities"
5. Kakushadze, Z. (2016). "101 Formulaic Alphas" - WorldQuant
6. Lo & MacKinlay (1988). "Stock Market Prices Do Not Follow Random Walks"
7. Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
8. Parkinson, M. (1980). "The Extreme Value Method"
9. Patton & Sheppard (2015). "Good Volatility, Bad Volatility"
10. Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread"

### Referências Adicionais (2015-2025)

11. Gatheral, Jaisson & Rosenbaum (2018). "Volatility is Rough" - Quantitative Finance
12. Bollerslev, Patton & Quaedvlieg (2016). "Exploiting the Errors" - Journal of Econometrics
13. Andersen, Dobrev & Schaumburg (2012). "Jump-Robust Volatility Estimation" - Journal of Econometrics
14. Easley, López de Prado & O'Hara (2012). "The Volume Clock" - Review of Financial Studies
15. Moskowitz, Ooi & Pedersen (2012). "Time Series Momentum" - Journal of Financial Economics
16. Ang, Chen & Xing (2006). "Downside Risk" - Review of Financial Studies
17. Kelly & Jiang (2014). "Tail Risk and Asset Prices" - Journal of Financial Economics
18. Kidger & Lyons (2020). "Signatory: Differentiable Computations of the Signature" - arXiv
19. Bennedsen, Lunde & Pakkanen (2022). "Decoupling the Short and Long-term Behavior of Stochastic Volatility"
20. Abdi & Ranaldo (2017). "A Simple Estimation of Bid-Ask Spreads" - Review of Financial Studies
21. Da, Gurun & Warachka (2014). "Frog in the Pan" - Review of Financial Studies
22. Baltas & Kosowski (2020). "Demystifying Time-Series Momentum Strategies"
23. Frazzini & Pedersen (2014). "Betting Against Beta" - Journal of Financial Economics
24. Acerbi & Tasche (2002). "On the Coherence of Expected Shortfall"

---

# PARTE III: Contribuições da Comunidade MQL5

## 13. Features da Comunidade MQL5 (mql5.com/en/articles)

A comunidade MQL5 é uma das maiores comunidades de trading algorítmico, com milhares de artigos sobre indicadores e estratégias. Abaixo estão features identificadas que:
- **Não existem em TA-Lib/ta/pandas-ta**
- **São calculáveis a partir de OHLCV**
- **Se enquadram no estilo funcional do Quantmaster**

---

### 13.1 Regime Detection & Market State

#### 13.1.1 `generalized_hurst_exponent`
**Referência:** MQL5 - "Implementing the Generalized Hurst Exponent and the Variance Ratio test in MQL5"

Extensão do Hurst exponent usando diferentes ordens de momentos (q).

```python
def generalized_hurst_exponent(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 100,
    q: float = 2.0,
    max_lag: int = 20,
    price_col: str = "close",
) -> pd.Series:
    # H(q) estimado via scaling de momentos de ordem q
    # q=1: absolute deviation scaling
    # q=2: variance scaling (long-range dependence)
```

**Por que incluir:** Generaliza o Hurst DFA existente; permite análise multi-escala; q=2 é mais robusto para detectar memória longa.

---

#### 13.1.2 `mean_reversion_half_life`
**Referência:** Ernest Chan - "Algorithmic Trading", MQL5 implementation

Tempo para desvio da média reduzir pela metade (derivado do processo OU).

```python
def mean_reversion_half_life(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # HL = -ln(2) / λ
    # λ obtido via regressão: Δp_t = λ*p_{t-1} + ε
    # λ < 0 indica mean-reversion; |λ| grande = reversão rápida
```

**Por que incluir:** Complementa `ornstein_uhlenbeck` existente; métrica prática para pairs trading; λ positivo descarta mean-reversion.

---

#### 13.1.3 `fractal_dimension_mincover`
**Referência:** MQL5 - "Evaluating the ability of Fractal index and Hurst exponent to predict financial time series"

Dimensão fractal via método de cobertura mínima (mais robusto que R/S).

```python
def fractal_dimension_mincover(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 100,
    price_col: str = "close",
) -> pd.Series:
    # D via método de área mínima coberta
    # D ≈ 2: random walk (sem tendência)
    # D < 2: série com tendência/memória
```

**Por que incluir:** Funciona com janelas menores que R/S tradicional; detecta tendências locais; base para trading systems adaptativos.

---

#### 13.1.4 `trend_strength_autocorr`
**Referência:** MQL5 - "Building a Custom Market Regime Detection System"

Força de tendência baseada em autocorrelação de retornos.

```python
def trend_strength_autocorr(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    lag: int = 1,
    price_col: str = "close",
) -> pd.Series:
    # TS = autocorr(r, lag) * sign(mean(r))
    # Positivo = trending up, Negativo = trending down
    # |TS| alto = tendência forte
```

**Por que incluir:** Combina direção e força; diferente do `trend_intensity` proposto; base para regime switching.

---

#### 13.1.5 `mean_reversion_strength`
**Referência:** MQL5 - "Building a Custom Market Regime Detection System"

Força de mean-reversion baseada em desvios do Z-score.

```python
def mean_reversion_strength(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 20,
    price_col: str = "close",
) -> pd.Series:
    # MRS = -corr(z_t, Δz_{t+1}) onde z = (P - μ) / σ
    # MRS > 0 indica mean-reversion
```

**Por que incluir:** Complementa half-life; útil para calibrar estratégias de reversão.

---

### 13.2 Entropy & Complexity (MQL5 Implementations)

#### 13.2.1 `cross_sample_entropy`
**Referência:** MQL5 - "Grokking market memory through differentiation and entropy analysis"

Entropia cruzada entre duas séries temporais.

```python
def cross_sample_entropy(
    data1: pd.Series,
    data2: pd.Series,
    *,
    m: int = 2,
    r: float = 0.2,
) -> pd.Series:
    # CrossSampEn mede similaridade de padrões entre séries
    # Útil para detectar lead-lag relationships
```

**Por que incluir:** Detecta relações não-lineares entre ativos; base para pairs trading; não existe em libs tradicionais.

---

#### 13.2.2 `approximate_entropy`
**Referência:** Pincus & Kalman (1991), MQL5 implementation

Predecessor do Sample Entropy, mede irregularidade.

```python
def approximate_entropy(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 100,
    m: int = 2,
    r: float = 0.2,
    price_col: str = "close",
) -> pd.Series:
    # ApEn = ln(C_m(r)) - ln(C_{m+1}(r))
    # Baixo ApEn = série regular/previsível
```

**Por que incluir:** Mais sensível que SampEn para séries curtas; detecta mudanças de regime; histórico de uso em finanças.

---

### 13.3 Adaptive Indicators

#### 13.3.1 `kaufman_efficiency_ratio`
**Referência:** Perry Kaufman - "Trading Systems and Methods", MQL5 implementations

Ratio de eficiência: movimento direcional vs. volatilidade total.

```python
def kaufman_efficiency_ratio(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 10,
    price_col: str = "close",
) -> pd.Series:
    # ER = |P_t - P_{t-n}| / sum(|P_i - P_{i-1}|)
    # ER = 1: tendência perfeita
    # ER ≈ 0: mercado sem direção
```

**Por que incluir:** Base para Adaptive Moving Average (AMA); métrica limpa de "trendiness"; range [0,1].

---

#### 13.3.2 `fractal_adaptive_factor`
**Referência:** MQL5 - "Fractal Adaptive Moving Average", John Ehlers

Fator de suavização baseado em dimensão fractal.

```python
def fractal_adaptive_factor(
    data: pd.DataFrame,
    *,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    # α = exp(-4.6 * (D - 1))
    # D = dimensão fractal local
    # α alto = mercado trending, α baixo = ranging
```

**Por que incluir:** Adapta automaticamente a velocidade de resposta; base para filtros adaptativos; usa OHLC.

---

#### 13.3.3 `chande_momentum_oscillator`
**Referência:** Tushar Chande, MQL5 - "Variable Index Dynamic Average"

Oscilador de momentum normalizado [-100, 100].

```python
def chande_momentum_oscillator(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 14,
    price_col: str = "close",
) -> pd.Series:
    # CMO = 100 * (sum(up) - sum(down)) / (sum(up) + sum(down))
    # up = ganhos, down = perdas (valores absolutos)
```

**Por que incluir:** Usado no VIDYA; normalizado diferente do RSI; detecta overbought/oversold.

---

### 13.4 Dynamic Channel & Volatility

#### 13.4.1 `nrtr_channel`
**Referência:** Konstantin Kopyrkin (Nick Rypock), MQL5 - "The NRTR indicator"

Canal dinâmico que se ajusta a cada nova tendência.

```python
def nrtr_channel(
    data: pd.DataFrame,
    *,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    # Canal que reseta em cada reversão de tendência
    # Retorna: support, resistance, trend_direction
```

**Por que incluir:** Canal adaptativo diferente de Keltner/Bollinger; reseta período a cada trend change; usado em breakout systems.

---

#### 13.4.2 `volatility_ratio`
**Referência:** Jack Schwager, MQL5 implementations

Ratio entre range atual e range histórico.

```python
def volatility_ratio(
    data: pd.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    # VR = True_Range_hoje / ATR_n
    # VR > 1.5: expansão de volatilidade (breakout potencial)
    # VR < 0.5: contração (squeeze)
```

**Por que incluir:** Detecta expansão/contração de volatilidade; complementa ATR; timing para breakouts.

---

### 13.5 Order Flow Proxies (OHLCV-based)

#### 13.5.1 `tick_imbalance_proxy`
**Referência:** MQL5 - "Tick Buffer VWAP and Short-Window Imbalance Engine"

Proxy de imbalance buy/sell usando OHLCV.

```python
def tick_imbalance_proxy(
    data: pd.DataFrame,
    *,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    # Imbalance = V * (2*C - H - L) / (H - L)
    # Positivo = buying pressure, Negativo = selling pressure
```

**Por que incluir:** Proxy de order flow sem tick data; usa CLV ponderado por volume; detecta pressão direcional.

---

#### 13.5.2 `volume_weighted_close_location`
**Referência:** MQL5 - Order flow analysis adaptations

Close location value ponderado por volume relativo.

```python
def volume_weighted_close_location(
    data: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.Series:
    # VWCLV = sum(CLV * V) / sum(V) over window
    # CLV = (2*C - H - L) / (H - L)
```

**Por que incluir:** Mais informativo que CLV simples; acumula pressão ao longo do tempo; base para money flow.

---

### 13.6 Price Pattern Features

#### 13.6.1 `bar_range_position`
**Referência:** MQL5 - Price action analysis patterns

Posição normalizada do close dentro da barra.

```python
def bar_range_position(
    data: pd.DataFrame,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    # BRP = (C - L) / (H - L) se H != L, else 0.5
    # Range [0, 1]: 0 = close no low, 1 = close no high
```

**Por que incluir:** Feature simples mas informativa; detecta rejeição de preço; base para candlestick patterns.

---

#### 13.6.2 `body_to_range_ratio`
**Referência:** MQL5 - Candlestick analysis

Ratio entre corpo e range total da vela.

```python
def body_to_range_ratio(
    data: pd.DataFrame,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    # BTR = |C - O| / (H - L)
    # BTR alto = vela de momentum
    # BTR baixo = indecisão (doji-like)
```

**Por que incluir:** Quantifica "qualidade" do candle; detecta indecisão vs. convicção; input útil para ML.

---

#### 13.6.3 `upper_shadow_ratio`
**Referência:** MQL5 - Candlestick component analysis

Ratio do shadow superior.

```python
def upper_shadow_ratio(
    data: pd.DataFrame,
) -> pd.Series:
    # USR = (H - max(O, C)) / (H - L)
    # Alto USR = rejeição de preços altos (bearish)
```

**Por que incluir:** Detecta selling pressure no topo; componente de patterns como shooting star.

---

#### 13.6.4 `lower_shadow_ratio`
**Referência:** MQL5 - Candlestick component analysis

Ratio do shadow inferior.

```python
def lower_shadow_ratio(
    data: pd.DataFrame,
) -> pd.Series:
    # LSR = (min(O, C) - L) / (H - L)
    # Alto LSR = rejeição de preços baixos (bullish)
```

**Por que incluir:** Detecta buying pressure no fundo; componente de patterns como hammer.

---

## 14. Priorização: Features MQL5

### Tier 1 (Fundamentais, Alto Valor)
1. `kaufman_efficiency_ratio` - Métrica limpa de tendência
2. `mean_reversion_half_life` - Complementa OU existente
3. `generalized_hurst_exponent` - Extensão do Hurst DFA
4. `tick_imbalance_proxy` - Order flow OHLCV

### Tier 2 (Úteis para Regime/ML)
5. `fractal_dimension_mincover` - Alternativa ao Hurst
6. `volatility_ratio` - Detecta squeeze/breakout
7. `chande_momentum_oscillator` - Base para VIDYA
8. `bar_range_position` - Candlestick feature

### Tier 3 (Especializadas)
9. `approximate_entropy` - Predecessor do SampEn
10. `cross_sample_entropy` - Pairs trading
11. `nrtr_channel` - Trading system
12. `body_to_range_ratio`, `upper/lower_shadow_ratio` - Candlestick ML

---

## 15. Comparativo: Features Acadêmicas vs. MQL5 Community

| Aspecto | Acadêmicas (Papers) | MQL5 Community |
|---------|---------------------|----------------|
| **Foco** | Rigor teórico | Aplicação prática |
| **Validação** | Peer review | Backtesting empírico |
| **Complexidade** | Alta (ex: roughness) | Moderada (ex: efficiency ratio) |
| **Implementação** | Python/R | MQL5 (C-like) |
| **Documentação** | Papers densos | Tutoriais passo-a-passo |
| **Uso** | Research, hedge funds | Retail traders |

**Conclusão:** A comunidade MQL5 oferece features complementares às acadêmicas, com foco em:
- Regime detection prático (Kaufman ER, half-life)
- Proxies de order flow para OHLCV
- Features de candlestick quantificadas
- Indicadores adaptativos (NRTR, FRAMA)

---

## Referências MQL5

25. MQL5 - "Implementing the Generalized Hurst Exponent and the Variance Ratio test"
26. MQL5 - "Evaluating the ability of Fractal index and Hurst exponent to predict financial time series"
27. MQL5 - "Building a Custom Market Regime Detection System"
28. MQL5 - "Grokking market memory through differentiation and entropy analysis"
29. MQL5 - "The NRTR indicator and trading modules"
30. MQL5 - "Comparing different types of moving averages in trading"
31. MQL5 - "Tick Buffer VWAP and Short-Window Imbalance Engine"
32. Kaufman, P. - "Trading Systems and Methods"
33. Chan, E. - "Algorithmic Trading: Winning Strategies and Their Rationale"
34. Chande, T. - "The New Technical Trader"

---

# APÊNDICE: Análise de Lookahead Bias

## ⚠️ Features com Risco de Lookahead Bias

As seguintes features propostas **requerem atenção especial na implementação** para evitar lookahead bias (uso de informação futura):

---

### 🔴 CRÍTICO - Lookahead na Fórmula Original

#### 1. `leverage_effect_measure` (Seção 8.3.2)
**Problema:** A fórmula original usa volatilidade futura:
```
LE = corr(r_t, RV_{t+horizon} - RV_t)
```
O termo `RV_{t+horizon}` representa a volatilidade **futura**, o que introduz lookahead bias direto.

**Solução:** Implementar versão **lagged** que mede a correlação histórica:
```python
# Versão correta (sem lookahead):
LE_t = corr(r_{t-window:t-horizon}, RV_{t-window+horizon:t})
# Calcula correlação entre retornos passados e mudanças de volatilidade subsequentes
# Sempre usando dados disponíveis até t
```

**Alternativa:** Usar como feature apenas para análise offline/research, não para trading em tempo real.

---

#### 2. `mean_reversion_strength` (Seção 13.1.5)
**Problema:** A fórmula usa variação futura do z-score:
```
MRS = -corr(z_t, Δz_{t+1})
```
O termo `Δz_{t+1} = z_{t+1} - z_t` usa informação do período **futuro**.

**Solução:** Implementar versão **lagged**:
```python
# Versão correta (sem lookahead):
MRS_t = -corr(z_{t-window-1:t-1}, Δz_{t-window:t})
# Calcula correlação entre z passado e variações subsequentes (já observadas)
```

---

### 🟡 ATENÇÃO - Requer Implementação Cuidadosa

#### 3. `medrv` (Seção 8.2.1)
**Problema:** A fórmula original usa retorno adjacente futuro:
```
MedRV = sum(med(|r_{t-1}|, |r_t|, |r_{t+1}|)²)
```
O termo `|r_{t+1}|` é futuro no contexto de cálculo rolling.

**Solução:** Na implementação rolling, usar mediana de retornos **passados**:
```python
# Versão correta para feature rolling:
# No ponto t, usar med(|r_{t-2}|, |r_{t-1}|, |r_t|)
# Ou aplicar shift(-1) após cálculo para alinhar corretamente
```

**Nota:** A fórmula original é para cálculo de volatilidade realizada de um dia completo (ex-post), não para features preditivas.

---

#### 4. `cusum_statistic` (Seção 3.5.1)
**Problema potencial:** Se μ e σ forem calculados sobre toda a janela:
```
CUSUM_t = sum(r_i - μ) / σ
```

**Verificação necessária:** Garantir que μ e σ sejam **expanding** ou **rolling backward-only**:
```python
# Correto: usar média/std até o ponto atual
μ_t = mean(r_{1:t})  # ou mean(r_{t-window:t})
σ_t = std(r_{1:t})   # ou std(r_{t-window:t})
```

---

#### 5. `har_rv_forecast` (Feature Existente)
**Verificação:** A implementação atual usa regressão rolling. Verificar que:
- Os coeficientes da regressão são estimados apenas com dados **anteriores** ao ponto de previsão
- A previsão `RV_{t+1|t}` usa apenas `RV_d`, `RV_w`, `RV_m` calculados até `t`

---

### 🟢 SEM RISCO - Features Seguras

As demais features propostas **não têm risco de lookahead bias** pois:

1. **Volatilidade Range-Based** (`parkinson`, `garman_klass`, `rogers_satchell`, `yang_zhang`):
   - Usam apenas OHLC da barra atual ou passadas

2. **Microstructure** (`amihud`, `roll_spread`, `corwin_schultz`, `vpin`, `order_flow_imbalance`):
   - Calculadas a partir de dados históricos

3. **Jump Detection** (`bipower_variation`, `jump_variation`, `realized_semivariance`, `signed_jump_variation`):
   - Usam retornos passados adjacentes `r_{t-1}` e `r_t`

4. **Entropy** (`sample_entropy`, `permutation_entropy`, `approximate_entropy`):
   - Calculadas sobre janela de dados passados

5. **Regime** (`variance_ratio`, `trend_intensity`, `runs_test`):
   - Usam apenas dados históricos

6. **Higher Moments** (`realized_skewness`, `realized_kurtosis`):
   - Calculados sobre retornos passados

7. **Tail Risk** (`value_at_risk`, `expected_shortfall`):
   - Baseados em quantis históricos

8. **Momentum** (`time_series_momentum`, `trend_strength`, `price_acceleration`, `kaufman_efficiency`):
   - Usam retornos e preços passados

9. **Candlestick Features** (`bar_range_position`, `body_to_range_ratio`, `shadow_ratios`):
   - Usam apenas OHLC da barra atual

10. **Autocorrelação** (`return_autocorrelation`, `absolute_return_autocorrelation`):
    - Calculadas sobre janela passada

---

## Checklist de Implementação Anti-Lookahead

Para cada feature implementada, verificar:

- [ ] **Rolling windows** usam apenas dados `[t-window, t]`, nunca `[t, t+window]`
- [ ] **Médias e desvios padrão** são calculados apenas com dados disponíveis até `t`
- [ ] **Correlações** entre séries usam alinhamento temporal correto
- [ ] **Previsões** são feitas com modelos treinados apenas em dados anteriores
- [ ] **Shifts** são aplicados corretamente (`shift(1)` para atrasar, `shift(-1)` para adiantar)
- [ ] **Testes unitários** verificam que `feature[t]` não muda quando `data[t+1:]` é modificado

---

## Recomendação Final

| Feature | Status | Ação |
|---------|--------|------|
| `leverage_effect_measure` | 🔴 Crítico | Reimplementar com lag ou usar apenas offline |
| `mean_reversion_strength` | 🔴 Crítico | Reimplementar com lag |
| `medrv` | 🟡 Atenção | Usar shift(-1) ou mediana backward |
| `cusum_statistic` | 🟡 Atenção | Verificar cálculo de μ e σ |
| `har_rv_forecast` | 🟡 Verificar | Confirmar rolling regression correta |
| **Demais 55 features** | 🟢 OK | Implementar normalmente |

**Nota:** As 2 features marcadas como 🔴 **não devem ser implementadas na forma original** para uso em backtesting ou trading. A versão lagged é aceitável mas tem interpretação diferente.
