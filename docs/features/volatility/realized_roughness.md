# REALIZED ROUGHNESS

## Intuição

A **Realized Roughness** estima o expoente de Hurst da volatilidade (tipicamente via variação dos incrementos do log da volatilidade em diferentes lags). Em mercados financeiros, valores de `H < 0.5` são interpretados como evidência de *rough volatility*.

## Definição

A implementação segue a ideia de Gatheral, Jaisson & Rosenbaum (2018):

1. Computa `RV_t` e `log(RV_t)`.
2. Para cada `lag` em `lags`, calcula:

`m(lag)_t = mean( (log(RV_t) - log(RV_{t-lag}))^2 )` em janela rolling `window`.

3. Em cada tempo `t`, estima a inclinação `β` na regressão:

`log(m(lag)_t) = a + β * log(lag)`

Então:

`H_t = β / 2`

## Uso

```python
from quantmaster.features.volatility import realized_roughness

df["realized_roughness_60"] = realized_roughness(df, window=60, lags=[1, 2, 5, 10])
```

## API

::: quantmaster.features.volatility.realized_roughness
