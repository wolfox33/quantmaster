# Trend Strength Autocorrelation

## Intuição

A **Trend Strength Autocorrelation** mede a autocorrelação rolling dos retornos (normalmente lag 1). Pode ser usada para diferenciar regimes de momentum (autocorrelação positiva) vs mean-reversion (autocorrelação negativa).

## Definição

Em janela `n` e defasagem `lag`:

`AC_t = corr(r_t, r_{t-lag})` (calculado rolling)

## Uso

```python
from quantmaster.features.trend import trend_strength_autocorr

df["ac_60_1"] = trend_strength_autocorr(df, window=60, lag=1)
```

## API

::: quantmaster.features.trend.trend_strength_autocorr
