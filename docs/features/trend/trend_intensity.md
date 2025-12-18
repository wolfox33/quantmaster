# Trend Intensity

## Intuição

A **Trend Intensity** mede quão direcional é o movimento do preço dentro de uma janela.

- Perto de `1`: retornos na janela estão majoritariamente no mesmo sentido (tendência forte)
- Perto de `0`: retornos alternam sinal com frequência (movimento “choppy”)

## Definição

Para retornos (log) `r_t` e janela `n`:

`TI_t = |sum(r_{t-n+1..t})| / sum(|r_{t-n+1..t}|)`

## Uso

```python
from quantmaster.features.trend import trend_intensity

df["trend_intensity_20"] = trend_intensity(df, window=20)
```

## API

::: quantmaster.features.trend.trend_intensity
