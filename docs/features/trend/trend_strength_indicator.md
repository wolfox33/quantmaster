# Trend Strength Indicator

## Intuição

O **Trend Strength Indicator (TSI)** combina múltiplos horizontes de momentum (time-series) e normaliza pelo risco (volatilidade) para produzir um score contínuo de força de tendência.

## Definição

Para cada janela `w` em `windows`:

- `r_w = ln(P_t) - ln(P_{t-w})`
- `σ_w = std(Δln(P), window=w) * sqrt(w)`
- `signal_w = sign(r_w) * min(|r_w/σ_w|, 2)`

O indicador final é a média dos `signal_w`.

## Uso

```python
from quantmaster.features.trend import trend_strength_indicator

df["tsi"] = trend_strength_indicator(df, windows=[21, 63, 126, 252])
```

## API

::: quantmaster.features.trend.trend_strength_indicator
