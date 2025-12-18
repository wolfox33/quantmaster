# Time-Series Momentum

## Intuição

O **Time-Series Momentum** (Moskowitz, Ooi & Pedersen) mede o momentum usando apenas o histórico do próprio ativo (não é cross-sectional).

Uma forma comum é escalar o retorno de `lookback` por uma estimativa de volatilidade (para comparar ativos/regimes distintos).

## Definição

Usando preço `P_t` e retornos log:

- `r_total = ln(P_t) - ln(P_{t-lookback})`
- `σ_t = std(Δln(P), window=volatility_window)`
- `TSMOM_t = r_total / σ_t`

## Uso

```python
from quantmaster.features.momentum import time_series_momentum

df["tsmom"] = time_series_momentum(df, lookback=252, volatility_window=60)
```

## API

::: quantmaster.features.momentum.time_series_momentum
