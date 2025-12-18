# ROLL SPREAD

## Intuição

O **Roll Spread** (Roll, 1984) estima o spread efetivo de bid-ask de forma implícita a partir da autocovariância (tipicamente negativa) dos retornos. É útil quando não há dados de book/bid-ask.

## Definição

Com retornos `r_t`:

`spread_t = 2 * sqrt(max(0, -Cov(r_t, r_{t-1})))`, onde a covariância é estimada em janela rolling `n`.

## Uso

```python
from quantmaster.features.microstructure import roll_spread

df["roll_spread_20"] = roll_spread(df, window=20)
```

## API

::: quantmaster.features.microstructure.roll_spread
