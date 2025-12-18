# Price Acceleration

## Intuição

A **Price Acceleration** é uma proxy simples da “segunda derivada” do preço: mede se o momentum (retorno) está aumentando ou diminuindo em relação ao que era `window` períodos atrás.

## Definição

Com retornos log `r_t = ln(P_t) - ln(P_{t-1})`:

`Accel_t = (r_t - r_{t-window}) / window`

## Uso

```python
from quantmaster.features.trend import price_acceleration

df["price_acceleration_20"] = price_acceleration(df, window=20)
```

## API

::: quantmaster.features.trend.price_acceleration
