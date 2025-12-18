# RELATIVE SPREAD PROXY

## Intuição

A **Relative Spread Proxy** é uma proxy de spread baseada em OHLC (Abdi & Ranaldo, 2017). A ideia é obter uma medida de spread (em unidades de preço) e normalizar pelo `close`, produzindo algo comparável entre ativos/níveis de preço.

## Definição

A função usa o termo:

`X_t = (H_t - C_t)(H_t - O_t) + (L_t - C_t)(L_t - O_t)`

Depois:

`spread_abs_t = 2 * sqrt(max(0, mean(X_t)))`

E normaliza:

`relative_spread_t = spread_abs_t / C_t`

## Uso

```python
from quantmaster.features.microstructure import relative_spread_proxy

df["relative_spread_proxy_20"] = relative_spread_proxy(df, window=20)
```

## API

::: quantmaster.features.microstructure.relative_spread_proxy
