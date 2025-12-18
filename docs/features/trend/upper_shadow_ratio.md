# Upper Shadow Ratio

## Intuição

Mede a proporção do *shadow* superior da vela, capturando rejeição de preços altos.

## Definição

- `USR = (high - max(open, close)) / (high - low)`

O resultado fica em `[0, 1]`.

## Uso

```python
from quantmaster.features.trend import upper_shadow_ratio

usr = upper_shadow_ratio(ohlc_df)
```

## API

::: quantmaster.features.trend.upper_shadow_ratio
