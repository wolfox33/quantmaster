# Lower Shadow Ratio

## Intuição

Mede a proporção do *shadow* inferior da vela, capturando rejeição de preços baixos.

## Definição

- `LSR = (min(open, close) - low) / (high - low)`

O resultado fica em `[0, 1]`.

## Uso

```python
from quantmaster.features.trend import lower_shadow_ratio

lsr = lower_shadow_ratio(ohlc_df)
```

## API

::: quantmaster.features.trend.lower_shadow_ratio
