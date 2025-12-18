# GARMAN–KLASS VOLATILITY

## Intuição

O estimador **Garman–Klass** combina `open`, `high`, `low`, `close` para estimar volatilidade diária com maior eficiência do que usar apenas retornos `close-to-close`.

## Definição

Seja `HL_t = ln(high_t/low_t)` e `CO_t = ln(close_t/open_t)`. A variância Garman–Klass é:

`GK_t = 0.5 * HL_t^2 - (2*ln(2)-1) * CO_t^2`

A série é agregada via média rolling em janela `n`, e a volatilidade é `sqrt(max(GK, 0))`.

## Uso

```python
from quantmaster.features.volatility import garman_klass_volatility

df["gk"] = garman_klass_volatility(df, window=20)
```

## API

::: quantmaster.features.volatility.garman_klass_volatility
