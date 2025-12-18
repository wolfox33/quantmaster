# PARKINSON VOLATILITY

## Intuição

O estimador de **Parkinson** usa apenas `high` e `low` para estimar volatilidade. Ele é mais eficiente que estimadores baseados apenas em `close-to-close` quando não há drift significativo.

## Definição

Seja `HL_t = ln(high_t / low_t)`. A variância estimada em janela `n` é:

`Var = E[HL_t^2] / (4 * ln(2))`

A volatilidade é `sqrt(Var)`.

## Uso

```python
from quantmaster.features.volatility import parkinson_volatility

df["parkinson"] = parkinson_volatility(df, window=20)
```

## API

::: quantmaster.features.volatility.parkinson_volatility
