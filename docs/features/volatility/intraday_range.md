# INTRADAY RANGE

## Intuição

O **intraday range** é uma proxy simples de volatilidade baseada no range intrabar `high/low`. Ele é útil para detectar mudanças de regime de volatilidade sem depender de retornos.

## Definição

`IR_t = ln(high_t / low_t)`

A feature retorna a média rolling de `IR_t` em janela `n`.

## Uso

```python
from quantmaster.features.volatility import intraday_range

df["intraday_range"] = intraday_range(df, window=20)
```

## API

::: quantmaster.features.volatility.intraday_range
