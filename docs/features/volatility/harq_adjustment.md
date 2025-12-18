# HARQ ADJUSTMENT

## Intuição

O **HARQ Adjustment** é um fator baseado em `RQ/RV²` que serve como proxy para a intensidade do erro de medição da RV. Ele é usado em variantes do HAR (HARQ) para ajustar dinamicamente previsões.

## Definição

Em uma janela `n`:

`adj_t = mean(RQ)_n / mean(RV)_n^2`

## Uso

```python
from quantmaster.features.volatility import harq_adjustment

df["harq_adjustment_22"] = harq_adjustment(df, window=22)
```

## API

::: quantmaster.features.volatility.harq_adjustment
