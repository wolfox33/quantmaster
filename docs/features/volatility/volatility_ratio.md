# VOLATILITY RATIO

## Intuição

O **volatility ratio** compara o True Range atual com um ATR (média do True Range). Valores maiores que 1 indicam expansão de volatilidade; valores menores sugerem contração.

## Definição

- `TR_t = max(high_t-low_t, |high_t-close_{t-1}|, |low_t-close_{t-1}|)`
- `ATR_t = mean(TR_{t-n+1:t})`

A feature retorna `VR_t = TR_t / ATR_t`.

## Uso

```python
from quantmaster.features.volatility import volatility_ratio

df["vr"] = volatility_ratio(df, window=14)
```

## API

::: quantmaster.features.volatility.volatility_ratio
