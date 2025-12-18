# Realized Kurtosis

## Intuição

A **Realized Kurtosis** mede o peso de caudas (fat tails) dos retornos realizados dentro de uma janela, usando uma forma normalizada baseada no 4º momento.

## Definição

Para retornos `r_t` em uma janela `n`:

- `M2 = sum(r^2)`
- `M4 = sum(r^4)`
- `RKurt = n * M4 / (M2^2)`

## Uso

```python
from quantmaster.features.statistical import realized_kurtosis

df["rkurt_20"] = realized_kurtosis(df, window=20)
```

## API

::: quantmaster.features.statistical.realized_kurtosis
