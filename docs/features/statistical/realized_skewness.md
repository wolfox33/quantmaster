# Realized Skewness

## Intuição

A **Realized Skewness** é uma medida de assimetria baseada em retornos realizados dentro de uma janela. Ela captura se, dentro da janela, a distribuição de retornos tende a ter cauda mais pesada para cima ou para baixo.

## Definição

Para retornos `r_t` em uma janela `n`:

- `M2 = sum(r^2)`
- `M3 = sum(r^3)`
- `RSkew = sqrt(n) * M3 / (M2^(3/2))`

## Uso

```python
from quantmaster.features.statistical import realized_skewness

df["rskew_20"] = realized_skewness(df, window=20)
```

## API

::: quantmaster.features.statistical.realized_skewness
