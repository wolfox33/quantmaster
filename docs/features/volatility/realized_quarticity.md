# REALIZED QUARTICITY

## Intuição

A **Realized Quarticity (RQ)** é uma medida de quarto momento dos retornos e é usada para estimar a variância do erro de medição da volatilidade realizada, aparecendo em extensões como HARQ.

## Definição

Para retornos `r_t`:

`RQ_t = r_t^4`

## Uso

```python
from quantmaster.features.volatility import realized_quarticity

df["rq"] = realized_quarticity(df)
```

## API

::: quantmaster.features.volatility.realized_quarticity
