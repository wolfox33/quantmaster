# REALIZED SEMIVARIANCE

## Intuição

A **Realized Semivariance** decompõe a variância realizada em componentes de retornos positivos e negativos, permitindo medir separadamente a volatilidade “boa” e “ruim”.

## Definição

Para retornos `r_t`:

- `RSV+_t = r_t^2 * I(r_t > 0)`
- `RSV-_t = r_t^2 * I(r_t < 0)`

A função retorna um `DataFrame` com as colunas `rsv_pos` e `rsv_neg`.

## Uso

```python
from quantmaster.features.volatility import realized_semivariance

out = realized_semivariance(df)
# df = df.join(out)
```

## API

::: quantmaster.features.volatility.realized_semivariance
