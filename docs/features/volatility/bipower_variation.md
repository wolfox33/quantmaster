# BIPOWER VARIATION

## Intuição

A **Bipower Variation (BV)** é uma estimativa da variação contínua (difusiva) da volatilidade, sendo mais robusta à presença de *jumps* do que a variância realizada (RV).

## Definição

Para retornos `r_t`:

`BV_t = (π/2) * |r_t| * |r_{t-1}|`

## Uso

```python
from quantmaster.features.volatility import bipower_variation

df["bv"] = bipower_variation(df)
```

## API

::: quantmaster.features.volatility.bipower_variation
