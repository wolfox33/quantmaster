# MINRV (Minimum Realized Variance)

## Intuição

O **MinRV** é um estimador de variância robusto a *jumps*, usando o mínimo de retornos adjacentes em magnitude (reduzindo a influência de outliers).

## Definição

Para retornos `r_t`:

`MinRV_t = (π / (π - 2)) * min(|r_t|, |r_{t-1}|)^2`

## Uso

```python
from quantmaster.features.volatility import minrv

df["minrv"] = minrv(df)
```

## API

::: quantmaster.features.volatility.minrv
