# MEDRV (Median Realized Variance)

## Intuição

O **MedRV** é um estimador robusto a *jumps* baseado na mediana de retornos adjacentes em magnitude.

## Definição

A fórmula original usa `|r_{t+1}|`, o que pode introduzir *lookahead* em contextos rolling. Nesta biblioteca a versão é **backward-only**, usando a mediana de uma janela de 3 observações passada:

`MedRV_t = c * median(|r_{t-2}|, |r_{t-1}|, |r_t|)^2`

onde `c = π / (6 - 4√3 + π)`.

## Uso

```python
from quantmaster.features.volatility import medrv

df["medrv"] = medrv(df)
```

## API

::: quantmaster.features.volatility.medrv
