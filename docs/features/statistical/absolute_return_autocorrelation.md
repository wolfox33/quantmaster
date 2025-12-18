# Absolute Return Autocorrelation

## Intuição

A **Absolute Return Autocorrelation** mede autocorrelação dos retornos em módulo, sendo uma proxy clássica para **volatility clustering** (memória longa na volatilidade).

## Definição

Em uma janela `n` e lag `k`:

`AC_abs(k) = corr(|r_t|, |r_{t-k}|)`

## Uso

```python
from quantmaster.features.statistical import absolute_return_autocorrelation

df["acabs_60_1"] = absolute_return_autocorrelation(df, window=60, lag=1)
```

## API

::: quantmaster.features.statistical.absolute_return_autocorrelation
