# Return Autocorrelation

## Intuição

A **Return Autocorrelation** mede a correlação entre retornos atuais e retornos defasados (`lag`) dentro de uma janela. Ajuda a identificar regimes de momentum (autocorr positiva) ou reversão (autocorr negativa).

## Definição

Em uma janela `n` e lag `k`:

`AC(k) = corr(r_t, r_{t-k})`

## Uso

```python
from quantmaster.features.statistical import return_autocorrelation

df["ac_60_1"] = return_autocorrelation(df, window=60, lag=1)
```

## API

::: quantmaster.features.statistical.return_autocorrelation
