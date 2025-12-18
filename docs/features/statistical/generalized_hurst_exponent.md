# Generalized Hurst Exponent

## Intuição

O **Generalized Hurst Exponent (GHE)** generaliza o Hurst para diferentes ordens de momentos (`q`), capturando propriedades multi-escala e memória longa.

## Definição

Para lags `tau = 1..max_lag`, compute:

`K_q(tau) = E(|X(t+tau) - X(t)|^q)`

Ajuste uma regressão em log-log:

`log(K_q(tau)) = a + (q*H(q)) * log(tau)`

Então:

`H(q) = slope / q`

## Uso

```python
from quantmaster.features.statistical import generalized_hurst_exponent

df["ghe"] = generalized_hurst_exponent(df, window=100, q=2.0, max_lag=10)
```

## API

::: quantmaster.features.statistical.generalized_hurst_exponent
