# Rolling Beta

## Intuição

O **Rolling Beta** mede a sensibilidade do retorno do ativo em relação ao retorno de um benchmark (CAPM), estimado em uma janela móvel.

## Definição

Em uma janela `n`, com retornos `r_a` (ativo) e `r_b` (benchmark):

`beta = Cov(r_a, r_b) / Var(r_b)`

## Uso

```python
from quantmaster.features.statistical import rolling_beta

beta = rolling_beta(df, benchmark_series, window=60)
```

## API

::: quantmaster.features.statistical.rolling_beta
