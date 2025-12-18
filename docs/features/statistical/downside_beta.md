# Downside Beta

## Intuição

O **Downside Beta** mede a sensibilidade do ativo ao benchmark **apenas** em períodos em que o benchmark está em queda (retorno negativo). É uma proxy de risco assimétrico.

## Definição

Em uma janela `n`, usando apenas observações com `r_b < 0`:

`beta_down = Cov(r_a, r_b | r_b < 0) / Var(r_b | r_b < 0)`

## Uso

```python
from quantmaster.features.statistical import downside_beta

beta_d = downside_beta(df, benchmark_series, window=60)
```

## API

::: quantmaster.features.statistical.downside_beta
