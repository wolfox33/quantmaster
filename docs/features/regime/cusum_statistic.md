# CUSUM Statistic

## Intuição

A estatística **CUSUM** acumula desvios do retorno em relação à média, padronizados pelo desvio-padrão, para sinalizar possíveis mudanças estruturais.

## Definição

Para retornos (log) `r_t`, e estimativas *rolling* backward-only em janela `n`:

- `z_t = (r_t - μ_t) / σ_t`
- `CUSUM_t = sum(z_{t-n+1..t})`

## Uso

```python
from quantmaster.features.regime import cusum_statistic

df["cusum_60"] = cusum_statistic(df, window=60)
```

## API

::: quantmaster.features.regime.cusum_statistic
