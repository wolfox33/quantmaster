# Variance Ratio

## Intuição

O **Variance Ratio** (Lo & MacKinlay) mede a relação entre a variância de retornos em um horizonte maior e a variância de retornos de 1 período. Em geral:

- Valores `> 1` sugerem persistência/momentum.
- Valores `< 1` sugerem reversão à média.

## Definição

Para um *holding period* `q` e janela `n`:

- `VR(q) = Var(r_t(q)) / (q * Var(r_t))`

onde `r_t(q)` é o retorno em `q` períodos (usando apenas dados passados).

## Uso

```python
from quantmaster.features.regime import variance_ratio

df["vr_120_5"] = variance_ratio(df, window=120, holding_period=5)
```

## API

::: quantmaster.features.regime.variance_ratio
