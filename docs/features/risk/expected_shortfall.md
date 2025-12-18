# Expected Shortfall

## Intuição

O **Expected Shortfall (ES)** (também conhecido como CVaR) mede a perda média condicional nas piores realizações além do VaR.

## Definição

Para retornos `r`, janela `n` e confiança `c`:

- `alpha = 1 - c`
- `q = quantile(r, alpha)`
- `ES = -mean(r | r <= q)`

## Uso

```python
from quantmaster.features.risk import expected_shortfall

df["es"] = expected_shortfall(df, window=252, confidence=0.95)
```

## API

::: quantmaster.features.risk.expected_shortfall
