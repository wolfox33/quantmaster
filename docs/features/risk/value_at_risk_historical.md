# Value at Risk (Historical)

## Intuição

O **Value at Risk (VaR) histórico** estima, a partir de uma janela de retornos passados, a perda (magnitude) que não deve ser excedida com uma determinada confiança.

## Definição

Para retornos `r` em uma janela `n` e confiança `c`:

- `alpha = 1 - c`
- `VaR = -quantile(r, alpha)`

## Uso

```python
from quantmaster.features.risk import value_at_risk_historical

df["var"] = value_at_risk_historical(df, window=252, confidence=0.95)
```

## API

::: quantmaster.features.risk.value_at_risk_historical
