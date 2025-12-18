# Tail Risk Measure

## Intuição

A **Tail Risk Measure** segue a ideia de Kelly & Jiang (2014): compara a severidade média das perdas na cauda com um quantil de referência.

## Definição

Para retornos `r`, janela `n` e quantil `q`:

- `t = quantile(r, q)`
- `TRM = mean(r | r <= t) / t`

## Uso

```python
from quantmaster.features.risk import tail_risk_measure

df["trm"] = tail_risk_measure(df, window=60, quantile=0.05)
```

## API

::: quantmaster.features.risk.tail_risk_measure
