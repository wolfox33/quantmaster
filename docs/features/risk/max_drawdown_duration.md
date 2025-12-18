# Max Drawdown Duration

## Intuição

A **Max Drawdown Duration** mede por quanto tempo o preço permaneceu abaixo de seu pico anterior (dentro de uma janela), capturando a “duração” do período de *underwater*.

## Definição

Em uma janela `n`, compute o máximo acumulado (pico) e o drawdown.

A duração é o maior número de períodos consecutivos com drawdown `< 0`.

## Uso

```python
from quantmaster.features.risk import max_drawdown_duration

df["mdd_dur"] = max_drawdown_duration(df, window=252)
```

## API

::: quantmaster.features.risk.max_drawdown_duration
