# Mean Reversion Half-Life

## Intuição

A **Mean Reversion Half-Life** estima o tempo (em períodos) para um desvio decair pela metade, a partir de uma regressão simples que aproxima um processo tipo OU.

## Definição

Em uma janela `n`, ajuste:

`Δp_t = λ * p_{t-1} + ε_t`

Se `λ < 0`, a meia-vida é:

`HL = -ln(2) / λ`

## Uso

```python
from quantmaster.features.statistical import mean_reversion_half_life

df["hl_60"] = mean_reversion_half_life(df, window=60)
```

## API

::: quantmaster.features.statistical.mean_reversion_half_life
