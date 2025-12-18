# Kaufman Efficiency Ratio

## Intuição

O **Kaufman Efficiency Ratio (ER)** mede quão “eficiente” foi o movimento do preço dentro de uma janela.

- Perto de `1`: o preço se moveu de forma direcional (pouco vai-e-vem)
- Perto de `0`: muito ruído / *chop* (muito vai-e-vem para pouco deslocamento líquido)

## Definição

Para preço `P_t` e janela `n`:

- `Change = |P_t - P_{t-n}|`
- `Volatility = sum_{i=t-n+1..t} |P_i - P_{i-1}|`
- `ER = Change / Volatility`

## Uso

```python
from quantmaster.features.trend import kaufman_efficiency_ratio

df["ker_10"] = kaufman_efficiency_ratio(df, window=10)
```

## API

::: quantmaster.features.trend.kaufman_efficiency_ratio
