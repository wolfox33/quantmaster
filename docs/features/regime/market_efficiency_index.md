# Market Efficiency Index

## Intuição

O **Market Efficiency Index (MEI)** é um índice composto simples para sintetizar sinais de ineficiência. Ele combina:

- Autocorrelação de 1 lag dos retornos
- Desvio do variance ratio em relação a 1

## Definição

Em janela `n`:

- `MEI = 1 - |AC(1)| - 0.5 * |VR(q) - 1|`

## Uso

```python
from quantmaster.features.regime import market_efficiency_index

df["mei_60_5"] = market_efficiency_index(df, window=60, holding_period=5)
```

## API

::: quantmaster.features.regime.market_efficiency_index
