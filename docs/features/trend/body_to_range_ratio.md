# Body to Range Ratio

## Intuição

Quantifica o quanto do range total da barra foi “corpo” (`|close - open|`). Ajuda a separar candles de convicção (corpo grande) de indecisão (doji-like).

## Definição

- `BTR = |close - open| / (high - low)`

O resultado fica em `[0, 1]`.

## Uso

```python
from quantmaster.features.trend import body_to_range_ratio

btr = body_to_range_ratio(ohlc_df)
```

## API

::: quantmaster.features.trend.body_to_range_ratio
