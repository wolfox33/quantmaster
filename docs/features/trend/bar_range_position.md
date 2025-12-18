# Bar Range Position

## Intuição

Mede onde o `close` ficou dentro do range da barra (`low`–`high`). É uma feature simples e muito informativa para *price action*.

## Definição

- Se `high != low`: `BRP = (close - low) / (high - low)`
- Caso contrário: `BRP = 0.5`

O resultado fica em `[0, 1]`.

## Uso

```python
from quantmaster.features.trend import bar_range_position

brp = bar_range_position(ohlc_df)
```

## API

::: quantmaster.features.trend.bar_range_position
