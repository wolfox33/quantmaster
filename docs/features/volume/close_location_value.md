# Close Location Value

## Intuição

O **Close Location Value (CLV)** mede onde o fechamento ocorreu dentro do range (high-low) do candle. Valores próximos de `+1` indicam fechamento perto da máxima; próximos de `-1`, perto da mínima.

## Definição

Para cada barra:

`CLV = ((close - low) - (high - close)) / (high - low)`

O output é limitado a `[-1, 1]`.

## Uso

```python
from quantmaster.features.volume import close_location_value

df["clv"] = close_location_value(df)
```

## API

::: quantmaster.features.volume.close_location_value
