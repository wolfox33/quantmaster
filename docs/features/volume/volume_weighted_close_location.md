# Volume Weighted Close Location

## Intuição

A **Volume Weighted Close Location** agrega o CLV ponderando por volume em uma janela móvel, destacando candles “mais relevantes” (com maior volume).

## Definição

Em uma janela `n`:

`VWCL = sum(CLV * volume) / sum(volume)`

O output fica em `[-1, 1]`.

## Uso

```python
from quantmaster.features.volume import volume_weighted_close_location

df["vwcl_20"] = volume_weighted_close_location(df, window=20)
```

## API

::: quantmaster.features.volume.volume_weighted_close_location
