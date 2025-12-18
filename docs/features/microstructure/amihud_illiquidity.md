# AMIHUD ILLIQUILITY

## Intuição

A **Amihud Illiquidity** é uma proxy clássica de *price impact*: quanto o preço se move (em magnitude) por unidade de volume negociado. Valores maiores indicam menor liquidez.

## Definição

Para um retorno diário `r_t` e *dollar volume* `DV_t = close_t * volume_t`:

`ILLIQ_t = mean(|r_t| / DV_t)`, calculado como média rolling em uma janela `n`.

## Uso

```python
from quantmaster.features.microstructure import amihud_illiquidity

df["amihud_illiquidity_20"] = amihud_illiquidity(df, window=20)
```

## API

::: quantmaster.features.microstructure.amihud_illiquidity
