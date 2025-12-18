# Spread Z-Score

## Intuição

O **Spread Z-Score** é uma base clássica de *pairs trading*: mede quão “esticado” está o spread entre um ativo e um benchmark em unidades de desvio-padrão, usando uma janela móvel.

## Definição

Em uma janela `n`:

- `x = log(P_asset)`
- `y = log(P_benchmark)`
- `beta = Cov(x, y) / Var(y)` (estimado na janela)
- `spread = x - beta * y`
- `z = (spread - mean(spread)) / std(spread)`

## Uso

```python
from quantmaster.features.statistical import spread_zscore

z = spread_zscore(asset_df, benchmark_series, window=60)
```

## API

::: quantmaster.features.statistical.spread_zscore
