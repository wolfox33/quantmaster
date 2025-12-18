# Volume Volatility Ratio

## Intuição

O **Volume Volatility Ratio** compara atividade (média de volume) com incerteza (volatilidade de retornos) em uma janela móvel. Pode capturar regimes de “muito volume para pouca volatilidade” e vice-versa.

## Definição

Em uma janela `n`:

- `vol_mean = mean(volume)`
- `sigma = std(returns)`
- `VVR = log(vol_mean / sigma)`

## Uso

```python
from quantmaster.features.volume import volume_volatility_ratio

df["vvr_20"] = volume_volatility_ratio(df, window=20)
```

## API

::: quantmaster.features.volume.volume_volatility_ratio
