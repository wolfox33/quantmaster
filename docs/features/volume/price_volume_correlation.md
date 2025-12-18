# Price Volume Correlation

## Intuição

A **Price Volume Correlation** mede a correlação entre retornos do preço e o nível (log) de volume em uma janela móvel, capturando se movimentos de preço tendem a ocorrer junto com maior (ou menor) atividade.

## Definição

Em uma janela `n`:

- `r_t` = retorno do preço (log ou pct)
- `v_t` = `log(volume_t)`
- `PVC = corr(r, v)`

## Uso

```python
from quantmaster.features.volume import price_volume_correlation

df["pvc_20"] = price_volume_correlation(df, window=20)
```

## API

::: quantmaster.features.volume.price_volume_correlation
