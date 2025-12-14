# RVOL (Relative Volume - log)

## Intuição

O RVOL mede quão alto (ou baixo) está o volume atual em relação ao seu comportamento recente.

A versão em log transforma uma razão multiplicativa em uma escala aditiva:

- Valores positivos indicam volume **acima** da média recente.
- Valores negativos indicam volume **abaixo** da média recente.

## Definição

Para uma janela `window`, definimos:

- `m_t = mean(volume_{t-window+1..t})`
- `rvol_t = log(volume_t / m_t)`

## Uso

```python
from quantmaster.features.volume import rvol

df["rvol_30"] = rvol(df, window=30)
```

## API

::: quantmaster.features.volume.rvol
