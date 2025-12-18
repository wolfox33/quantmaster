# Fractal Dimension (MinCover)

## Intuição

A **Fractal Dimension (MinCover)** estima a dimensão fractal local usando um método de cobertura mínima, sendo uma alternativa prática para detectar “trendiness” vs. comportamento mais próximo de random walk.

## Definição

Em uma janela, para escalas `k`, particione a série em blocos de tamanho `k` e some os ranges em cada bloco. Ajuste uma regressão em log-log e derive:

`D = 2 - slope`

## Uso

```python
from quantmaster.features.statistical import fractal_dimension_mincover

df["fd"] = fractal_dimension_mincover(df, window=100, max_scale=10)
```

## API

::: quantmaster.features.statistical.fractal_dimension_mincover
