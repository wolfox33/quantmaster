# RSI (Relative Strength Index)

## Intuição

O RSI é um oscilador de momentum que mede a razão entre ganhos e perdas médios em uma janela, produzindo valores entre 0 e 100.

- Valores altos (ex: > 70) são frequentemente interpretados como **sobrecompra**.
- Valores baixos (ex: < 30) são frequentemente interpretados como **sobrevenda**.

A interpretação depende do ativo, horizonte e do regime de volatilidade; por isso, o RSI é melhor usado como *feature* em modelos ou como parte de um conjunto de regras.

## Uso

```python
from quantmaster.features.momentum import rsi

df["rsi_14"] = rsi(df, window=14)
```

## API

::: quantmaster.features.momentum.rsi
