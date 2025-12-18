# CORWIN–SCHULTZ SPREAD

## Intuição

O **Corwin–Schultz Spread** (2012) estima o bid-ask spread a partir de preços **high/low**, explorando a diferença entre ranges de 1 dia e de 2 dias. É uma alternativa ao Roll Spread e costuma funcionar melhor em alguns cenários usando apenas OHLC.

## Definição

A implementação segue a forma usual do estimador:

- Computa `beta` a partir de `log(high/low)^2` em dois dias consecutivos
- Computa `gamma` a partir do range de 2 dias (max(high), min(low))
- Obtém `alpha` e converte para spread via

`S = 2*(exp(alpha)-1)/(1+exp(alpha))`

O retorno da função é a média rolling de `S` em janela `n`.

## Uso

```python
from quantmaster.features.microstructure import corwin_schultz_spread

df["corwin_schultz_spread_20"] = corwin_schultz_spread(df, window=20)
```

## API

::: quantmaster.features.microstructure.corwin_schultz_spread
