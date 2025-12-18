# JUMP VARIATION

## Intuição

A **Jump Variation (JV)** tenta isolar a componente de *jumps* (saltos/discontinuidades) na variação total, usando a diferença entre **RV** (variância realizada) e **BV** (bipower variation).

## Definição

`JV_t = max(RV_t - BV_t, 0)`

## Uso

```python
from quantmaster.features.volatility import jump_variation

df["jv"] = jump_variation(df)
```

## API

::: quantmaster.features.volatility.jump_variation
