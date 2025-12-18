# Information Discreteness

## Intuição

A **Information Discreteness** (Da, Gurun & Warachka) tenta capturar se a informação chega via muitos pequenos movimentos (mais “contínuo”) ou poucos grandes movimentos (mais “discreto”).

## Definição

Em uma janela `n`:

- `r_total = sum(r_i)`
- `s_total = sign(r_total)`
- `ID = s_total * ( %dias com sign(r_i)=s_total  -  %dias com sign(r_i)=-s_total )`

O output fica em `[-1, 1]`.

## Uso

```python
from quantmaster.features.statistical import information_discreteness

df["id_20"] = information_discreteness(df, window=20)
```

## API

::: quantmaster.features.statistical.information_discreteness
