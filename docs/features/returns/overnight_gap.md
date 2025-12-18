# Overnight Gap

## Intuição

O **Overnight Gap** captura o retorno entre o *close* do dia anterior e o *open* do dia atual. Ele é frequentemente associado a notícias e eventos fora do pregão.

## Definição

`gap_t = ln(open_t / close_{t-1})`

## Uso

```python
from quantmaster.features.returns import overnight_gap

df["overnight_gap"] = overnight_gap(df)
```

## API

::: quantmaster.features.returns.overnight_gap
