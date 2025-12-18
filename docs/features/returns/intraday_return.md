# Intraday Return

## Intuição

O **Intraday Return** mede o retorno do *open* para o *close* no mesmo dia (movimento “durante o pregão”).

## Definição

`r_intra_t = ln(close_t / open_t)`

## Uso

```python
from quantmaster.features.returns import intraday_return

df["intraday_return"] = intraday_return(df)
```

## API

::: quantmaster.features.returns.intraday_return
