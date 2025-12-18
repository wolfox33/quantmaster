# LOG VOLATILITY INCREMENT

## Intuição

O **Log Volatility Increment** é uma série de incrementos do log da volatilidade (aqui, do log da RV suavizada), usada como building block para análises de rough volatility e estimação de roughness.

## Definição

1. Calcula `RV_t`.
2. Suaviza com média rolling em janela `window`:

`\overline{RV}_t = mean(RV)_window`

3. Define:

`ΔlogV_t = log(\overline{RV}_t) - log(\overline{RV}_{t-lag})`

## Uso

```python
from quantmaster.features.volatility import log_volatility_increment

df["log_volatility_increment_20_1"] = log_volatility_increment(df, window=20, lag=1)
```

## API

::: quantmaster.features.volatility.log_volatility_increment
