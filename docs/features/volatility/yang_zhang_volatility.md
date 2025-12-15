# YANG–ZHANG VOLATILITY

## Intuição

O estimador **Yang–Zhang** combina informação de:

- variação entre barras (`open_t` vs `close_{t-1}`),
- retorno intrabar (`close_t` vs `open_t`),
- range intrabar (via termo Rogers–Satchell).

Como feature, ele tende a capturar **regimes de volatilidade** usando mais informação do que retornos `close-to-close`.

## Definição

Seja:

- `o_t = ln(open_t / close_{t-1})`
- `c_t = ln(close_t / open_t)`
- `RS_t = ln(high_t/open_t)*ln(high_t/close_t) + ln(low_t/open_t)*ln(low_t/close_t)`

A variância Yang–Zhang em uma janela `n` é:

`YZ_var = Var(o_t) + k * Var(c_t) + (1-k) * E[RS_t]`

com:

`k = 0.34 / (1.34 + (n+1)/(n-1))`

E a volatilidade é `sqrt(YZ_var)`.

## Uso

```python
from quantmaster.features.volatility import yang_zhang_volatility

df["yz"] = yang_zhang_volatility(df, window=20)
```

## API

::: quantmaster.features.volatility.yang_zhang_volatility
