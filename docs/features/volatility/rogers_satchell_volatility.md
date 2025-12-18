# ROGERS–SATCHELL VOLATILITY

## Intuição

O estimador **Rogers–Satchell** é um estimador range-based que permanece útil em presença de drift (tendências), sendo frequentemente usado como componente em estimadores mais completos (ex: Yang–Zhang).

## Definição

Seja:

- `HO_t = ln(high_t/open_t)`
- `HC_t = ln(high_t/close_t)`
- `LO_t = ln(low_t/open_t)`
- `LC_t = ln(low_t/close_t)`

A variância RS diária é:

`RS_t = HC_t * HO_t + LC_t * LO_t`

A série é agregada via média rolling em janela `n`, e a volatilidade é `sqrt(max(RS, 0))`.

## Uso

```python
from quantmaster.features.volatility import rogers_satchell_volatility

df["rs"] = rogers_satchell_volatility(df, window=20)
```

## API

::: quantmaster.features.volatility.rogers_satchell_volatility
