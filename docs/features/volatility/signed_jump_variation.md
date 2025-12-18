# SIGNED JUMP VARIATION

## Intuição

A **Signed Jump Variation** separa a componente de jumps em partes associadas a retornos positivos e negativos, capturando assimetria (jumps “bons” vs “ruins”).

## Definição

Com `RSV+`, `RSV-` e `BV`:

- `ΔJ+_t = max(RSV+_t - BV_t/2, 0)`
- `ΔJ-_t = max(RSV-_t - BV_t/2, 0)`

A função retorna um `DataFrame` com as colunas `signed_jump_pos` e `signed_jump_neg`.

## Uso

```python
from quantmaster.features.volatility import signed_jump_variation

out = signed_jump_variation(df)
# df = df.join(out)
```

## API

::: quantmaster.features.volatility.signed_jump_variation
