# SHAR COMPONENTS

## Intuição

O **SHAR Components** é uma extensão do HAR que usa semivariâncias (`RSV+` e `RSV-`) em múltiplos horizontes (diário/semanal/mensal) para capturar assimetria entre volatilidade positiva e negativa.

## Definição

A função retorna 6 colunas:

- `rsv_pos_d`, `rsv_neg_d`
- `rsv_pos_w`, `rsv_neg_w`
- `rsv_pos_m`, `rsv_neg_m`

onde as componentes semanal/mensal são médias rolling com `weekly_window` e `monthly_window`.

## Uso

```python
from quantmaster.features.volatility import shar_components

out = shar_components(df, weekly_window=5, monthly_window=22)
# df = df.join(out)
```

## API

::: quantmaster.features.volatility.shar_components
