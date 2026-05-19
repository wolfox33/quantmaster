# Feature Status

## Fonte única de status

O status de cada feature é definido em:

- `src/quantmaster/feature_status.py`

Status suportados:

- `approved`
- `experimental`
- `parked`
- `rejected`

## Regra no `create_all`

`create_all` executa apenas features `approved` por padrão.

Para incluir outras classes:

```python
from quantmaster.features.utils import create_all

df = create_all(df, include_statuses=["approved", "experimental"])
```

## Objetivo de governança

- Evitar inflar a API com features ainda não validadas.
- Permitir ciclo de pesquisa sem quebrar produção.
- Tornar explícita a transição de `experimental` para `approved`.

