# create_all

## Intuição

Quando você está explorando a biblioteca ou construindo um dataset para ML, é comum querer gerar várias features de uma vez.

`create_all` faz isso automaticamente: ela percorre as features públicas do `quantmaster.features` (via `__all__`), calcula cada feature com parâmetros default quando possível e adiciona o resultado ao `DataFrame`.

## Comportamento

- Retorna um novo `DataFrame` por padrão (`inplace=False`).
- Não sobrescreve colunas existentes por padrão (`overwrite=False`).
- Features que exigem argumentos extras (ex.: `benchmark`) são puladas se você não fornecer.
- Features com dependências opcionais (ex.: `path_signature_features`) podem ser puladas automaticamente.
- Você pode controlar o comportamento com `include`, `exclude` e `errors`.

## Uso

```python
from quantmaster.features.utils import create_all

# adiciona todas as features possíveis com defaults
# (as que dependem de benchmark/dependência opcional podem ser puladas)
df = create_all(df)

# com benchmark para habilitar betas e spread_zscore
# df = create_all(df, benchmark=spx_close)

# gerar apenas um subconjunto
# df = create_all(df, include=["rsi", "har_rv"])
```

## API

::: quantmaster.features.utils.create_all
