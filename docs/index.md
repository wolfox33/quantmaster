# Quantmaster

## Objetivo

Padronizar e centralizar features quantitativas (OHLCV -> features) para uso em pipelines de pesquisa e algotrading.

## API

- Features são funções puras que recebem `pandas.DataFrame` (OHLCV) e retornam `pandas.Series` ou `pandas.DataFrame`.
- Você decide como adicionar no `df`:

```python
df["rsi_10"] = rsi(df, window=10)
df = df.join(har_rv(df))

# ou gerar várias features de uma vez (com defaults)
df = create_all(df)
```
