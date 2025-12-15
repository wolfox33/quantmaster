# HURST (DFA)

## Intuição

O **Hurst exponent** é um resumo do grau de persistência de uma série temporal:

- `H > 0.5`: comportamento mais persistente (tendência / “trending”).
- `H < 0.5`: comportamento mais anti-persistente (reversão à média).

Para dados financeiros com não-estacionaridade e drift, uma forma robusta de estimar H é via **DFA (Detrended Fluctuation Analysis)**.

## Definição

A DFA consiste em:

1. (Opcional) trabalhar com `log(price)`.
2. Construir o perfil (soma cumulativa) da série de-meaned dentro de uma janela.
3. Para vários tamanhos de caixa `s`, ajustar uma tendência linear e medir a flutuação RMS `F(s)`.
4. Estimar `H` como o coeficiente angular da regressão `log(F(s)) ~ H * log(s)`.

## Uso

```python
from quantmaster.features.statistical import hurst_dfa

df["hurst"] = hurst_dfa(df, window=240)
```

## API

::: quantmaster.features.statistical.hurst_dfa
