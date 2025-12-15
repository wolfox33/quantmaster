# FracDiff (Fractional Differentiation)

## Intuição

Séries de preços costumam ser **não-estacionárias** (tendem a ter raiz unitária), o que dificulta o uso direto em muitos modelos estatísticos e de machine learning.

A diferenciação inteira (`d = 1`) ajuda a tornar a série mais estacionária, mas pode remover informação de longo prazo. A **diferenciação fracionária** busca um meio-termo: reduzir a não-estacionariedade preservando mais *memory* do sinal.

## Definição

A diferenciação fracionária de ordem `d` pode ser escrita como a expansão de:

- `(1 - L)^d x_t = sum_{k=0..∞} w_k x_{t-k}`

onde `L` é o operador de defasagem e os pesos são dados recursivamente por:

- `w_0 = 1`
- `w_k = -w_{k-1} * (d - k + 1) / k`

Na prática, truncamos a soma quando `|w_k| < thresh` (ou usando `max_lags`).

## Uso

```python
from quantmaster.features.statistical import fracdiff

df["fracdiff_0.5"] = fracdiff(df, d=0.5, thresh=1e-5)
```

## API

::: quantmaster.features.statistical.fracdiff
