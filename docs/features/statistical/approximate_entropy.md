# APPROXIMATE ENTROPY

## Intuição

A **Approximate Entropy (ApEn)** é um predecessor do Sample Entropy e também mede irregularidade. Em geral, ApEn é mais sensível a séries curtas, mas tem algumas diferenças metodológicas (inclui auto-matches).

## Definição

Em uma janela com parâmetros `m` e `r`:

`ApEn = φ(m) - φ(m+1)`

onde `φ(m)` é a média de `ln(C_i^m(r))` e `C_i^m(r)` é a fração de vetores de embedding que ficam dentro da tolerância.

## Uso

```python
from quantmaster.features.statistical import approximate_entropy

df["approximate_entropy_100"] = approximate_entropy(df, window=100, m=2, r=0.2)
```

## API

::: quantmaster.features.statistical.approximate_entropy
