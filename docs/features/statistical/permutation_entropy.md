# PERMUTATION ENTROPY

## Intuição

A **Permutation Entropy (Bandt & Pompe, 2002)** mede complexidade com base em padrões ordinais (ranking) de valores consecutivos. É relativamente robusta a ruído e útil para detectar mudanças de regime.

## Definição

Para `order` e `delay`, obtém-se a distribuição empírica dos padrões ordinais em janelas e calcula-se:

`PE = - Σ p_i ln(p_i)`

## Uso

```python
from quantmaster.features.statistical import permutation_entropy

df["pe"] = permutation_entropy(df, window=100, order=3, delay=1)
```

## API

::: quantmaster.features.statistical.permutation_entropy
