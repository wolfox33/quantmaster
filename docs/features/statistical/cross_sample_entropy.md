# CROSS SAMPLE ENTROPY

## Intuição

A **Cross Sample Entropy** mede a similaridade de padrões entre duas séries temporais. É útil para investigar relações não-lineares, sincronização e possíveis efeitos de lead-lag (por exemplo em pairs).

## Definição

É análoga ao SampEn, mas compara subsequências extraídas de `data1` contra subsequências de `data2`.

## Uso

```python
from quantmaster.features.statistical import cross_sample_entropy

out = cross_sample_entropy(s1, s2, window=100, m=2, r=0.2)
```

## API

::: quantmaster.features.statistical.cross_sample_entropy
