# SAMPLE ENTROPY

## Intuição

A **Sample Entropy (SampEn)** mede a irregularidade/complexidade de uma série temporal. Em termos práticos, valores maiores indicam uma série menos previsível (mais “ruidosa”).

## Definição

Para uma janela de dados e parâmetros `m` e `r`:

`SampEn = -ln(A/B)`

onde:

- `B` é o número de pares de subsequências de comprimento `m` cuja distância (máximo dos desvios absolutos) é `<= r * std(window)`
- `A` é o número de pares equivalentes para comprimento `m+1`

## Uso

```python
from quantmaster.features.statistical import sample_entropy

df["sample_entropy_100"] = sample_entropy(df, window=100, m=2, r=0.2)
```

## API

::: quantmaster.features.statistical.sample_entropy
