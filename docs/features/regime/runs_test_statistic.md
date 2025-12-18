# Runs Test Statistic

## Intuição

O **Runs Test** (Wald–Wolfowitz) testa a aleatoriedade da sequência de sinais (positivo/negativo) dos retornos contando o número de *runs* (blocos consecutivos de mesmo sinal).

- Poucos runs: possível *clustering* (persistência)
- Muitos runs: possível alternância/reversão

## Definição

Para uma janela `n`, definimos o sinal de `r_t` (ignorando zeros) e calculamos:

- `R`: número de runs na janela
- `Z = (R - μ_R) / σ_R`

com `μ_R` e `σ_R` dados pelas fórmulas padrão do teste (usando apenas dados dentro da janela).

## Uso

```python
from quantmaster.features.regime import runs_test_statistic

df["runs_z_60"] = runs_test_statistic(df, window=60)
```

## API

::: quantmaster.features.regime.runs_test_statistic
