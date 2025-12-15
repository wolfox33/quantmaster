# ORNSTEIN–UHLENBECK

## Intuição

O processo de **Ornstein–Uhlenbeck (OU)** é um modelo clássico de **reversão à média**. Para séries financeiras, o preço bruto raramente é OU; por isso, uma abordagem prática é aplicar OU em uma série **detrended**.

Nesta feature, usamos:

- `x = log(close) - MA(log(close), m)`

E estimamos parâmetros OU em janela móvel, retornando múltiplas colunas úteis para ML.

## Definição

Usamos a forma discreta (AR(1)):

`x_t = c + phi * x_{t-1} + eps_t`

onde `phi = exp(-kappa)` e:

- `kappa = -log(phi)` (para `0 < phi < 1`)
- `theta = c / (1 - phi)`
- `sigma_eq` (desvio padrão de equilíbrio) é estimado via variância dos resíduos
- `half_life = log(2) / kappa`
- `zscore = (x_t - theta) / sigma_eq`

## Uso

```python
from quantmaster.features.statistical import ornstein_uhlenbeck

feat = ornstein_uhlenbeck(df, window=240, detrend_window=48)
df = df.join(feat)
```

## API

::: quantmaster.features.statistical.ornstein_uhlenbeck
