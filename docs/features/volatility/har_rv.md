# HAR-RV (Heterogeneous AutoRegressive - Realized Variance)

## Intuição

O modelo HAR-RV é uma forma simples e muito usada para descrever/predizer a dinâmica da volatilidade realizada usando múltiplas escalas de tempo (curta, média e longa). A ideia é capturar a **persistência** da volatilidade com médias móveis de diferentes janelas.

Nesta implementação, construímos as seguintes séries derivadas de uma proxy de variância realizada (RV) baseada em retornos:

- `har_rv_d`: RV diária (instantânea)
- `har_rv_w`: média móvel da RV em janela semanal (padrão 5)
- `har_rv_m`: média móvel da RV em janela mensal (padrão 22)

Essas colunas são frequentemente usadas como regressoras em modelos do tipo HAR para prever volatilidade futura.

## Uso

```python
from quantmaster.features.volatility import har_rv

df = df.join(har_rv(df, weekly_window=5, monthly_window=22))
```

## API

::: quantmaster.features.volatility.har_rv

::: quantmaster.features.volatility.realized_variance
