# HAR-RV Forecast (rolling OLS)

## Intuição

O HAR-RV clássico usa regressão (HAR) para prever volatilidade futura a partir de múltiplas escalas da volatilidade realizada.

Nesta feature, fazemos uma regressão **rolling** para produzir, em cada tempo `t`, uma previsão de `log(RV_{t+horizon})` usando como regressoras:

- `log(RV_t)`
- `mean(log(RV))` em janela semanal
- `mean(log(RV))` em janela mensal

Isso evita look-ahead: os coeficientes em `t` são ajustados apenas com dados até `t-horizon`.

## Definição

Para cada `t`, ajustamos uma OLS em janela `estimation_window` e calculamos:

- `y_{t+h} = beta0 + beta1*x_d(t) + beta2*x_w(t) + beta3*x_m(t)`

onde `x_d, x_w, x_m` são as features do HAR sobre `log(RV)`.

## Uso

```python
from quantmaster.features.volatility import har_rv_forecast

df["har_rv_forecast_1_100"] = har_rv_forecast(df, horizon=1, estimation_window=100)
```

## API

::: quantmaster.features.volatility.har_rv_forecast
