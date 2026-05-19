# `momentum_volatility_state`

```python
from quantmaster.features.momentum import momentum_volatility_state

df["momentum_volatility_state_20_20_60"] = momentum_volatility_state(
    df,
    mom_window=20,
    vol_window=20,
    state_window=60,
)
```

::: quantmaster.features.momentum.momentum_volatility_state

