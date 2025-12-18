# Chande Momentum Oscillator

## Intuição

O **Chande Momentum Oscillator (CMO)** é um oscilador de momentum semelhante ao RSI, mas baseado em somas (não médias) de ganhos e perdas na janela.

Ele retorna valores em `[-100, 100]`.

## Definição

Para `ΔP_t = P_t - P_{t-1}` e janela `n`:

- `Gain_t = sum(max(ΔP, 0))`
- `Loss_t = sum(max(-ΔP, 0))`
- `CMO_t = 100 * (Gain_t - Loss_t) / (Gain_t + Loss_t)`

## Uso

```python
from quantmaster.features.momentum import chande_momentum_oscillator

df["cmo_14"] = chande_momentum_oscillator(df, window=14)
```

## API

::: quantmaster.features.momentum.chande_momentum_oscillator
