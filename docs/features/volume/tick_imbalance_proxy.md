# Tick Imbalance Proxy

## Intuição

O **Tick Imbalance Proxy** aproxima desequilíbrio de agressão compradora/vendedora usando apenas o sinal do retorno: média do sinal dos retornos em uma janela.

## Definição

Em uma janela `n`:

- `s_t = sign(r_t)`
- `TIP = mean(s_t)`

O output fica em `[-1, 1]`.

## Uso

```python
from quantmaster.features.volume import tick_imbalance_proxy

df["tip_20"] = tick_imbalance_proxy(df, window=20)
```

## API

::: quantmaster.features.volume.tick_imbalance_proxy
