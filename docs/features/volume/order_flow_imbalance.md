# Order Flow Imbalance

## Intuição

O **Order Flow Imbalance** aproxima desequilíbrio de fluxo usando o volume e o sinal do retorno, produzindo um indicador normalizado em `[-1, 1]`.

## Definição

Em uma janela `n`:

- `s_t = sign(r_t)`
- `OFI = sum(volume * s) / sum(volume)`

## Uso

```python
from quantmaster.features.volume import order_flow_imbalance

df["ofi_20"] = order_flow_imbalance(df, window=20)
```

## API

::: quantmaster.features.volume.order_flow_imbalance
