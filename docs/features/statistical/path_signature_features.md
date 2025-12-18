# Path Signature Features

## Intuição

*Path signatures* são uma família de features que representam uma série temporal multivariada como integrais iteradas (uma expansão tipo “séries de Taylor” para caminhos), capturando dependências temporais e interações entre dimensões de forma compacta.

Na prática, são muito usadas em ML para séries temporais.

## Definição

Dado um caminho multivariado `X_t` (ex.: OHLCV), a assinatura truncada em profundidade `d` é um vetor de integrais iteradas até ordem `d`.

Nesta implementação:

- Usamos um caminho 5D: `open, high, low, close, volume`.
- Aplicamos `log` (opcional) para estabilizar escala.
- Em cada janela rolling de tamanho `window`, calculamos a assinatura com `iisignature.sig`.

## Dependência opcional

Este recurso requer a dependência opcional **`iisignature`**.

```bash
pip install iisignature
```

Se não estiver instalada, a função levanta `ImportError` com mensagem clara.

## Uso

```python
from quantmaster.features.statistical import path_signature_features

Xsig = path_signature_features(ohlcv_df, depth=2, window=20)
```

## API

::: quantmaster.features.statistical.path_signature_features
