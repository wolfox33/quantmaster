# Features

As features são organizadas por categoria (Momentum, Volatility, Trend, etc.).

## Importar features

Você pode importar features diretamente de seus módulos:

```python
from quantmaster.features.momentum import rsi
from quantmaster.features.volatility import yang_zhang_volatility
```

Ou importar várias features de uma vez pelo namespace `quantmaster.features` (exporta as features públicas em `__all__`):

```python
from quantmaster.features import rsi, yang_zhang_volatility, hurst_dfa
```

Em notebooks, `from quantmaster.features import *` pode ser conveniente, mas não é recomendado para código de produção.

Para gerar várias features de uma vez (com parâmetros default), você pode usar `create_all`:

```python
from quantmaster.features import create_all

df = create_all(df)
```

Cada página de feature contém:

- Explicação teórica
- Observações práticas (parâmetros, unidades, limitações)
- API + código (gerado automaticamente)
