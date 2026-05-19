# Walkthrough: Adição de Features Causais

Implementei as features solicitadas baseadas no documento `causal_features_trading.md`, garantindo cobertura de testes e integração com o pacote `quantmaster`.

## Features Implementadas

### 1. Entropia (`quantmaster.features.entropy`)
Módulo novo criado para métricas de complexidade e informação.
- **`shannon_entropy`**: Mede a incerteza/informação na distribuição de preços/retornos.
- **`permutation_entropy`**: Mede a complexidade da série temporal baseada em padrões ordinais (movida de `statistical` para cá).

### 2. Microestrutura (`quantmaster.features.microstructure`)
Expandido com features de fluxo de ordens e spreads.
- **`vwap_deviation`**: Desvio percentual do preço em relação ao VWAP.
- **`order_flow_imbalance_range`**: Proxy de OFI normalizado pelo range da barra.
- **`vpin_proxy`**: Estimativa de Volume-Synchronized Probability of Informed Trading usando classificação de volume por direção.

### 3. Estatística (`quantmaster.features.statistical`)
Adicionadas métricas de dependência serial e volatilidade.
- **`volatility_clustering`**: Autocorrelação dos retornos ao quadrado (efeito ARCH).
- **`ljung_box_stat`**: Estatística Q para testar ausência de autocorrelação serial.

> **Nota:** A feature `variance_ratio` já existia em `quantmaster.features.regime`, então não foi duplicada.

## Verificação

### Testes Automatizados
Todos os testes foram executados com sucesso no ambiente `ml`.
- `tests/test_entropy.py`: Validado cálculo de entropia.
- `tests/test_microstructure.py`: Validado VPIN (incluindo correção lógica), VWAP Deviation e OFI.
- `tests/test_statistical.py`: Validadas novas features estatísticas.

### Correções Realizadas
- **Setup de Ambiente**: Instalação do pacote em modo editável (`pip install -e .`) e instalação do `pytest` no ambiente `ml` para corrigir erros de importação.
- **Bug Fix**: Correção na lógica de cálculo do numerador do `vpin_proxy` (ordem de `abs()` e `sum()`).
- **Refatoração**: Criação de `tests/__init__.py` para permitir importação de helpers.

## Como Usar
As novas features estão acessíveis diretamente via `quantmaster.features`:

```python
from quantmaster.features import (
    shannon_entropy,
    vpin_proxy,
    volatility_clustering
)
# df é seu DataFrame OHLCV
ent = shannon_entropy(df)
vpin = vpin_proxy(df)
arch = volatility_clustering(df)
```
