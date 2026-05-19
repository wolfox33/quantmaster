# Quantmaster

Biblioteca de features quantitativas para adicionar colunas em `pandas.DataFrame` com dados OHLCV.

## Instalação (desenvolvimento)

```bash
pip install -e ".[dev]"
```

## Instalação (uso)

```bash
pip install quantmaster
```

## Uso rápido

```python
from quantmaster.features.momentum import rsi
from quantmaster.features.volatility import har_rv
from quantmaster.features.utils import create_all

df["rsi_10"] = rsi(df, window=10)
df = df.join(har_rv(df))

# gerar várias features approved de uma vez (com defaults)
df = create_all(df)

# incluir também features experimentais
# df = create_all(df, include_statuses=["approved", "experimental"])
```

## Importar features de uma vez

Se você quiser importar várias features sem ficar apontando para cada submódulo, use o namespace `quantmaster.features` (exporta as features públicas em `__all__`):

```python
from quantmaster.features import rsi, har_rv, yang_zhang_volatility, hurst_dfa
```

Você também pode fazer import wildcard (não recomendado em código de produção, mas útil em notebooks):

```python
from quantmaster.features import *
```

## Estrutura

- Features ficam em `src/quantmaster/features/` separadas por categoria (Momentum, Trend, Volatility, etc.).
- Cada feature é uma função que recebe `DataFrame` (ou `Series`) e retorna `Series` (ou `DataFrame`) alinhado ao índice.

## Qualidade

```bash
python -m ruff check .
python -m pytest
python -m build
```

## Fluxo Para Agentes

- Skills locais: `.agents/skills/`
- Especificações de feature: `feature_specs/`
- Templates de scaffold: `templates/`
- Script de scaffold: `scripts/new_feature.ps1`
- Script de verificação completa: `scripts/agent_verify.ps1`

Exemplo de scaffold:

```powershell
scripts/new_feature.ps1 -Module momentum -Feature my_new_feature -DefaultWindow 20
```

Verificação ponta a ponta:

```powershell
scripts/agent_verify.ps1
```
