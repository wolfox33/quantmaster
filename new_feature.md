# Instruções: adicionar uma nova feature (via LLM)

Este arquivo é um *playbook* para pedir a uma LLM que implemente uma nova feature no **Quantmaster** mantendo padrões de código, testes e documentação.

> O `plan.md` cobre a visão geral e checklist, mas este arquivo é mais prescritivo (passo a passo + templates + quais arquivos alterar).

---

## Objetivo

Adicionar uma feature que:

- Receba `pandas.DataFrame` (OHLCV) ou `pandas.Series` (quando fizer sentido)
- Retorne `pandas.Series` (1 coluna) ou `pandas.DataFrame` (múltiplas colunas)
- Seja fácil de usar:
  - `df["minha_feature"] = minha_feature(df, ...)`
  - `df = df.join(minha_feature(df, ...))` (quando retornar `DataFrame`)
- Siga PEP-8, type hints, e seja extensível

---

## Onde colocar a feature (categorias)

As features ficam em:

- `src/quantmaster/features/`

Categorias atuais (pode criar novas se necessário):

- **Trend**: `trend.py`
- **Momentum**: `momentum.py`
- **Volatility**: `volatility.py`
- **Volume**: `volume.py`
- **Returns**: `returns.py`
- **Risk**: `risk.py`
- **Statistical**: `statistical.py`
- **Regime**: `regime.py`
- **Microstructure**: `microstructure.py`

Regra prática:

- Se a categoria já existe, adicione a função no arquivo correspondente.
- Se não existir, crie o arquivo `src/quantmaster/features/<categoria>.py`.

---

## Padrão de assinatura e comportamento

- **Assinatura**: primeiro argumento `data`, demais parâmetros keyword-only.
- **Sem efeitos colaterais**: não modificar `data` in-place.
- **Alinhamento**: retorno deve manter o mesmo índice do input.
- **Nome**:
  - Se retornar `Series`, definir `Series.name`.
  - Se retornar `DataFrame`, usar nomes de colunas estáveis e documentados.

### Template de função (Series)

```python
from __future__ import annotations

import pandas as pd

from quantmaster.features.utils import get_price_series, validate_positive_int


def feature_name(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 10,
    price_col: str = "close",
) -> pd.Series:
    window = validate_positive_int(window, name="window")

    price = get_price_series(data, price_col=price_col).astype(float)

    out = ...  # cálculo
    out.name = f"feature_name_{window}"
    return out
```

### Quando precisar de OHLCV (ex: high/low)

Use validações utilitárias (ou adicione novas em `utils.py`):

- `validate_columns(df, required=[...])`
- `get_price_series(...)`

---

## Testes (obrigatório)

Crie/atualize testes em `tests/`.

- Se já existir `tests/test_<categoria>.py`, adicione lá.
- Se não existir, crie `tests/test_<categoria>.py`.

### Template de teste mínimo

```python
import pandas as pd

from quantmaster.features.<categoria> import feature_name


def test_feature_name_shape_and_index() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "open": range(1, 51),
            "high": range(2, 52),
            "low": range(0, 50),
            "close": range(1, 51),
            "volume": [100] * 50,
        },
        index=idx,
    )

    out = feature_name(df, window=10)

    assert out.index.equals(df.index)
    assert out.name is not None
```

---

## Export público (obrigatório)

Se a feature for pública, exporte em:

- `src/quantmaster/features/__init__.py`

Adicione:

- import da função
- nome no `__all__`

---

## Documentação (obrigatório)

A documentação do usuário é o site do **MkDocs**.

### Criar a página da feature

Crie um arquivo em:

- `docs/features/<categoria>/<feature_name>.md`

Exemplos existentes:

- `docs/features/momentum/rsi.md`
- `docs/features/volatility/har_rv.md`

### Template da página

```md
# NOME DA FEATURE

## Intuição

Explique a motivação e o que a feature mede.

## Definição

Explique como é calculada (pode incluir fórmulas se necessário).

## Uso

```python
from quantmaster.features.<categoria> import feature_name

df["..."] = feature_name(df, ...)
```

## API

::: quantmaster.features.<categoria>.feature_name
```

### Atualizar o menu (mkdocs.yml)

Adicione a nova página em `mkdocs.yml` na seção `nav` dentro da categoria correta.

---

## Checklist final (para a LLM)

- Implementou a função no módulo correto (`src/quantmaster/features/...`)
- Adicionou/atualizou testes em `tests/`
- Exportou no `src/quantmaster/features/__init__.py`
- Criou página de docs com explicação teórica + API
- Atualizou `mkdocs.yml` (nav)
- Rodou (ou ao menos garantiu) que `pytest` passa

---

## Prompt sugerido (copie e cole para a LLM)

"""
Você está adicionando uma nova feature na biblioteca Python Quantmaster.

Regras:
- Siga o arquivo new_feature.md e o padrão do projeto.
- Implemente a feature como função pura em src/quantmaster/features/<categoria>.py
- Adicione testes em tests/
- Exporte em src/quantmaster/features/__init__.py
- Crie a página docs/features/<categoria>/<feature>.md com explicação teórica + uso + seção mkdocstrings
- Atualize mkdocs.yml para incluir a nova página no nav

Feature a implementar:
- Nome:
- Categoria:
- Descrição/teoria:
- Assinatura desejada:
- Colunas necessárias (OHLCV):

Entregáveis: PR pronto com código, testes e docs.
"""
