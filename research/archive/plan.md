# Plan: Quantmaster Features

Este arquivo define o padrão para criar e manter features quantitativas no projeto.

## Objetivos da lib

- Receber `pandas.DataFrame` com OHLCV (ou `Series` quando fizer sentido).
- Retornar `Series`/`DataFrame` alinhado ao índice, pronto para ser atribuído no `df`.
- Minimizar dependências (base: `numpy`, `pandas`).
- Evitar look-ahead bias (garantir que a feature use apenas informação disponível até o timestamp).
- Facilitar extensão: features divididas por módulos/categorias.

## Categorias (módulos)

Arquivos sugeridos em `src/quantmaster/features/`:

- **Trend**: `trend.py`
- **Momentum**: `momentum.py`
- **Volatility**: `volatility.py`
- **Volume**: `volume.py`
- **Returns**: `returns.py`
- **Risk**: `risk.py`
- **Statistical**: `statistical.py`
- **Regime**: `regime.py`
- **Microstructure**: `microstructure.py`
- **Utils (interno)**: `utils.py`

Você pode criar novas categorias conforme necessário.

## Padrão de assinatura (recomendado)

- Primeiro argumento: `data` (`DataFrame` ou `Series`).
- Parâmetros restantes: keyword-only.
- Permitir customização de nomes de colunas (ex: `price_col="close"`).

Template:

```python
def feature_name(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 10,
    price_col: str = "close",
) -> pd.Series:
    ...
```

## Checklist para adicionar uma nova feature

- **Definição**
  - Nome curto e consistente (snake_case).
  - Categoria correta (Trend/Momentum/Volatility/...).
  - Decidir se retorna `Series` (1 coluna) ou `DataFrame` (múltiplas colunas).

- **Input/Validação**
  - Validar `window`/parâmetros (inteiros positivos, etc.).
  - Validar colunas necessárias (quando `data` for `DataFrame`).

- **Look-ahead bias**
  - Verificar se a feature usa valores futuros sem querer.

- **Output**
  - Retornar alinhado ao índice original.
  - Definir `Series.name` ou nomes das colunas (prefixo consistente).

- **Testes**
  - Criar teste em `tests/`:
    - formato (`Series` vs `DataFrame`)
    - alinhamento de índice
    - comportamento em dados curtos (NaNs iniciais)
    - checagem de não-lookahead (quando aplicável)

- **Export**
  - Exportar no `src/quantmaster/features/__init__.py` se for uma feature pública.

- **Docs**
  - Adicionar exemplo de uso em `README.md` e/ou `docs/index.md`.
