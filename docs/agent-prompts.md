# Agent Prompts

## 1) Descoberta de feature

```text
Analise as features já existentes em src/quantmaster/features e proponha 3 novas features não duplicadas para [domínio].
Para cada uma: hipótese, fórmula, entradas, parâmetros, riscos de implementação e prioridade.
Depois escolha 1 e gere a spec em feature_specs/<nome>.md.
```

## 2) Implementação completa

```text
Com base em feature_specs/<nome>.md, implemente a feature no módulo correto, exporte em __init__.py, crie testes e docs.
No final rode: uv run ruff check ., uv run pytest, .venv\Scripts\python.exe -m build, uv run mkdocs build.
```

## 3) Refatoração segura

```text
Refatore apenas o módulo [arquivo] mantendo API e comportamento.
Não mude assinaturas públicas.
Confirme com testes existentes e reporte qualquer risco residual.
```

## 4) Hardening de testes

```text
Fortaleça testes da feature [nome] com foco em no-lookahead, NaN handling, parâmetros inválidos e regressão numérica básica.
```

