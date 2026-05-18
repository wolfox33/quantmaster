# Agent Workflow

Este projeto suporta fluxo de contribuição orientado a agentes (Codex e similares).

## Fluxo padrão

1. Descobrir uma feature e registrar spec em `feature_specs/<feature>.md`.
2. Implementar feature no módulo correto em `src/quantmaster/features/`.
3. Exportar API pública em `src/quantmaster/features/__init__.py`.
4. Criar/atualizar testes em `tests/`.
5. Criar docs da feature e atualizar `mkdocs.yml`.
6. Rodar validação completa:
   - `uv run ruff check .`
   - `uv run pytest`
   - `.venv\Scripts\python.exe -m build`
   - `uv run mkdocs build`

## Automação

- Scaffold de arquivos de feature:
  - `scripts/new_feature.ps1 -Module <module> -Feature <feature_name> -DefaultWindow 20`
- Validação ponta a ponta:
  - `scripts/agent_verify.ps1`

## Definition of Done

- Spec criada/atualizada em `feature_specs/`
- Feature integrada e exportada
- Testes cobrindo contrato e no-lookahead
- Docs renderizando sem erro
- Lint, testes, build e docs build passando

