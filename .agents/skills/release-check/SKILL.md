# Release Check Skill

## Objective

Run the full quality gate before opening or merging PRs.

## Commands

```powershell
uv run ruff check .
uv run pytest
.venv\Scripts\python.exe -m build
uv run mkdocs build
```

## Acceptance Criteria

- Lint passes
- Tests pass
- Build artifacts generated
- Docs build passes

