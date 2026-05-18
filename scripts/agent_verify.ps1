$ErrorActionPreference = "Stop"

Write-Host "[1/4] Ruff"
uv run ruff check .

Write-Host "[2/4] Pytest"
uv run pytest

Write-Host "[3/4] Build"
uv run python -m build

Write-Host "[4/4] Docs"
uv run mkdocs build

Write-Host "All agent checks passed."
