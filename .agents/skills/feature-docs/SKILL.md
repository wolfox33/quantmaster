# Feature Docs Skill

## Objective

Document new features and keep docs navigation aligned with public API.

## Inputs

- New feature name and module

## Process

1. Create doc page under `docs/features/<module>/<feature>.md`.
2. Use pattern:
   - short code usage example
   - `::: quantmaster.features.<module>.<feature>`
3. Add page to `mkdocs.yml` nav in matching section.
4. Run `uv run mkdocs build`.

## Acceptance Criteria

- Docs page exists and renders
- `mkdocs build` succeeds
- Nav contains the new page

