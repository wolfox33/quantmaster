---
name: feature-docs
description: "Use when a feature needs documentation and mkdocs navigation integration."
capability: "Feature Documentation"
status: "experimental"
---

# Feature Documentation

## Purpose

Document one feature and keep docs navigation aligned with public API.

## Use when

- A new feature was added and needs a docs page.
- Existing feature docs are missing or outdated.
- Navigation in `mkdocs.yml` must include the feature page.

## Do not use when

- The request is implementation-only with no docs scope.
- The feature is not yet implemented/exported.
- The change is unrelated to feature documentation.

## Inputs

- Feature name and module name.
- Existing docs structure under `docs/features/`.
- `mkdocs.yml`.

## Procedure

1. Create or update `docs/features/<module>/<feature>.md`.
2. Include a minimal usage snippet and mkdocstrings reference.
3. Add/update entry in `mkdocs.yml` nav under the correct section.
4. Run `uv run mkdocs build` and ensure docs render without errors.

## Output

The skill should produce:

- Feature docs page under `docs/features/`.
- Updated `mkdocs.yml` nav entry.
- A short note confirming docs build status.

## Anti-bloat rules

- One run documents one feature surface.
- Keep docs concise and API-focused.
- Do not add broad editorial rewrites unless requested.
- Do not add assets/scripts unless clearly required.
