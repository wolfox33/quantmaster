---
name: feature-implementation
description: "Use when an approved feature spec should be implemented and integrated into the public API."
capability: "Feature Implementation"
status: "experimental"
---

# Feature Implementation

## Purpose

Implement one feature from spec and integrate it into Quantmaster public API.
Implementation must be causal: no repaint and no lookahead bias.

## Use when

- A feature spec is approved and ready to code.
- Implementation must follow repository contracts and conventions.
- Public exports need synchronized update.

## Do not use when

- Discovery/specification is still incomplete.
- The request is broad refactor unrelated to a specific feature.
- Tests/docs are the only pending task.

## Inputs

- `feature_specs/<feature_name>.md`.
- Target module under `src/quantmaster/features/`.
- Current public exports in `src/quantmaster/features/__init__.py`.

## Procedure

1. Implement function with repository style (`DataFrame | Series` input, index-aligned output).
2. Apply shared validations and numeric coercion via `utils` helpers when appropriate.
3. Set deterministic output name (`out.name = ...`).
4. Enforce temporal causality:
   - never use future rows (`shift(-k)`, forward windows, centered rolling)
   - avoid any post-hoc smoothing that rewrites past timestamps
5. Export the feature in `src/quantmaster/features/__init__.py`.

## Output

The skill should produce:

- Implemented feature function in the correct module.
- Updated public export in `src/quantmaster/features/__init__.py`.
- A short note describing any non-obvious implementation tradeoff.
- A short causal note stating why the implementation is no-repaint and no-lookahead.

## Anti-bloat rules

- One run implements one feature only.
- Avoid unrelated refactors.
- Do not add extra abstractions without clear local reuse.
- Do not touch registry files unless explicitly requested.
- Reject implementation approaches that require future data to compute output at `t`.
