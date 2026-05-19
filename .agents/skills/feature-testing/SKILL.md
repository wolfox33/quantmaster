---
name: feature-testing
description: "Use when a feature needs contract tests, no-lookahead checks, and parameter validation."
capability: "Feature Testing"
status: "experimental"
---

# Feature Testing

## Purpose

Add focused tests for one feature and enforce Quantmaster behavioral contracts.
Contracts must explicitly include no-lookahead bias and no-repaint behavior.

## Use when

- A new feature was implemented and needs test coverage.
- Existing feature behavior changed and regression tests are needed.
- No-lookahead and parameter validation must be guaranteed.

## Do not use when

- The feature code is not implemented yet.
- The request is docs-only.
- Broad test-suite refactor is requested instead of feature-scoped tests.

## Inputs

- Implemented feature function.
- Related module test file in `tests/`.
- Shared helper `tests/helpers.py`.

## Procedure

1. Add or update tests in the module-specific test file.
2. Cover shape/index alignment, output name, and basic numeric expectations.
3. Add `assert_no_lookahead` coverage.
4. Add no-repaint validation:
   - recompute after appending future rows
   - assert historical values up to prior cutoff remain unchanged
5. Add parameter validation tests when the feature exposes constraints.
6. Run `uv run pytest` and verify the changed feature tests pass.

## Output

The skill should produce:

- Feature-scoped tests in `tests/`.
- No-lookahead coverage for the feature.
- No-repaint coverage for the feature.
- A brief note on any residual risk not covered by tests.

## Anti-bloat rules

- One run focuses on one feature's test surface.
- Do not rewrite unrelated tests.
- Do not add heavy benchmark logic unless explicitly requested.
- Keep fixtures minimal and deterministic.
- Reject test sets that cannot detect temporal leakage or repaint behavior.
