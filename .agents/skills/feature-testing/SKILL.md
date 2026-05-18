# Feature Testing Skill

## Objective

Add robust tests for a new feature and enforce contract behavior.

## Inputs

- Implemented feature function
- Synthetic dataset fixture pattern from existing tests

## Process

1. Add tests in module-specific test file under `tests/`.
2. Cover:
   - shape and index alignment
   - output name
   - finite/range expectations when applicable
   - no-lookahead using `tests.helpers.assert_no_lookahead`
3. Add validation tests for bad parameters when relevant.

## Acceptance Criteria

- Tests pass locally with `uv run pytest`
- New feature has no-lookahead coverage
- Expected error paths are validated

