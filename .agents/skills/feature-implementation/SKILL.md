# Feature Implementation Skill

## Objective

Implement a feature from spec and integrate it into Quantmaster API.

## Inputs

- `feature_specs/<feature_name>.md`
- Target module in `src/quantmaster/features/`

## Process

1. Create function using repository style:
   - Input: `pd.DataFrame | pd.Series`
   - Output: `pd.Series` (or `pd.DataFrame` when needed)
   - Numeric coercion and validation through shared utils
2. Name output series deterministically (`out.name = ...`).
3. Export function in `src/quantmaster/features/__init__.py`.
4. Keep implementation pure and index-aligned.

## Acceptance Criteria

- Feature implemented in correct module
- Public API export updated
- No lookahead introduced
- Works with expected NaN/zero behavior from spec

