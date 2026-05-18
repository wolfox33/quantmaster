# Feature Discovery Skill

## Objective

Propose new quantitative features with clear financial hypothesis and implementation-ready specification.

## Inputs

- Domain focus (momentum, volatility, microstructure, risk, etc.)
- Data constraints (OHLCV frequency, assets, market)
- Target horizon and use case

## Process

1. Search for feature gaps in existing modules under `src/quantmaster/features`.
2. Propose 3-5 candidate features with:
   - Name
   - Formula
   - Economic intuition
   - Required columns
   - Main parameters and defaults
3. Rank candidates by implementation effort and expected value.
4. Choose one candidate and write a spec in `feature_specs/<feature_name>.md`.

## Acceptance Criteria

- Spec created in `feature_specs/`
- Feature is not duplicate of current public API
- Inputs and edge-case behavior are explicit

