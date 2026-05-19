---
name: feature-discovery
description: "Use when a new quantitative feature should be proposed as an implementation-ready spec."
capability: "Feature Discovery"
status: "experimental"
---

# Feature Discovery

## Purpose

Propose one high-value quantitative feature with measurable alpha potential and materialize it as an implementation-ready spec.
The feature must be explicitly non-repainting and free from lookahead bias.

## Use when

- A contributor needs new feature ideas grounded in current API gaps.
- A repeated discovery step should generate a spec in `feature_specs/`.
- The team wants a ranked shortlist before implementation.

## Do not use when

- A feature spec already exists and implementation should start directly.
- The request is speculative and not approved for materialization.
- The candidate clearly duplicates an existing public feature.

## Inputs

- Domain focus (momentum, volatility, microstructure, risk, etc.).
- Data constraints (OHLCV frequency, instruments, market).
- Current modules under `src/quantmaster/features`.
- Existing specs under `feature_specs/`.
- Modeling context (target horizon, prediction target, rebalance frequency).

## Procedure

1. Scan public features for overlap or gaps.
2. Propose 3-5 candidates with:
   - feature name
   - formula
   - economic mechanism
   - required columns
   - default parameters
3. For each candidate, score five quant dimensions from `1` (weak) to `5` (strong):
   - expected predictive power (IC/rank-IC intuition)
   - implementability (data availability, latency, complexity)
   - execution realism (turnover/cost sensitivity)
   - robustness expectation (regime stability)
   - leakage risk (lookahead/proxy leakage risk)
4. Rank by total score, then break ties by lower leakage risk.
5. Select one candidate and write `feature_specs/<feature_name>.md` with:
   - explicit edge-case behavior
   - expected sign intuition
   - minimal validation plan (IC, stability by regime, turnover impact)
   - proof sketch that computation uses only information available at timestamp `t`
   - repaint policy: once value at `t` is emitted, it must not change when future data arrives

## Output

The skill should produce:

- One new or updated feature spec in `feature_specs/`.
- A short rationale for why this feature was selected over alternatives.
- A compact score table for all evaluated candidates.
- A clear statement that the chosen candidate is no-repaint and no-lookahead by construction.

## Anti-bloat rules

- One run should materialize one implementation-ready feature spec.
- Do not embed long literature review inside the skill.
- Do not add scripts or assets for discovery.
- Do not update registry files unless explicitly requested.
- Do not approve a feature that cannot define a leakage-safe computation path.
- Reject any candidate that depends on future bars or retrospective recalculation of past outputs.
