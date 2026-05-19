---
name: release-check
description: "Use when full repository quality gates must run before merge or release."
capability: "Release Validation"
status: "experimental"
---

# Release Validation

## Purpose

Run the full quality gate and report pass/fail before merge or release.
The gate must enforce temporal integrity: no-lookahead bias and no-repaint coverage for feature changes.

## Use when

- A PR is ready for final validation.
- A release candidate needs full repository checks.
- CI-equivalent checks are required locally.

## Do not use when

- Only a single unit test is requested.
- Discovery/implementation/docs work is still in progress.
- The user asks for partial validation only.

## Inputs

- Repository root with dependencies installed.
- Validation script `scripts/agent_verify.ps1`.
- Changed files in the PR/branch.

## Procedure

1. Run `scripts/agent_verify.ps1`.
2. Inspect changed files and detect whether feature logic under `src/quantmaster/features/` changed.
3. If feature logic changed, require explicit temporal-test evidence in corresponding tests:
   - no-lookahead check
   - no-repaint check
4. Capture and report status of lint, tests, package build, docs build, and temporal-test evidence.
5. If any step fails, report the failing step and stop.

## Output

The skill should produce:

- One consolidated pass/fail result for all quality gates.
- The first failing gate and concise failure context when not green.
- A temporal-integrity verdict (`pass` or `fail`) when features were modified.

## Anti-bloat rules

- Do not expand scope beyond quality-gate execution.
- Do not edit source code unless explicitly requested.
- Keep reporting concise and actionable.
- Reject release readiness when feature changes lack no-lookahead or no-repaint test evidence.
