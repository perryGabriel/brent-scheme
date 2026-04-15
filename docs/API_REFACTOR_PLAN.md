# API Refactor Plan (Future Work)

This document proposes a gradual migration from utility-style classes toward a more idiomatic Python API while preserving backwards compatibility.

## Current style

The project exposes mostly stateless classes:

- `SchemaFactory`
- `SchemeManipulator`
- `Stepper`
- `Trainer`
- `SchemeDisplay`

This works, but users often expect either:

1. Functional utilities (`set_naive(scheme)`, `train_step(scheme, ...)`), or
2. Stateful trainer/manipulator objects that own optimizer config and logs.

## Goals

- Keep existing behavior and import paths stable.
- Reduce accidental mutable coupling.
- Improve discoverability and tab-completion.
- Make training workflows easier to compose and test.

## Proposed phases

### Phase 1 (non-breaking)

- Keep classes intact.
- Add explicit type hints and docstrings for public methods.
- Add high-level helper functions in a new `brentscheme.api` module:
  - `make_scheme(...)`
  - `train_scheme(...)`
  - `evaluate_scheme(...)`
  - `compose_schemes(...)`

### Phase 2 (dual API)

- Introduce dataclasses for training configuration:
  - `TrainingConfig`, `BasisOptimizationConfig`.
- Provide a stateful trainer:
  - `SchemeTrainer(config)` with `.fit(scheme)` and history tracking.
- Keep old `Trainer` methods as wrappers (emit deprecation warnings).

### Phase 3 (deprecation window)

- Mark legacy utility methods in release notes.
- Encourage functional or stateful API as primary docs path.
- Maintain wrappers for at least one minor release cycle.

## Compatibility policy

- Existing notebooks/scripts should continue to run unmodified during Phase 1 and Phase 2.
- Any deprecation should include clear replacement examples.
