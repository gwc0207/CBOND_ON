# CBOND_ON Target Architecture

This document records the architecture target for the current phased refactor.
The first phase must preserve current factor, model, strategy, and backtest
results.

## Runnable Entrypoints

Only these five commands are considered stable user-facing entrypoints:

```text
live
factor_batch
model_score
model_eval
strategy_backtest
```

Legacy `cbond_on/run/*.py` files remain as compatibility wrappers.

## Layer Rules

```text
cli          parses arguments and starts one workflow
bootstrap    assembles configs and concrete dependencies
workflows    orchestrate flow steps
domain       owns business rules
ports        define external boundaries
infra        implements external details
schemas      validate configs, contracts, and artifacts
tests        prevent architecture drift
```

`workflows/common` may only hold reusable orchestration fragments. It must not
own business rules or IO implementations.

## Phase 1 Shape

```text
cbond_on/
  cli/
  workflows/
    production/
    research/
    backtest/
    common/
  bootstrap/
  ports/
  schemas/
  factor_contracts/
  model_registry/
```

The new workflow layer currently delegates into the existing
`app/pipelines -> app/usecases` implementation to avoid changing production
behavior. Later phases should move orchestration out of `app/pipelines` only
after tests cover the current outputs.

