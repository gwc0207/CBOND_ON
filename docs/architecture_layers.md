# CBOND_ON Layered Architecture (v2)

## 1. Layer Overview

- `interfaces`: CLI entrypoints only (argument parsing and invoking pipelines/usecases)
- `app`: orchestration layer
  - `commands`: input command objects
  - `dto`: cross-layer return objects
  - `ports`: abstract contracts for infra
  - `usecases`: single business action
  - `pipelines`: multi-stage orchestration
- `domain`: pure business rules (no IO/FFI)
  - includes `signals` and `portfolio` split
- `infra`: concrete implementations (rust/model/io/report/cache)
- `config`: config loader/schema facade
- `common`: shared constants/errors

## 2. Dependency Rules

1. `interfaces -> app -> domain`
2. `app -> ports -> infra`
3. `domain` must not import `infra`
4. `interfaces` must not hold business logic

## 3. Current Landing Scope

Implemented in this change set:

1. New package skeleton:
   - `cbond_on/interfaces`
   - `cbond_on/app`
   - `cbond_on/domain`
   - `cbond_on/infra`
   - `cbond_on/common`
2. Domain extraction:
   - `domain/signals/service.py`
   - `domain/portfolio/service.py`
3. Service integration:
   - `services/backtest/backtest_service.py` now uses domain `signals/portfolio`
   - `services/live/live_service.py` now uses domain `signals`
4. CLI migration:
   - all `run/*.py` entrypoints now delegate to `interfaces/cli/*`
5. App orchestration modules:
   - usecases and pipelines are available under `cbond_on/app`

## 4. Compatibility

- Existing commands remain unchanged, e.g.:
  - `python cbond_on/run/factor_batch.py`
  - `python cbond_on/run/model_score.py`
  - `python cbond_on/run/model_eval.py`
  - `python cbond_on/run/backtest.py`
  - `python cbond_on/run/live.py`

