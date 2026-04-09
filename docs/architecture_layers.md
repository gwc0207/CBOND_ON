# CBOND_ON Architecture Layers (Current)

## Layer Structure

- `interfaces`
  - CLI/entry adapters only.
  - Converts args to pipeline calls.
- `app`
  - `usecases`: single business actions.
  - `pipelines`: multi-step orchestration.
  - `commands` / `dto` / `ports`: cross-layer contracts.
- `domain`
  - pure business rules and semantics (`signals`, `portfolio`, factor definitions/spec).
  - no infra / cli dependency.
- `infra`
  - concrete implementations (`rust`, `factors-pipeline`, `model`, `data`, `io`, `live`, `backtest`, `report`, `cache`).
- `common`
  - cross-cutting helpers.

## Dependency Rules

1. `interfaces -> app`.
2. `app -> domain` and `app -> infra`.
3. `domain` must not import `app` / `infra` / `interfaces`.
4. `run/*` must delegate to `interfaces/cli/*`.
5. no code can import `cbond_on.services.*` (legacy layer removed).
6. no code can import legacy packages:
   - `cbond_on.data`
   - `cbond_on.factors`
   - `cbond_on.models`
   - `cbond_on.backtest`
   - `cbond_on.report`
   - `cbond_on.model_eval`
   - `cbond_on.strategies`
   - `cbond_on.factor_batch`
   - `cbond_on.live`

## Migration Status

Completed:

- legacy package directories removed:
  - `cbond_on/services`
  - `cbond_on/data`
  - `cbond_on/factors`
  - `cbond_on/models`
  - `cbond_on/backtest`
  - `cbond_on/report`
  - `cbond_on/model_eval`
  - `cbond_on/strategies`
  - `cbond_on/factor_batch`
  - `cbond_on/live`
- `run/*` entry scripts route through `interfaces/cli/*`.
- `liveLaunch/*` routes through `app.pipelines.live_pipeline`.
- factor definitions/spec moved to `domain/factors/*`.
- factor execution pipeline moved to `infra/factors/*`.
- data adapters moved to `infra/data/*`.
- model core + eval moved to `infra/model/*` and `infra/model/eval/*`.
- backtest adapters moved to `infra/backtest/*`.
- reporting moved to `infra/report/*`.

## Guard Command

Use this to enforce boundaries after changes:

```bash
python cbond_on/run/architecture_guard.py
```

Use this to enforce repository cleanliness (runtime/dependency artifacts not tracked):

```bash
python cbond_on/run/repo_hygiene_guard.py
```
