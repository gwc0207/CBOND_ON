# CBOND_ON Architecture Layers (Current)

## Layer Structure

- `run`
  - compatibility wrappers only.
  - delegates to `cbond_on.cli`; no business logic or one-off scripts.
- `cli`
  - stable command entrypoints and argument parsing.
- `bootstrap`
  - loads and validates configs and assembles workflow inputs.
- `workflows`
  - production, research, and backtest workflow entrypoints.
- `interfaces`
  - legacy-compatible CLI adapters.
  - delegates to `cbond_on.cli` while old import paths are retained.
- `app`
  - `usecases`: single business actions.
  - `pipelines`: multi-step orchestration.
- `domain`
  - pure business rules and semantics (`signals`, `portfolio`, factor definitions/spec).
  - no infra / cli dependency.
- `infra`
  - concrete implementations (`factors-pipeline`, `model`, `data`, `io`, `live`, `backtest`, `benchmark`, `report`, `universe`, `ai`, `data_hub`).
- `common`
  - cross-cutting helpers.
- `harness`
  - agent operating layer only.
  - stores policies, workflows, skills, and templates that guide maintenance
    behavior.
  - not part of production runtime, research runtime, or live scheduling.

## Dependency Rules

1. `run/* -> cli` and wrappers must stay thin.
2. `interfaces/cli/* -> cli` for compatibility only.
3. `cli -> bootstrap / workflows`.
4. `workflows -> app/pipelines`, and `app -> domain / infra`.
5. `domain` must not import `app`, `infra`, `interfaces`, `cli`, or `workflows`.
6. no code can import `cbond_on.services.*` (legacy layer removed).
7. `cbond_on/*` runtime code must not import `harness/*`.
8. no code can import legacy packages:
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
- `run/*` entry scripts route through `cbond_on.cli/*`.
- `interfaces/cli/*` remains as a compatibility adapter over `cbond_on.cli/*`.
- `cli -> bootstrap -> workflows` is the active command-side architecture;
  workflows currently delegate into `app/pipelines -> app/usecases` to preserve
  production behavior during the migration.
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
python -m cbond_on.common.architecture_guard
```

Use this to enforce repository cleanliness (runtime/dependency artifacts not tracked):

```bash
python -m cbond_on.common.repo_hygiene_guard
```

Use this to run factor quality checks (read-only, no auto-clean):

```bash
python -m cbond_on.common.factor_quality_guard --config factor
```

Apply actions:

```bash
# disable bad factors (writes config/factor/guards/factor_disabled_factors.json)
python -m cbond_on.common.factor_quality_guard --config factor --apply-disable-bad

# remove deprecated factor columns from factor store
python -m cbond_on.common.factor_quality_guard --config factor --apply-remove-deprecated

# do both in one run
python -m cbond_on.common.factor_quality_guard --config factor --apply-disable-bad --apply-remove-deprecated
```
