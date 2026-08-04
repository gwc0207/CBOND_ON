# Factor mining: v8 serial queue toward 50 candidates (2026-08-03)

## Objective

Advance the research-only factor pool toward **50 exact-MIS retained factors**
without changing the fixed factor/label/mask contract or touching live assets.
This record is a queue-and-validation record, not a claim that the 50-factor
target has been reached.

## Frozen research contract

- Score range: requested `2025-01-01..2026-07-30` (381 score days).
- Factor/label: T1430 / 14:30 factor values and same-score-day 14:42 labels.
- Universe: existing T-1 `quant_factor_dev.researcher_xuvb.o_0005`, fail closed.
- Quality: `abs(mean daily Pearson IC) > 0.02`, 250 total valid days, at least
  50 valid days in each chronological 60/20/20 partition.
- Redundancy: `max(mean abs daily Pearson, mean abs daily Spearman) < 0.80`
  within family and `< 0.70` across family, with at least 200 common days.
- Outputs: `D:/cbond_on/research_scratch` only; no backtest/report stage,
  production FactorStore, live config, model, DB, scheduler, or strategy write.

## Inputs and status at launch

- Existing complete baseline: unified v7, 535 signals / 77 families / 381
  days.  Its exact-MIS maximum is 33, so a new, globally screened expansion is
  required to approach 50.
- Active predecessor: `factor_mining_20260803_aggregate_catalog_v5_full`,
  214 signals / 75 families.  It was still writing at queue launch and must
  finish naturally before any next full build starts.
- Next catalogue: `research_factor_mining_daily_orthogonal_batch_v8`, 24
  signals / 13 families.  A no-write composition against v7 and aggregate-v5
  produced 773 signals / 165 families with no signal/family collision.

## Validation performed

```powershell
py -3.11 -B -m pytest -q `
  tests/test_research_factor_mining_daily_breadth_regime_relation_v1.py `
  tests/test_research_factor_mining_daily_observable_seasoning_v1.py `
  tests/test_research_factor_mining_daily_orthogonal_batch_v5.py `
  tests/test_research_factor_mining_daily_orthogonal_batch_v6.py `
  tests/test_research_factor_mining_daily_orthogonal_batch_v7.py `
  tests/test_research_factor_mining_daily_orthogonal_batch_v8.py `
  tests/test_run_factor_mining_expansion.py -p no:cacheprovider
```

Result: `23 passed`.

The full-root preflight for
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_orthogonal_batch_v8_full_r1`
passed with Python/clean-direct/DataHub-only execution and all reports disabled.

The one-day combined smoke at
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_orthogonal_batch_v8_smoke_20260428_r1`
completed successfully:

- 330 unique `(dt, code)` rows at T1430;
- 24 signal columns / 13 families;
- 7,670 finite values, zero Inf;
- 290--330 non-null observations per signal;
- one research-only run manifest.

## Background serial queue

The queue implementation is
`harness/tools/queue_factor_mining_v8_to50.ps1`.  It was launched hidden as
PowerShell PID `31320` with state under:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_to50_serial_queue_r1
```

The queue records `status.json`, `queue_plan.json`, stage stdout/stderr logs,
and a retained lock.  It will, in order:

1. wait for aggregate-v5 to emit exactly one valid research-only manifest and
   exactly 381 FactorStore days;
2. fail closed on any frozen research-source/config/tool hash drift or fewer
   than two consecutive 6 GB free-memory observations;
3. run the fresh v8 full root;
4. compose v7 + aggregate-v5 + v8, outer-union merge the three immutable
   stores, run the fixed global screen, then exact MIS;
5. write whether `maximum_selected_count >= 50` to `status.json`.

No full builds overlap.  A predecessor failure, incomplete root, source drift,
or memory gate failure stops the queue; it never restarts or overwrites a root.

