# Factor Mining Aggregate v4 Strict-PIT Smoke

## Status

Research-only mechanics smoke passed. This record does not claim IC eligibility,
correlation eligibility, model admission, or live promotion.

## Fixed Contract

- requested IC start: `2025-01-01` (first score day `2025-01-02`)
- factor panel/time: `T1430` / `14:30`
- label: same-score-day `14:42`
- universe for the later screen: strict T-1
  `quant_factor_dev.researcher_xuvb.o_0005`
- later quality gates: absolute mean daily Pearson IC `> 0.02`, at least 250
  valid days and 50 in every 60/20/20 partition, 200 common redundancy days,
  within-family absolute correlation `< 0.80`, cross-family `< 0.70`

The smoke range is `2026-04-28..2026-04-29`; it does not replace any of the
full-window gates.

## Catalogue

`cbond_on.domain.factors.defs.research_factor_mining_aggregate_catalog_v4`
contains 193 signals in 67 families. It prefixes the immutable healthy v3 core
and adds only independently smoked or fail-closed healthy-wrapper sources. It
intentionally excludes the coverage-insufficient rank-lattice and event-clock
wrappers, the debt-floor source, known persistent-empty modules, and all
un-smoked newly designed modules.

## Validation

```powershell
py -3.11 -m pytest -q \
  tests/test_research_factor_mining_aggregate_catalog_v3.py \
  tests/test_research_factor_mining_aggregate_catalog_v4.py \
  tests/test_research_factor_mining_intraday_state_gated_microstructure_healthy_catalog_v1.py \
  tests/test_research_factor_mining_daily_asymmetric_state_transitions_healthy_catalog_v1.py \
  tests/test_research_factor_mining_daily_asymmetric_state_transitions_crossperiod_healthy_catalog_v1.py \
  tests/test_research_factor_mining_cross_sectional_microstructure_neighborhood_healthy_catalog_v1.py \
  tests/test_research_factor_mining_intraday_queue_state_healthy_catalog_v1.py

py -3.11 harness/tools/run_factor_mining_expansion.py \
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_aggregate_catalog_v4 \
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_aggregate_catalog_v4_smoke_20260428_20260429 \
  --start 2026-04-28 --end 2026-04-29 --execute
```

- focused tests: `14 passed`
- Ruff check and formatting: passed
- no-write full-window preflight: passed; full root remained absent
- smoke root:
  `D:/cbond_on/research_scratch/factor_mining_20260803_aggregate_catalog_v4_smoke_20260428_20260429`
- one research-only manifest and one family catalogue, both matching 193
  signals / 67 families
- date files: two (330 and 324 rows), exact stable 193-column schema, unique
  `(dt, code)` index, zero Inf, zero all-empty signals, and zero
  cross-sectionally constant signals on either day

## Boundary And Next Step

Only the named scratch root was written. No production FactorStore, database,
live/model config, scheduler, mask, or trading rule changed.

Additional newly designed factor families remain outside v4 until their own
numeric-quality and strict-PIT smoke evidence is complete. After that decision,
freeze one successor catalogue, repeat preflight/smoke, then launch one new
full-window scratch build. Merge only an immutable full root into v7 and rerun
the global fixed-universe IC/redundancy screen.
