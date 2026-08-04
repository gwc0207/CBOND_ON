# Factor Mining Aggregate v3 Strict-PIT Smoke

## Status

Research-only smoke passed.  This is not an IC screen, model admission, or
live promotion.

## Frozen Contract

- requested start: `2025-01-01` (first score day is `2025-01-02`)
- panel/factor time: `T1430` / `14:30`
- label: same-score-day `14:42`
- universe: strict previous-trading-day `quant_factor_dev.researcher_xuvb.o_0005`
- eventual screen: absolute mean daily Pearson IC `> 0.02`, 250 valid days,
  60/20/20 partition minimums, 200 common redundancy days, within-family
  correlation `< 0.80`, cross-family correlation `< 0.70`

The smoke date range was `2026-04-28..2026-04-29`.  It validates factor
mechanics only; it does not relax or substitute for the full-window gates.

## Catalogue

`cbond_on.domain.factors.defs.research_factor_mining_aggregate_catalog_v3`
contains 97 signals in 35 families.  It is a research-only successor to v2:

- excludes the persistent-empty limit-pressure, conversion-parity, and
  trigger-state modules;
- keeps the cross-period coverage-fractured debt-floor source outside the
  aggregate pending a dedicated coverage audit;
- uses the explicit healthy state-gated wrapper rather than the original
  module.

It is not imported by `defs.__init__`, any live config, or a model config.

## Commands And Evidence

```powershell
py -3.11 -m pytest -q \
  tests/test_research_factor_mining_aggregate_catalog_v2.py \
  tests/test_research_factor_mining_aggregate_catalog_v3.py \
  tests/test_research_factor_mining_intraday_state_gated_microstructure_healthy_catalog_v1.py

py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_aggregate_catalog_v3.py \
  tests/test_research_factor_mining_aggregate_catalog_v3.py

py -3.11 -m ruff format --check \
  cbond_on/domain/factors/defs/research_factor_mining_aggregate_catalog_v3.py \
  tests/test_research_factor_mining_aggregate_catalog_v3.py

py -3.11 harness/tools/run_factor_mining_expansion.py \
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_aggregate_catalog_v3 \
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_aggregate_catalog_v3_smoke_20260428_20260429 \
  --start 2026-04-28 --end 2026-04-29 --execute
```

- focused tests: `6 passed`
- Ruff check/format: passed
- smoke root:
  `D:/cbond_on/research_scratch/factor_mining_20260803_aggregate_catalog_v3_smoke_20260428_20260429`
- manifest/catalogue: one of each, research-only
- FactorStore files: two, stable 97-column schema, with 330 and 324 rows
- integrity: unique `(dt, code)`, zero Inf, zero all-empty signals, and zero
  cross-sectionally constant signals on either smoke day

## Boundary

The batch wrote only the named scratch root.  It did not write a production
FactorStore, DB, scheduler, live configuration, model configuration, mask, or
trading rule.

## Next Step

After the remaining independent candidate-smoke and coverage audits complete,
compose one fresh aggregate successor, repeat this no-write preflight and
two-day smoke, then launch a single new full-window scratch build.  Merge it
with v7 only after its immutable full root passes audit; rerun the fixed global
screen and exact independent-set optimizer from that new merged root.
