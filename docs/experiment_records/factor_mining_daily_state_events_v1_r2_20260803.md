# Daily state-event factor-family expansion (research-only)

## Question and fixed contract

Can locally available T-1 daily state changes add information dimensions that
are not merely another return, amount, TWAP, intraday-path, static valuation,
or trigger-revision window variant?

- score contract: T1430 / 14:30 factor and 14:42 label;
- intended screen: fixed 2025-01-02 through 2026-07-30 calendar with the
  previous-session `quant_factor_dev.researcher_xuvb.o_0005` universe;
- source rule: every daily input is restricted in the factor to
  `trade_date < score_date`; `daily_price` provides an independent T-1
  calendar anchor, and a stale `daily_base` terminal row produces `NaN`;
- scope: Python research catalogue only.  No live config, database,
  scheduler, production FactorStore, model, mask, or trading rule changed.

## Source audit

The context loader supports `market_cbond.daily_vwap`,
`market_cbond.daily_deriv`, and `market_cbond.daily_rating`, but the current
DataHub raw root contains none of their directories or parquet partitions:
all have `0/381` score-date and T-1 availability.  They were not fabricated
or requested by the module.

The three actual local daily tables have complete T-1 file availability for
all 381 evaluation score days: `daily_price`, `daily_base`, and `daily_twap`.
This batch uses only previously underused state/event fields from the first
two tables.

Rejected source dimensions:

- `trigger_process` exactly matched `(trigger_cum_days, trigger_reach_days)`
  on all 126,573 jointly numeric observations; the revised form matched on
  all 113,991.  It is an alias, not a new factor input.
- `trigger_type` / `trigger_type_name` were physically available for only 68
  days; `trigger_date` and `trigger_date_revise` had no day with 30 valid
  bonds.  Both fail the fixed screening-support requirement.
- Standalone convertible-bond and stock prior-close adjustment events occurred
  on only 228 and 186 T-1 `o_0005` days.  The module instead uses cross-asset
  combinations that are non-constant whenever either leg has an event
  (`302/381` such days).

## Implemented catalogue

Module: `cbond_on/domain/factors/defs/research_factor_mining_daily_state_events_v1.py`

Version: `20260803_daily_state_events_v1_r2`; 24 candidates in four families.

1. `capital_supply_transition` — one-day log remaining-size change, shrink
   magnitude, 20-day event rate / cumulative change, 60-day event recency,
   and a shrink-by-current-liquidity impulse.  This uses *changes* of
   outstanding supply, not the existing static `remain_size` or float-adjusted
   amount level.
2. `call_activation_lifecycle` — observed activation flag, contiguous active
   age, activation recency, and the current / five-session velocity / required
   days of the revised contract progression.  Existing families use a raw
   original-versus-revised wedge; this family uses the lifecycle conditional on
   being active.
3. `prior_close_adjustment_discontinuity` — net and absolute cross-asset
   adjusted-prior-close shifts, signed gap, magnitude, and event rate /
   recency.  It represents prior published corporate-action discontinuities,
   not a T1430 price-path transformation.
4. `bond_stock_quantity_regime` — own-history-normalised relative volume,
   trade-unit, and notional-per-unit surprises plus volume coupling, lead-lag,
   and imbalance persistence.  This is deliberately distinct from the prior
   amount/deal share family: quantities are first normalised within each leg
   before cross-asset comparison.

All inputs are declared through `daily_requirements()`; the module has no file
or DB access and does not fill missing source values with zero.

## Verification

Focused tests:

```powershell
py -3.11 -m pytest -q tests/test_research_factor_mining_daily_state_events_v1.py tests/test_run_factor_mining_expansion.py
```

Result: `10 passed`.

The test suite confirms explicit requirements, import-only registration,
strict score-day/future-row exclusion, missing-field and duplicate-row hard
failure, stale-base rejection, no `Inf`, and representative formulas.

Real isolated smoke:

```powershell
py -3.11 harness/tools/run_factor_mining_expansion.py `
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_daily_state_events_v1 `
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_daily_state_events_v1_r2_smoke_20260428 `
  --start 2026-04-28 --end 2026-04-28 --execute
```

Output FactorStore:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_state_events_v1_r2_smoke_20260428/factor_data/factors/T1430/2026-04/20260428.parquet
```

It has 330 unique `(dt, code)` rows and all 24 expected columns: 7,707 finite
cells, 213 `NaN`, and zero `Inf`.  The matching run manifest records four
families, 24 signals, Python engine, clean-direct inputs, and disabled
backtest/screening/report stages.

## Status and next step

This is a smoke-validated candidate catalogue, not an IC result and not a
promotion.  No full-window build has been started.  If capacity permits, the
next isolated root should be newly created as:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_state_events_v1_r2_full
```

Only after an immutable 381-day build, merged fixed-universe screen, and the
existing IC / family-redundancy gates may any candidate be called accepted.
