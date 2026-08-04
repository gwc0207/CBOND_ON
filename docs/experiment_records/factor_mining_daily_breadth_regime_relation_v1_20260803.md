# Daily breadth-regime relation v1 — 2026-08-03

## Status

Research-only candidate family.  It is not imported by `defs.__init__`, and is
not present in live/model configuration, factor contracts, production
FactorStore, database, or scheduler.  The full-window numbers below are a
pre-screen and the single-day build is a mechanical/PIT smoke; neither is a
final all-pool admission decision.

## Fixed contract

- Requested IC start: `2025-01-01`; actual score calendar:
  `2025-01-02..2026-07-30` (381 sessions).
- Factor timestamp: same-day T1430/`14:30`; label: same-day `14:42`.
- Input: only `market_cbond.daily_price` rows whose `trade_date < score_date`.
- Formula source cross-section: all valid convertible-bond daily-price rows;
  it never uses `o_0005` as a factor input.
- Later fixed screen: strict T-1 `o_0005`, `abs(mean daily Pearson IC) > 0.02`,
  at least 250 valid days, at least 50 valid days in each chronological
  60/20/20 partition, at least 200 common redundancy days, within-family
  redundancy `< 0.80`, and cross-family redundancy `< 0.70`.

## Mechanism

`brr_rank_low_high_spread60` is the difference between a security's mean
cross-sectional rank of strict-prior log return during the low-breadth and
high-breadth regimes of the latest 60 completed sessions.  On each historical
session, breadth is the fraction of the complete valid bond source
cross-section with a positive daily return.  Low/high regimes are the local
25th/75th breadth quantiles.  A missing or invalid terminal strict-prior rank,
insufficient observations, or insufficient observations in either regime
returns `NaN`; missing data is never zero-filled.

## Zero-write full-window pre-screen

| factor | overall daily Pearson IC | discovery | validation | holdout | valid days | max v7 redundancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `brr_rank_low_high_spread60` | -0.020012648 | -0.019423 | -0.031035 | -0.010878 | 381 | 0.540629053 |

The provisional v7 comparison had no cross-family conflict at `>= 0.70`.
This is a deliberately marginal candidate: its absolute IC clears the strict
gate by only `0.000012648`, and its holdout magnitude is below `0.02`.  It
must therefore remain provisional until the immutable complete roots are
outer-union merged and re-screened globally.

## Implementation and verification

Research-only files:

- `cbond_on/domain/factors/defs/research_factor_mining_daily_breadth_regime_relation_v1.py`
- `tests/test_research_factor_mining_daily_breadth_regime_relation_v1.py`
- `cbond_on/domain/factors/defs/research_factor_mining_daily_orthogonal_batch_v8.py`
- `tests/test_research_factor_mining_daily_orthogonal_batch_v8.py`

The implementation declares `market_cbond.daily_price` columns
`exchange_code`, `prev_close_price`, and `close_price`, with a 65-day
lookback.  It uses no stock panel, bond-stock map, reconstructed OHLC, direct
file read, or external service.  It is Python-only research code; no Rust or
live registration was added.

Focused verification after correcting the pandas-compatible session slice:

```powershell
py -3.11 -B -m pytest -q tests/test_research_factor_mining_daily_breadth_regime_relation_v1.py tests/test_research_factor_mining_daily_observable_seasoning_v1.py tests/test_research_factor_mining_daily_orthogonal_batch_v5.py tests/test_research_factor_mining_daily_orthogonal_batch_v6.py tests/test_research_factor_mining_daily_orthogonal_batch_v7.py tests/test_research_factor_mining_daily_orthogonal_batch_v8.py tests/test_run_factor_mining_expansion.py -p no:cacheprovider
py -3.11 -m ruff check cbond_on/domain/factors/defs/research_factor_mining_daily_breadth_regime_relation_v1.py cbond_on/domain/factors/defs/research_factor_mining_daily_orthogonal_batch_v8.py tests/test_research_factor_mining_daily_breadth_regime_relation_v1.py tests/test_research_factor_mining_daily_orthogonal_batch_v8.py
```

Result: `23 passed`; Ruff clean; scoped `git diff --check` clean.

The generic runner first passed a no-write preflight, then completed an
isolated one-day Python-engine smoke:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_breadth_regime_relation_v1_smoke_20260428
```

The T1430 parquet had 330 unique `(dt, code)` rows at exactly
`2026-04-28 14:30:00`, 323 finite values, seven explicit fail-closed `NaN`s,
and zero `Inf`.  A separate raw DataHub calculation loaded strict-prior daily
price files, independently rebuilt the complete-market ranks and breadth, and
used the final 60 sessions (`2026-01-23..2026-04-27`).  It matched the output
NaN mask exactly and had maximum absolute difference `0.0` across all 323
finite rows.

## Next action

`daily_orthogonal_batch_v8` is a future scratch catalogue only.  Do not start
its full root while the independent aggregate-v5 full build remains active.
After that build naturally completes and passes immutable-root audit, use a
fresh serial full build, audit it, outer-union all complete roots, run the
fixed global screen, and then exact MIS.  A final global conflict or marginal
IC can still reject this candidate.
