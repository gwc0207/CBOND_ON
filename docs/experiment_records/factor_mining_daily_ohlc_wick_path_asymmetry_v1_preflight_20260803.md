# Daily OHLC wick/path asymmetry v1 - research-only preflight

## Fixed contract

- requested score start: `2025-01-01`; actual score calendar:
  `2025-01-02..2026-07-30` (381 days);
- factor timestamp: T1430 / `14:30`; label: same-score-day `14:42`;
- universe is applied only by the later screen as the existing strict
  previous-trading-day `quant_factor_dev.researcher_xuvb.o_0005` allowlist;
- final gates remain absolute mean daily Pearson IC strictly above `0.02`, at
  least 250 valid days, at least 50 days in each official 228/76/77
  chronological partition, 200 common redundancy days, within-family
  redundancy below `0.80`, cross-family redundancy below `0.70`, and a
  maximum selected count of 100.

No mask, label definition, trading rule, production FactorStore, model/live
configuration, database, scheduler, or production result was changed.

## Family and strict-PIT formula

Implementation:

```text
cbond_on/domain/factors/defs/research_factor_mining_daily_ohlc_wick_path_asymmetry_v1.py
kernel: factor_mining_daily_ohlc_wick_path_asymmetry_v1
family: prior_daily_ohlc_wick_path_asymmetry
```

It declares only the already-whitelisted `market_cbond.daily_price` fields
`exchange_code`, `prev_close_price`, `open_price`, `high_price`, `low_price`,
and `close_price`, with a 75-session context request.  It filters every source
row to `trade_date < score_date`, requires the latest strict-prior source
anchor for each code, and never reads `o_0005`, labels, PnL, scores, files, a
database, or Redis inside factor code.

For a valid completed historical daily OHLC bar, define:

```text
wick_asymmetry =
    [high - max(open, close) - (min(open, close) - low)] / (high - low)
intraday_return = log(close / open)
intraday_range = log(high / low)
```

The catalogue contains two genuine members of the same price-auction-shape
family, not window/sign/scale variants:

| Signal | Formula | Interpretation |
| --- | --- | --- |
| `dohw_mean_wick_asymmetry60` | 60-session mean historical `wick_asymmetry` | persistent upper-versus-lower rejection imbalance |
| `dohw_intraday_sign_range_asymmetry60` | mean historical `intraday_range` on positive `intraday_return` sessions minus its mean on negative sessions | whether completed range expands more on up than down auctions |

Both require a finite terminal OHLC state and at least 45 finite historical
observations.  The directional-range signal additionally requires at least 8
positive and 8 negative completed sessions.  Invalid OHLC bars, stale anchors,
and malformed duplicate normalized daily keys fail closed to `NaN` or an
explicit error; no missing value is filled with zero.

## Batch pre-screen, fixed before inspection

A no-write exploratory batch evaluated 19 daily-OHLC path hypotheses together:
gap/intraday-return and gap/range correlations, location and wick persistence,
gap reversal, close location, wick asymmetry, path efficiency, sign-conditioned
range/location asymmetry, high-range/high-gap conditional responses, and
cross-sectional range/gap-rank dynamics.  It used the fixed 381 score days,
strict T-1 pool at scoring time, and same-day 14:42 labels.  The 17 unselected
hypotheses had maximum absolute mean daily Pearson IC `0.018361` and were not
implemented.

The two pre-registered retained candidates were:

| Signal | Mean daily Pearson IC | Discovery | Validation | Holdout | Valid days |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dohw_mean_wick_asymmetry60` | +0.025656 | +0.023414 | +0.033361 | +0.024689 | 381 |
| `dohw_intraday_sign_range_asymmetry60` | +0.022637 | +0.013629 | +0.027914 | +0.044103 | 381 |

This is selection evidence, not final admission: the full-window number was
used only to decide whether the family deserves an isolated FactorStore build,
and the split values remain visible for overfit assessment.

## Provisional redundancy audit

The two values were independently reconstructed on every score day and
compared under the canonical maximum of mean daily absolute Pearson and
Spearman correlations, using the fixed T-1 pool.

| Candidate | Largest v7 quality-eligible redundancy | Existing factor | Common days |
| --- | ---: | --- | ---: |
| `dohw_mean_wick_asymmetry60` | 0.335210 | `exp_stick_price_change_rate` | 379 |
| `dohw_intraday_sign_range_asymmetry60` | 0.509127 | `twap_segment_volatility` | 381 |

Their within-family redundancy was `0.298079` over 381 common days, below the
0.80 gate.  These are v7-only pre-checks.  The final result must be recomputed
against every quality-eligible factor in the later, newest complete global
pool; a v7 pre-check cannot establish final low correlation.

## Static, preflight, and smoke evidence

Focused verification:

```text
py -3.11 -B -m pytest -q \
  tests/test_research_factor_mining_daily_ohlc_wick_path_asymmetry_v1.py \
  tests/test_research_factor_mining_daily_relative_rank_flow_coupling_v2.py \
  tests/test_factor_mining_screen.py \
  tests/test_run_factor_mining_expansion.py -p no:cacheprovider
# 26 passed

py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_ohlc_wick_path_asymmetry_v1.py \
  tests/test_research_factor_mining_daily_ohlc_wick_path_asymmetry_v1.py
# All checks passed
```

The no-write full-window preflight passed with the previously absent intended
root:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_ohlc_wick_path_asymmetry_v1_full
```

The isolated 2026-04-28 scratch smoke wrote only:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_ohlc_wick_path_asymmetry_v1_smoke_20260428
```

It has 330 unique `(dt, code)` rows, exact `2026-04-28 14:30:00` timestamps,
two exact catalogue columns, 323 finite values and seven explicit `NaN` values
per signal, zero `Inf`, and nonconstant values.  An independent raw-data
reconstruction on the same 330 panel codes reproduced the smoke output exactly
for both signals (same `NaN` mask, maximum absolute difference `0.0`).  Its
run manifest is marked `research_only`.

## Deferred full-build boundary

The aggregate v5 full build remains active, so this module has no full root,
is not imported by `defs.__init__`, and is absent from live/model config,
factor contracts, Rust/live paths, DB, scheduler, and production FactorStore.
After v5 naturally completes and its full root passes immutable-root audit,
this family may be scheduled serially in a fresh root.  It can enter only a
new all-complete-roots catalogue, outer-union merge, fixed-contract screen,
and exact MIS selection; it must never be appended directly to an accepted
list.
