# Daily relative return-flow rank coupling v2 preflight (research-only)

## Candidate family and fixed contract

This is the future-build successor of the one-signal v1 rank-coupling module.
The v1 smoke remains an immutable audit artifact and must not be merged beside
v2.  v2 has one genuine family, not renamed subfamilies:
`prior_relative_return_flow_rank_coupling`.

| Signal | Strict-prior formula | Mean daily Pearson IC | Discovery / validation / holdout |
| --- | --- | ---: | --- |
| `drrc_return_amount_rank_spearman60` | corr(return-rank, amount-rank) | +0.023629 | +0.018622 / +0.033116 / +0.029092 |
| `drrc_return_trade_size_rank_spearman60` | corr(return-rank, rank(amount/deal)) | +0.023481 | +0.022064 / +0.027721 / +0.023490 |

Each value uses the latest up-to-60 strict-prior daily-price source sessions,
at least 45 joint finite observations, and finite terminal ranks.  Each
historical rank uses the full valid date-local daily convertible-bond market.
The factor sees no label, mask, PnL, score, or `o_0005`; strict T-1 `o_0005`
is applied only by the later fixed research screen.

## Redundancy preflight

- amount/trade-size coupling within the one family: `0.724683` in the direct
  fixed-pool audit (a conservative independent reproduction was `0.745407`);
  both remain strictly below the within-family 0.80 gate;
- maximum v7 quality-eligible redundancy: `0.475931` for amount and
  `0.444370` for trade size, each with 381 common days;
- trade-size coupling's maximum relation to the five current independently
  pre-screened candidates is `0.506524` (trade-size-sign MI), below 0.70;
- rank-tail contradiction is a separate nonlinear family and has cross-family
  redundancy `0.262079` / `0.223802` with amount / trade-size coupling.

These are zero-write candidate checks, not final admission.  A fresh global
screen must recompute every pair after immutable full roots are merged.

## Boundary

The module is research-only, not imported by `defs.__init__`, and absent from
configs, contracts, Rust/live paths, DB, scheduler, and production output.  It
requires focused tests, a no-write full preflight, and a fresh smoke before any
full-root launch after aggregate v5 finishes.

## Static and isolated scratch smoke

Focused verification passed:

```text
py -3.11 -B -m pytest -q \
  tests/test_research_factor_mining_daily_relative_rank_flow_coupling_v2.py \
  tests/test_run_factor_mining_expansion.py -p no:cacheprovider
# 10 passed

py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_relative_rank_flow_coupling_v2.py \
  tests/test_research_factor_mining_daily_relative_rank_flow_coupling_v2.py
# All checks passed
```

The no-write full preflight passed with the previously absent intended root
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_relative_rank_flow_coupling_v2_full`.
Its isolated 2026-04-28 smoke wrote only
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_relative_rank_flow_coupling_v2_smoke_20260428`.
The FactorStore has 330 unique `(dt, code)` rows, exact 14:30 timestamps, two
catalogue columns, 323 finite values and seven explicit `NaN` per signal, zero
`Inf`, and 323 distinct finite values per signal.  The manifest is marked
research-only.  No full root has been started.
