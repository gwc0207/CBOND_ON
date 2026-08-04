# Daily bond-stock copula tail dependence v1 preflight (research-only)

## Question and fixed contract

Can strict-prior empirical tail co-movement between a convertible bond and its
mapped underlying stock contribute a low-redundancy candidate under the fixed
factor-mining contract?

- requested IC start: `2025-01-01`; actual score calendar:
  `2025-01-02..2026-07-30` (381 days);
- factor/label timestamps: `T1430 / 14:30` and same-score-day `14:42`;
- universe: strict prior-trading-day
  `quant_factor_dev.researcher_xuvb.o_0005`, no fallback;
- final gates: absolute mean daily Pearson IC strictly above `0.02`, at least
  250 valid days, at least 50 in each 60/20/20 chronological partition, and
  at least 200 common days for redundancy;
- redundancy: maximum of mean absolute daily Pearson and Spearman, strictly
  below `0.80` inside a family and `0.70` across families.

No live configuration, production FactorStore, database, scheduler, model,
mask, or trading rule is in scope.

## Family and implementation

Module:

```text
cbond_on.domain.factors.defs.research_factor_mining_daily_bond_stock_copula_tail_dependence_v1
```

It uses only strict-prior `market_cbond.daily_price`
(`exchange_code`, `prev_close_price`, `close_price`) and
`market_cbond.daily_base` (`exchange_code`, `stk_prev_close_price`,
`stk_close_price`) rows.  Every source row at or after score day `T` is
dropped; both sources must share the same latest strict-prior session, and a
bond must have a finite row at that anchor.  Thus no stale source, label,
future return, or current incomplete daily bar can enter the output.

| Family | Signal | Formula |
| --- | --- | --- |
| `prior_bond_stock_copula_tail_dependence` | `bsct_upper_tail_dependence60` | `P(rank(bond return)>=q75 | rank(stock return)>=q75)` over the latest up-to-60 strict-prior joint observations. |
| same | `bsct_lower_tail_dependence60` | `P(rank(bond return)<=q25 | rank(stock return)<=q25)` on the same window. |
| same | `bsct_tail_dependence_asymmetry60` | upper conditional tail dependence minus lower conditional tail dependence. |

All three require 45 finite joint observations, at least eight observations
in each stock tail, nonconstant returns, and a valid latest strict-prior pair;
otherwise they return `NaN`, never zero.

## Zero-write 381-day pre-screen

The check computed the module's exact helper formula in memory from local
DataHub daily inputs, then applied the same fixed T-1 `o_0005` list and
same-day 14:42 labels as the canonical screen.  It created no FactorStore,
result root, configuration, or live artifact.

| Signal | Valid days | Mean daily Pearson IC | Discovery | Validation | Holdout | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bsct_upper_tail_dependence60` | 381 | `+0.022812` | `+0.023074` | `+0.023443` | `+0.021414` | queued |
| `bsct_lower_tail_dependence60` | 381 | `+0.016786` | `+0.018293` | `+0.023104` | `+0.006088` | rejected at IC gate |
| `bsct_tail_dependence_asymmetry60` | 381 | `+0.006221` | `+0.005750` | `+0.000408` | `+0.013352` | rejected at IC gate |

Only the upper-tail expression is suitable for the later complete-pool test.
The module retains the three pre-registered family members so the eventual
canonical screen, not this pre-screen, is the final mechanism-level decision.

## Provisional v7 redundancy audit

`bsct_upper_tail_dependence60` was compared with every one of the 64 v7
IC-and-coverage-eligible factors using the final daily common-sample
Pearson/Spearman redundancy definition.  Its maximum was `0.503544` against
`barrier_side_asymmetry` (381 common days), so it passes the provisional
cross-family limit.  This is not final: it must be recomputed against v5 and
all later completed roots.

## Static, runner, and smoke verification

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
py -3.11 -B -m pytest -q \
  tests/test_research_factor_mining_daily_bond_stock_copula_tail_dependence_v1.py \
  tests/test_run_factor_mining_expansion.py -p no:cacheprovider
# 10 passed

py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_bond_stock_copula_tail_dependence_v1.py \
  tests/test_research_factor_mining_daily_bond_stock_copula_tail_dependence_v1.py
```

The generic no-write runner preflight passed for the exact full range and a
previously absent future root:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_bond_stock_copula_tail_dependence_v1_full
```

The isolated one-day scratch smoke at
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_bond_stock_copula_tail_dependence_v1_smoke_20260428`
produced 330 unique `(dt, code)` rows at exact `2026-04-28 14:30:00`.
Each signal had 323 finite values, seven `NaN`, zero `Inf`, and nonconstant
finite output.  The smoke validates mechanics only, not final admission.

## Next step

The qualified signal is exposed through the separate
`daily_bond_stock_copula_tail_dependence_healthy_catalog_v1` wrapper, then
appended to the future `daily_orthogonal_batch_v3` catalogue, which preserves
v2's source prefix without building the two already rejected expressions.  Do
not launch that full build while aggregate v5 is active.  After v5 naturally
completes and passes its immutable-root audit, v3 may run once in a fresh
scratch root, then all complete roots must be outer-union merged and globally
screened before exact MIS selection.
