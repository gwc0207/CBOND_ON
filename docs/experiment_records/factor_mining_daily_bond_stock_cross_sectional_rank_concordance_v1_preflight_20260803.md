# Daily bond-stock cross-sectional rank concordance v1 preflight (research-only)

## Question and fixed contract

Can a convertible bond's strict-prior agreement with the *relative standing*
of its mapped underlying stock contribute distinct factor information under the
fixed 2025-01-01 factor-mining contract?

- actual score dates: `2025-01-02..2026-07-30` (381 days);
- factor/label timestamps: T1430 `14:30` / same-score-day `14:42`;
- universe: strict previous-trading-day `o_0005`, no fallback;
- quality: absolute mean daily Pearson IC `>0.02`, at least 250 valid days,
  and at least 50 valid days in each fixed 228/76/77 partition;
- redundancy: minimum 200 common days; maximum of mean absolute daily
  Pearson/Spearman `<0.80` inside family and `<0.70` across family.

No mask, trading rule, label definition, live config, production FactorStore,
database, scheduler, model, or live result is touched.

## Family, data, and time visibility

Module:

```text
cbond_on.domain.factors.defs.research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1
```

Inputs are only strict-prior rows from:

- `market_cbond.daily_price`: `exchange_code`, `prev_close_price`,
  `close_price`;
- `market_cbond.daily_base`: `exchange_code`, `stock_code`,
  `stk_prev_close_price`, `stk_close_price`.

For every historical source date, the module ranks bond returns over all valid
convertible bonds and ranks distinct mapped-stock returns over all valid
underlyings.  Each bond then receives its exact mapped-stock rank; duplicate
underlying rows must have exactly the same return, otherwise the module fails
closed.  Rows on or after score date are discarded, global price/base anchors
must agree, and each output bond must have a valid anchor row.

| Family | Signal | Formula |
| --- | --- | --- |
| `prior_bond_stock_cross_sectional_rank_concordance` | `bssrc_upper_rank_tail_alignment60` | `P(bond return rank>=q75 | mapped-stock return rank>=q75)` over latest up-to-60 strict-prior sessions. |
| same | `bssrc_bond_stock_rank_correlation60` | Temporal Pearson correlation of the two date-local percentile ranks. |
| same | `bssrc_lower_rank_tail_alignment60` | `P(bond return rank<=q25 | mapped-stock return rank<=q25)` over the same window. |

Every signal requires 45 finite joint pairs, at least eight observations in
both stock tails, finite terminal ranks, and nonconstant ranks.  Missing or
ambiguous input never becomes zero.

## Zero-write 381-day pre-screen

An in-memory evaluation read local DataHub daily sources, resolved the fixed
T-1 pool for all 381 days before labels, and used the same 14:42 labels and
official partitions as the canonical screen.  It wrote no FactorStore or
result root.

| Signal | Valid days | Mean daily Pearson IC | Discovery | Validation | Holdout |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bssrc_upper_rank_tail_alignment60` | 381 | `+0.027487` | `+0.027505` | `+0.030039` | `+0.024914` |
| `bssrc_bond_stock_rank_correlation60` | 381 | `+0.024546` | `+0.023678` | `+0.029043` | `+0.022680` |
| `bssrc_lower_rank_tail_alignment60` | 381 | `+0.021430` | `+0.020642` | `+0.026160` | `+0.019096` |

The two pre-registered siblings—tail-alignment asymmetry and mean rank
displacement—had IC `+0.003567` and `+0.003562`; they are deliberately not
implemented.

## Provisional redundancy against v7

Each qualifying signal was compared to every 64 v7 IC-and-coverage-eligible
factor on the final daily common-sample definition:

| Signal | Max v7 redundancy | Factor | Common days |
| --- | ---: | --- | ---: |
| upper tail | `0.610472` | `twap_segment_volatility` | 381 |
| rank correlation | `0.579420` | `barrier_width_to_price` | 381 |
| lower tail | `0.631020` | `barrier_center_position` | 381 |

The within-family redundancies are `0.768985` (upper/correlation), `0.773341`
(correlation/lower), and `0.617142` (upper/lower), all below 0.80.  These are
only v7 pre-checks; v5 and later roots must be included in the final audit.

## Verification and smoke

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
py -3.11 -B -m pytest -q \
  tests/test_research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1.py \
  tests/test_run_factor_mining_expansion.py -p no:cacheprovider
# 11 passed

py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1.py \
  tests/test_research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1.py
```

The full-range runner preflight passed with an absent future root.  The
isolated `2026-04-28` scratch smoke wrote 330 unique `(dt, code)` rows at
exact 14:30.  Each signal had 290 finite values, 40 explicit `NaN`, zero
`Inf`, and 105–290 distinct finite values.  An independent raw-data
recalculation matched every finite smoke output exactly and matched its NaN
pattern.  This lower finite count reflects explicit lack of a valid mapped
stock/rank anchor, not imputation.

## Next step

The three signals extend the future `daily_orthogonal_batch_v4` catalogue,
which preserves every v3 source entry.  Its full scratch root remains absent
and must not launch until aggregate v5 naturally completes and passes the
immutable-root audit.  The ultimate decision still requires an all-complete
root outer-union merge, fixed global screen, and exact MIS.
