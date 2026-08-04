# Daily return/liquidity topology v1 - selected research-batch preflight

## Fixed contract

All results below use the immutable research contract:

- score calendar: 2025-01-02 through 2026-07-30 (381 days), requested start
  2025-01-01;
- factor: T1430 at 14:30;
- label: same score day at 14:42;
- universe: strict previous-trading-day o_0005, with all 381 allowlists
  resolved before label access;
- primary measure: mean daily cross-sectional Pearson IC;
- final quality/redundancy gates: absolute mean IC above 0.02, 250 valid
  days, 50 valid days per official 228/76/77 partition, 200 common
  redundancy days, within-family below 0.80 and cross-family below 0.70.

No mask, trading rule, label definition, production FactorStore, model, live
configuration, database, scheduler, or production results were changed.

## Pre-registered batch

Sixteen prior-only daily-price candidates were evaluated together, before
selecting any winner. They covered information dependence, lead-lag
orientation, joint state topology, return/liquidity run topology, state
regime drift, and magnitude coupling. Twelve did not clear the absolute IC
gate and were not implemented.

The four IC-qualified candidates were:

| Candidate | Mean IC | Discovery | Validation | Holdout | Valid days |
| --- | ---: | ---: | ---: | ---: | ---: |
| return/deal-sign MI | +0.031709 | +0.023150 | +0.042967 | +0.045942 | 381 |
| return/amount-sign MI | +0.028962 | +0.017724 | +0.044949 | +0.046456 | 381 |
| amount joint-transition entropy | -0.025608 | -0.014716 | -0.040768 | -0.042896 | 381 |
| return/trade-size-sign MI | +0.023813 | +0.014184 | +0.039243 | +0.037095 | 381 |

The amount-sign MI was excluded by the strict pairwise constraint. The
remaining three are the module catalogue:

| Family | Signal | Mean IC |
| --- | --- | ---: |
| prior_return_liquidity_information_dependence | rlmi_return_deal_sign_mutual_information60 | +0.031709 |
| prior_return_liquidity_information_dependence | rlmi_return_trade_size_sign_mutual_information60 | +0.023813 |
| prior_joint_return_liquidity_state_topology | rjst_amount_joint_transition_entropy60 | -0.025608 |

## Pairwise redundancy pre-check

The selected three were compared with one another, the 64 v7
IC-and-validity-eligible factors, and the two separately pre-screened
asymmetric equity-beta candidates. Each result is the maximum of mean daily
absolute Pearson and Spearman on the strict pool.

| Pair or comparison | Largest redundancy | Common days | Threshold | Result |
| --- | ---: | ---: | ---: | --- |
| deal-sign MI / trade-size-sign MI | 0.576732 | 381 | 0.80 | pass |
| deal-sign MI / joint-transition entropy | 0.675611 | 381 | 0.70 | pass |
| trade-size-sign MI / joint-transition entropy | 0.555389 | 381 | 0.70 | pass |
| deal-sign MI / all v7 eligible factors | 0.525887 | 381 | 0.70 | pass |
| trade-size-sign MI / all v7 eligible factors | 0.503478 | 381 | 0.70 | pass |
| joint-transition entropy / all v7 eligible factors | 0.444287 | 381 | 0.70 | pass |
| selected topology set / either asymmetric beta | 0.370797 or lower | 381 | 0.70 | pass |

These are provisional pre-checks only. A final complete-pool screen must
recompute all pairs after every immutable FactorStore root is merged.

## Implementation and PIT behavior

    cbond_on/domain/factors/defs/research_factor_mining_daily_return_liquidity_topology_v1.py
    kernel: factor_mining_daily_return_liquidity_topology_v1

The module declares only market_cbond.daily_price fields:
exchange_code, prev_close_price, close_price, amount, and deal, with 75
daily source sessions available for a 60-session factor window.

All source rows on or after score date are excluded. A code must have the
exact newest strict-prior source anchor. Each state change and each joint
transition requires consecutive raw session positions, with 45 source rows,
40 valid adjacent pairs, and a valid terminal pair. Missing values are
preserved as NaN; no zero filling or inferred calendar adjacency is used.

## Verification and deferred build

    py -3.11 -B -m pytest -q \
      tests/test_research_factor_mining_daily_return_liquidity_topology_v1.py \
      tests/test_research_factor_mining_daily_return_liquidity_dependence_v1.py \
      tests/test_research_factor_mining_daily_asymmetric_equity_beta_v1.py \
      tests/test_factor_mining_screen.py \
      tests/test_run_factor_mining_expansion.py \
      -p no:cacheprovider
    # 34 passed

    py -3.11 -m ruff check \
      cbond_on/domain/factors/defs/research_factor_mining_daily_return_liquidity_topology_v1.py \
      tests/test_research_factor_mining_daily_return_liquidity_topology_v1.py
    # All checks passed

The no-write expansion preflight validated the previously absent proposed
scratch root:

    D:/cbond_on/research_scratch/factor_mining_20260803_daily_return_liquidity_topology_v1_full

The isolated 2026-04-28 scratch smoke subsequently completed under:

    D:/cbond_on/research_scratch/factor_mining_20260803_daily_return_liquidity_topology_v1_smoke_20260428

It has 330 unique `(dt, code)` rows, exact `2026-04-28 14:30:00` timestamps,
the three exact catalogue columns, and zero `Inf`.  Each signal has 323 finite
values and seven explicit `NaN`; all three are nonconstant (247, 246, and 309
distinct finite values respectively).  The root manifest is research-only.
This is structural smoke evidence, not full-window admission.  No full
execution started because the independent aggregate v5 full build is active.
