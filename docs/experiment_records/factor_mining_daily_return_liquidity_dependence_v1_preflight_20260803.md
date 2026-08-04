# Daily return/liquidity information dependence v1 - strict-PIT preflight

## Question and immutable research contract

Can a prior-only daily return/liquidity dependence state provide a distinct
research candidate under the fixed factor-mining IC contract?

- requested IC start: 2025-01-01; actual score calendar:
  2025-01-02..2026-07-30 (381 days);
- factor timestamp: T1430 / 14:30;
- label timestamp: same score day / 14:42;
- universe: strict previous-trading-day
  quant_factor_dev.researcher_xuvb.o_0005, with no no-filter fallback;
- metric: mean of daily cross-sectional Pearson IC;
- quality gates: absolute mean IC strictly above 0.02, at least 250 valid
  days and at least 50 valid days in each 228/76/77 chronological
  discovery/validation/holdout partition;
- redundancy: at least 200 common valid days, maximum of mean daily absolute
  Pearson and Spearman strictly below 0.80 within family and 0.70 across
  families.

Existing masks, labels, trading rules, model configuration, live
configuration, database, scheduler, production FactorStore, and result roots
are out of scope.

## New research-only family

Implementation:

    cbond_on/domain/factors/defs/research_factor_mining_daily_return_liquidity_dependence_v1.py
    kernel: factor_mining_daily_return_liquidity_dependence_v1
    family: prior_return_liquidity_information_dependence
    signal: rlmi_return_liquidity_sign_mutual_information60

The signal is the normalized mutual information between:

    sign(log(close_price / prev_close_price))
    sign(delta log(amount))

on an up-to-60-session strict-prior window. It uses a fixed 3 x 3
Jeffreys-pseudocount table. Only adjacent rows whose raw daily source session
positions differ by exactly one may form an amount-change pair; a suspension
or missing source row is never compressed into a synthetic adjacent
observation. A valid output requires 45 source rows, 40 valid adjacent pairs,
and a valid newest strict-prior pair.

The kernel declares only:

    market_cbond.daily_price:
      exchange_code, prev_close_price, close_price, amount
      lookback_days=75

All source rows with trade_date >= score_date are discarded. A code with a
stale latest source row, a nonpositive latest amount, missing required fields,
or duplicate normalized daily key fails closed to NaN.

## Zero-write fixed-contract pre-screen

The pre-screen first resolved all 381 strict T-1 allowlists, then read
same-score-day 14:42 labels and evaluated daily Pearson IC. It used raw
daily-price rows only through 2026-07-29 for the final score day
2026-07-30.

| Signal | Mean IC | Discovery (228) | Validation (76) | Holdout (77) | Valid days | Mean cross-section |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rlmi_return_liquidity_sign_mutual_information60 | +0.028962 | +0.017724 | +0.044949 | +0.046456 | 381 | 390.7 |

Two pre-registered neighboring temporal candidates did not clear the IC gate
and were not implemented:

| Candidate | Mean IC |
| --- | ---: |
| return time-reversal asymmetry | -0.001119 |
| three-point return permutation entropy | +0.002665 |

The screen used the official 228/76/77 partition helper, not a hard-coded
split. No pre-screen output file, FactorStore, or result directory was
created.

## v7 global-pool redundancy pre-check

The qualifying candidate was compared on each of the 381 fixed-pool days with
all 64 v7 IC-and-validity-eligible factors. The same daily absolute
Pearson/Spearman maximum definition was used.

| Candidate | Largest redundancy | Existing factor | Common valid days | Result |
| --- | ---: | --- | ---: | --- |
| rlmi_return_liquidity_sign_mutual_information60 | 0.531886 | twap_segment_volatility | 381 | passes provisional cross-family below 0.70 check |

This is only a v7 pre-check. The final check must be recomputed against the
fresh complete global candidate pool after the independent FactorStore build.

## Focused verification

    py -3.11 -B -m pytest -q \
      tests/test_research_factor_mining_daily_return_liquidity_dependence_v1.py \
      tests/test_factor_mining_screen.py \
      tests/test_run_factor_mining_expansion.py \
      -p no:cacheprovider
    # 21 passed

    py -3.11 -m ruff check \
      cbond_on/domain/factors/defs/research_factor_mining_daily_return_liquidity_dependence_v1.py \
      tests/test_research_factor_mining_daily_return_liquidity_dependence_v1.py
    # All checks passed

The focused test suite covers registration/catalogue shape, strict score-day
and future-row exclusion, stale-source fail-closed behavior, nonpositive
latest amount fail-closed behavior, duplicate normalized-key rejection, and
terminal missing-session adjacency rejection.

## Selection consequence and full-build boundary

The amount-sign MI candidate is retained as a reproducible audit result, but
it is not scheduled for a full build: in the same fixed-pool redundancy
pre-check it conflicts with the selected trade-count MI at within-family
redundancy 0.856568, and with selected joint-transition entropy at
cross-family redundancy 0.777223. The thresholds were not relaxed.

The non-conflicting selected set is implemented separately in:

    cbond_on/domain/factors/defs/research_factor_mining_daily_return_liquidity_topology_v1.py

No full build was started while factor_mining_20260803_aggregate_catalog_v5_full
remains active. A complete immutable root for any later selected module must
enter a newly composed outer-union merge, fixed-contract global screen, and
exact MIS selection. It must never be appended directly to an accepted list
or promoted to a model or live chain.
