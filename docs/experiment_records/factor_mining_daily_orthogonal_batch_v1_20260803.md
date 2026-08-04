# Daily orthogonal batch v1 - research-only build readiness

## Contract

This is a future scratch-only batch, not a live/model pack and not an incremental accepted-list update.

- score window: requested `2025-01-01`; actual `2025-01-02..2026-07-30` (381 days);
- factor timestamp: T1430 `14:30`; label timestamp: same-score-day `14:42`;
- universe: strict T-1 `o_0005` applied only by the later screen;
- final gates: absolute mean daily Pearson IC `>0.02`, 250 valid days, 50 valid days per official 60/20/20 partition, 200 redundancy days, within-family `<0.80`, cross-family `<0.70`, maximum 100 selections.

The catalogue is `research_factor_mining_daily_orthogonal_batch_v1.py`, version `20260803_daily_orthogonal_batch_v1`.

| Source module | Signals | Families |
| --- | ---: | ---: |
| daily asymmetric equity beta v1 | 2 | 1 |
| daily return/liquidity topology v1 | 3 | 2 |
| daily relative rank-flow coupling v2 | 2 | 1 |
| daily relative rank-tail contradiction v1 | 1 | 1 |
| daily OHLC wick/path asymmetry v1 | 2 | 1 |
| **Total** | **10** | **6** |

The catalogue uses rank-flow v2 rather than the original one-signal v1, so the amount-rank signal cannot be built twice. A direct audit found no signal or family collision with the active v5 catalogue (214 signals / 75 families at audit time).

Every member is research-only and consumes only T-1 `daily_price`/`daily_base` context through `FactorComputeContext`. No member reads labels, `o_0005`, PnL, files, databases, or Redis in factor code. No production FactorStore, live/model configuration, mask, trading rule, DB, scheduler, or result root changed.

## Verification

```text
py -3.11 -B -m pytest -q \
  tests/test_research_factor_mining_daily_orthogonal_batch_v1.py \
  tests/test_research_factor_mining_daily_ohlc_wick_path_asymmetry_v1.py \
  tests/test_research_factor_mining_daily_asymmetric_equity_beta_v1.py \
  tests/test_research_factor_mining_daily_return_liquidity_topology_v1.py \
  tests/test_research_factor_mining_daily_relative_rank_flow_coupling_v2.py \
  tests/test_research_factor_mining_daily_relative_rank_tail_contradiction_v1.py \
  tests/test_research_factor_mining_aggregate_catalog_v3.py \
  tests/test_run_factor_mining_expansion.py -p no:cacheprovider
# 38 passed

py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_orthogonal_batch_v1.py \
  tests/test_research_factor_mining_daily_orthogonal_batch_v1.py
# All checks passed
```

The no-write full preflight passed for the previously absent root `D:/cbond_on/research_scratch/factor_mining_20260803_daily_orthogonal_batch_v1_full`.

The combined one-day scratch smoke completed at `D:/cbond_on/research_scratch/factor_mining_20260803_daily_orthogonal_batch_v1_smoke_20260428`.

It produced 330 unique `(dt, code)` rows at exact `2026-04-28 14:30:00`, all ten catalogue columns, zero `Inf`, and no cross-sectionally constant signal. The two beta factors have 324 finite/six `NaN` values; each of the other eight has 323 finite/seven `NaN` values. Its manifest is `research_only`.

## Scheduling boundary

The active aggregate v5 root is still partial, so this batch has not been launched full-window. After v5 naturally exits and passes its immutable-root audit, this batch may run once in its fresh full root. It then must be root-audited and folded into a newly composed all-complete-roots catalogue, outer-union merge, strict global screen, and exact MIS. Its pre-screens and smoke cannot be called final accepted factors.
