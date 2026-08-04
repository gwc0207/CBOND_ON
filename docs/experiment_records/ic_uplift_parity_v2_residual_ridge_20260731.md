# Pool-free parity v2 residual-Ridge IC experiment（2026-07-31）

## Decision

`parity_adjusted_stock_lag_v2` does **not** improve the frozen Regsim score
contract sufficiently to continue.  Both pre-frozen anchored-Ridge variants
are rejected: the parity-only arm is essentially flat in validation and lower
than Regsim in the final-reporting segment; the parity-plus-vwap arm is lower
in both segments.  This is research-only evidence and does not change any
live factor, model, mask, strategy, DB, scheduler, or output.

## Fixed question and controls

Question: can a causal, pool-free stock/bond tail-gap feature improve Regsim
cross-sectional IC when only score-model inputs are changed?

Unchanged controls:

- `strategy01_topk_turnover`, every existing mask including `o_0005`, trading
  windows, turnover, fees, benchmark, and execution contract;
- live config, Champion, model state, DB, scheduler, and live outputs;
- score/evaluate separation: score T may read labels strictly before T only;
  evaluate is the first stage permitted to open T's 14:42 label.

The factor is deliberately not a mask input:

```text
kappa(i,T) = clip(conv_value(i,T-1) / cb_close_price(i,T-1), 0, 2)
factor(i,T) = kappa(i,T) * stock_tail_return(i,T,14:00--14:29)
              - cbond_tail_return(i,T,14:00--14:29)
```

It uses only the explicit `< T` `daily_base` mapping and parity; no
`o_0005`/pool data is read by the factor.  Missing panel/schema context is
fail-fast.

## Build and input audit

- Implementation: `cbond_on/domain/factors/defs/parity_adjusted_stock_lag_v2.py`
  (SHA-256 `9c61e03264440970c120085bce74e2f9cfc16b8d9d31133dfd273e8e66af7d80`).
- Research build config SHA-256:
  `fb6c00e2ea4015f5398a1ba467ff571dd2fa244511230d70b27df636b3e21f92`.
- Scratch FactorStore:
  `D:/cbond_on/research_scratch/ic_uplift_oos_20260731/parity_v2/factor_data/factors/T1430/`.
- Clean-direct FactorBatch ran only with a scratch paths profile and the Python
  engine.  The initial two-worker run was stopped for memory pressure after
  its completed files were verified; it resumed one day at a time with
  `refresh=false`, `overwrite=false`.
- In the aligned `2024-05-08..2026-06-05` primary-factor/Regsim-score window,
  the sidecar covers 504/505 days.  The sole missing day is `2026-04-23`:
  the bond panel had 162,987 tail rows across 323 codes, while the stock panel
  ended at `11:18:49` and had zero 14:00--14:29 rows.  No value was filled or
  backdated; later dates were resumed from `2026-04-24`.
- Every covered file has exactly the v2 column, a unique `MultiIndex(dt, code)`,
  `dt=14:30:00` on its factor day, and finite values.  Cross-day row count is
  189--536 (median 466.5); value range is -11.51% to +22.07%.

## Frozen score specification

Before score/evaluate, exactly two plans were frozen in
`harness/tools/ic_uplift_oos_residual_ridge.py` under `parity_v2_r1`:

| Arm | Inputs | Target | Lookback / alpha |
| --- | --- | --- | --- |
| parity only | parity v2 | `z(y) - z(Regsim)` | 120 days / 20 |
| parity + vwap | parity v2, `vwap_30m` | `z(y) - z(Regsim)` | 120 days / 20 |

Each day is cross-sectionally z-scored; every training day is equally weighted.
No final-period result selected features, weights, alpha, or a threshold.

## Leakage audit and artifacts

Valid result root:

```text
D:/cbond_on/results/experiments/ic_uplift_oos_20260731/parity_v2_r1_20260731_002/
```

- Score stage: 444 output days (`2024-08-01..2026-06-05`), 888 model-day audit
  rows, and zero `train_label_max_day >= score_day` violations.
- `2026-04-23` is the only `feature_stage_skipped` day in the score manifest.
- `evaluate` was executed only after `oof_scores.parquet` existed.
- An earlier `_001` score-only root omitted the Regsim baseline column due to a
  runner-output bug.  It was never evaluated or interpreted, was not
  overwritten, and `_002` is the corrected comparable run.  The regression
  test now asserts baseline preservation.

## Results

Daily cross-sectional score IC, equal-weighted by day:

| Segment | Regsim Pearson IC | parity-only delta | parity+vwap delta |
| --- | ---: | ---: | ---: |
| Development, 285 days | 0.05094 | +0.00003 (t=0.13) | -0.00069 (t=-1.77) |
| Validation, 136 days | 0.00372 | +0.00016 (t=1.02) | -0.00208 (t=-2.32) |
| Final reporting, 23 days | 0.02410 | -0.00020 (t=-1.00) | -0.00075 (t=-0.28) |

The raw parity diagnostic points in the same direction: its partial Pearson
IC conditional on Regsim is +0.01585 in development and +0.01688 in
validation, but -0.01791 in final reporting.  The feature therefore has some
earlier rank/residual association but no stable incremental signal under this
fixed score mapping.

No strategy backtest was run, because the score candidate failed the
predeclared IC gate.  The 23-day final segment is chronological historical
evidence, not a new prospective shadow.

## Verification

```powershell
py -m pytest tests/test_parity_adjusted_stock_lag_v2.py tests/test_ic_uplift_oos_residual_ridge.py -q
# 8 passed

py harness/tools/ic_uplift_oos_residual_ridge.py score \
  --output-root D:/cbond_on/results/experiments/ic_uplift_oos_20260731/parity_v2_r1_20260731_002 \
  --candidate-set parity_v2_r1 \
  --sidecar-factor-root D:/cbond_on/research_scratch/ic_uplift_oos_20260731/parity_v2/factor_data/factors/T1430 \
  --start 2024-05-08 --end 2026-06-05

py harness/tools/ic_uplift_oos_residual_ridge.py evaluate \
  --output-root D:/cbond_on/results/experiments/ic_uplift_oos_20260731/parity_v2_r1_20260731_002 \
  --validation-start 2025-10-08 --final-start 2026-05-06
```
