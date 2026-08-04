# Wave80 liquidity-volatility residual-Ridge diagnostic (2026-07-31)

## Decision

Reject both frozen `wave80_r1` arms.  Neither arm improves the current Regsim
cross-sectional Pearson IC in validation or final reporting.  This is a
read-only historical diagnostic of a legacy production-FactorStore column; it
does not make that factor eligible for live use.

## Fixed question and controls

Question: can the pre-existing research-only
`t1430_w80_liq_vol_balance_45m_l5_v1` column improve the Regsim score through
a causal residual Ridge correction?

Unchanged controls:

- all strategy rules, the fixed `o_0005` and every other mask, execution
  windows, turnover, fees, and benchmark;
- live config, Champion, DB, scheduler, model state, and live outputs;
- `anchored_residual`, Ridge `alpha=20`, and exactly 120 prior score-calendar
  days for every score day.

Before scoring, exactly two arms were frozen:

| Arm | Inputs | Target | History / alpha |
| --- | --- | --- | --- |
| Wave80 | `t1430_w80_liq_vol_balance_45m_l5_v1` | `z(y) - z(Regsim)` | 120 days / 20 |
| Wave80 + VWAP | Wave80 + `vwap_30m` | `z(y) - z(Regsim)` | 120 days / 20 |

## Input and leakage audit

- The existing primary FactorStore and Regsim score calendar have a complete
  346-day intersection from `2025-01-02` through `2026-06-10`.
- All 346 date sections have 100% factor/Regsim cross-sectional coverage:
  no NaN filling, day filter, pool, or mask was used.
- The research runner was hardened before execution: primary-store routing is
  explicit; every training window must contain all 120 immediately preceding
  Regsim score-calendar days; a missing factor/score/usable-label day blocks
  scoring until it has aged out rather than being bridged.
- Score stage emitted 226 days from `2025-07-04` onward, 452 candidate-day
  audit rows, exactly 120 training days per row, and zero
  `train_label_max_day >= score_day` violations.
- Same-day 14:42 labels were only opened by the separate `evaluate` command
  after `oof_scores.parquet` and `score_manifest.json` existed.

## Results

Daily equal-weighted score IC, paired with Regsim on the same dates:

| Segment | Regsim Pearson IC | Wave80 delta | Wave80 + VWAP delta |
| --- | ---: | ---: | ---: |
| Development, 63 days | 0.04668 | -0.00021 (t=-1.68) | -0.00163 (t=-3.19) |
| Validation, 137 days | 0.00476 | -0.00009 (t=-1.20) | -0.00230 (t=-2.73) |
| Final reporting, 26 days | 0.02549 | -0.00005 (t=-0.85) | -0.00037 (t=-0.13) |

No strategy backtest was run because neither predeclared arm passed the IC
gate.  The 26-day final segment is only a chronological historical screen, not
a prospective shadow.

## Caveat and follow-up

The legacy factor consumes T1430 panels and no label, mask, pool, daily,
stock, or DB data.  Its FactorStore timestamp alone does not independently
prove a strict 14:29 historical upstream cutoff, so it is research-only even
if it had passed.  The separate legacy `amount_accel_depth` Wave80 formula is
not used here: its direct summation of cumulative amount is economically
mis-specified.  A new scratch-only delta-amount version must be specified,
tested, built, and evaluated as an independent candidate without tuning these
rejected arms.

## Reproduction

```powershell
py -m pytest tests\test_ic_uplift_oos_residual_ridge.py tests\test_parity_adjusted_stock_lag_v2.py -q
# 12 passed

py harness\tools\ic_uplift_oos_residual_ridge.py score `
  --output-root D:\cbond_on\results\experiments\ic_uplift_oos_20260731\wave80_r1_20260731_001 `
  --candidate-set wave80_r1 --start 2025-01-02 --end 2026-06-10

py harness\tools\ic_uplift_oos_residual_ridge.py evaluate `
  --output-root D:\cbond_on\results\experiments\ic_uplift_oos_20260731\wave80_r1_20260731_001 `
  --validation-start 2025-10-08 --final-start 2026-05-06
```
