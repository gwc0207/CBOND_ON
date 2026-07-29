# Experiment Record: MT-014 Similar60 path-full state training

## Experiment ID

- `MT-014`

## Question

- Does selecting the 60 most similar historical T1430 market-state days improve the daily LGBM relative to an otherwise identical model trained on the latest 60 days?

## Baseline

- Model/config: `models/lgbm/lgbm_screened_no_winsor_neutral_tminus1_latest60_pathfull360_cold_v1_20260727`.
- State: the same research-only repaired `path_full_t1430` state CSV; latest-date selection rather than nearest-state selection.
- Date window: `2025-10-30` to `2026-07-23`; 176 common score and strict-return days.
- Warm start/refit: disabled / daily cold-start.
- Training/validation: 60 train days + 20 validation days, all other LGBM, factor, neutralization, winsor and z-score settings inherited from the current neutral T-1 base.
- Label/sell window: buy `twap_1442_1457`; sell next trading day `twap_0930_0939`; complete cycle return.
- Benchmark: strict official benchmark under the same buy/sell and 1.0/1.2 bp fee contract.
- Universe: prior-trading-day `o_0005` only.

## Variant

- Model/config: `models/lgbm/lgbm_screened_no_winsor_neutral_tminus1_similar60_pathfull360_cold_v1_20260727`.
- Candidate pool: latest 360 effective historical state days, with 40 extra source days read to tolerate source gaps.
- Similarity: candidate-pool z-score plus Euclidean distance over 44 `path_full_t1430` features.
- Train/validation: nearest 60 / ranks 61--80.
- Sample weights: no HL20 time decay and no binary Regsim weighting, to isolate hard similar-day selection.

## Commands

```powershell
py -c "from cbond_on.infra.model.runners.train_lgbm import main; main(config_path=r'cbond_on/config/models/lgbm/lgbm_screened_no_winsor_neutral_tminus1_similar60_pathfull360_cold_v1_20260727_config.json5', start='2025-10-30', end='2026-07-23')"
py -c "from cbond_on.infra.model.runners.train_lgbm import main; main(config_path=r'cbond_on/config/models/lgbm/lgbm_screened_no_winsor_neutral_tminus1_latest60_pathfull360_cold_v1_20260727_config.json5', start='2025-10-30', end='2026-07-23')"
py -m cbond_on.cli.strategy_backtest --config backtest_pipeline/backtest_lgbm_neutral_tminus1_similar60_pathfull360_cold_20251030_20260723
py -m cbond_on.cli.strategy_backtest --config backtest_pipeline/backtest_lgbm_neutral_tminus1_latest60_pathfull360_cold_20251030_20260723
```

## Results

| Metric | Similar60 | Latest60 |
| --- | ---: | ---: |
| Strategy total return | 15.99% | 4.37% |
| Strategy Sharpe | 1.653 | 0.528 |
| Max drawdown | -7.09% | -9.48% |
| Excess return | 10.84% | -0.27% |
| Excess Sharpe | 1.791 | -0.008 |
| Average turnover | 78.30% | 77.78% |
| Model daily rank-IC mean | 0.00533 | -0.00598 |
| Top20 overlap | 26.68% | -- |

- Paired daily Similar60-minus-Latest60 mean return was 6.01 bp (176 common days; t=1.863, two-sided p=0.064).
- Similar60 beat Latest60 on 96 days, lost on 76 and tied on the four state-gap fallback days.
- After removing the five best relative days, Similar60 still returned 10.91% versus Latest60 5.66%; after removing the ten best relative days the gap was only 0.74%, so the effect has material tail-day concentration.
- Similar60's selected train days had median calendar age 162 days; 77.2% were more than 60 calendar days old. It is a genuinely different sample set, not a short-window proxy.

## Artifacts

- Comparison summary: `D:\cbond_on\results\experiments\similar60_pathfull360_20260727\comparison_summary.json`
- Monthly and period comparison: `comparison_by_month.csv`, `comparison_by_period.csv`, `comparison_daily_returns.csv` in the same root.
- Similar60 model audit: `D:\cbond_on\results\models\lgbm_screened_no_winsor_neutral_tminus1_similar60_pathfull360_cold_v1_20260727\2025-10-30_2026-07-23\20260727_201354\rolling_similar_days.csv`
- Similar60 backtest: `D:\cbond_on\results\backtest\2025-10-30_2026-07-23\Research_Similar60_pathfull360_cold_20251030_20260723\20260727_202908\summary_metrics.json`
- Latest60 backtest: `D:\cbond_on\results\backtest\2025-10-30_2026-07-23\Research_Latest60_pathfull360_cold_20251030_20260723\20260727_203025\summary_metrics.json`
- State repair manifest: `D:\cbond_on\results\experiments\similar60_pathfull360_20260727\state\state_repair_manifest.json`

## Caveats

- Four state days (`2026-05-12`, `2026-05-15`, `2026-05-19`, `2026-06-17`) cannot be reconstructed from clean data and used the same explicit 41/18 rolling fallback in both variants.
- Two dates (`2026-06-11`, `2026-06-12`) had no model score and were skipped by both strict backtests.
- The state CSV is built with data through 14:30 whereas the current live factor chain has a 14:29 cutoff. This is an offline research result, not a live-ready model.
- This is a mechanism comparison with Latest60, not a fresh aligned comparison against the current HL20/Regsim production family. Do not promote or replace the live champion based on this result.

## Promotion Status

- Research-only candidate. A point-in-time 14:29 state rebuild, out-of-sample confirmation, and a separately isolated current-production comparison are required before any live-scope decision.
