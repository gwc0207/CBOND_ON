# Experiment Record: MT-015 Soft360Pool path-full kernel weighting

## Experiment ID

- `MT-015`

## Question

- Does continuously weighting historical days by full T1430 market-state similarity outperform the Hard Similar60 sample-selection mechanism while retaining an approximately 60-day effective state sample?

## Comparison Contract

- Window: `2025-10-30` to `2026-07-23`; 176 common strict strategy-return days.
- Factor/label: current neutral T-1 base, T1430 factors, no HL20 time decay, no binary Regsim weight, daily cold-start and daily refit.
- Execution: buy `twap_1442_1457`, sell next-trading-day `twap_0930_0939`, buy/sell fees of 1.0 / 1.2 bp, complete cycle return.
- Universe: previous-trading-day `o_0005` only; Top20 with 5% per name.
- State source: the same repaired research-only `path_full_t1430` CSV used by MT-014.

## Baselines

- Hard Similar60: 360 effective state candidates, nearest 60 for training, ranks 61--80 for validation.
- Latest60: the identical 60/20 training contract, using latest eligible dates instead of nearest state dates.

## Variant

- Model/config: `models/lgbm/lgbm_screened_no_winsor_neutral_tminus1_soft360pool_pathfull360_ess60_cold_v1_20260727`.
- Candidate state pool: most recent 360 complete and trainable historical days; candidate-only z-score and 44-dimensional Euclidean `path_full_t1430` distance.
- Validation: distance ranks 61--80, identical to Hard Similar60.
- Fit set: the remaining 340 candidate days. Each day receives a Gaussian kernel multiplier
  `exp(-0.5 * (distance / bandwidth)^2)`.
- Bandwidth: solved causally per target day by bisection to a kernel day-level effective sample size of 60.
- Row contract: weights multiply the existing equal-row training contract; final row-mass effective day count is recorded separately rather than silently changing to day-balanced training.

## Commands

```powershell
py -c "from cbond_on.infra.model.runners.train_lgbm import main; main(config_path=r'cbond_on/config/models/lgbm/lgbm_screened_no_winsor_neutral_tminus1_soft360pool_pathfull360_ess60_cold_v1_20260727_config.json5', start='2025-10-30', end='2026-07-23', execution={'train_processes': 1, 'prep_workers': 1, 'prefetch_windows': 1})"
py -m cbond_on.cli.strategy_backtest --config backtest_pipeline/backtest_lgbm_neutral_tminus1_soft360pool_pathfull360_ess60_cold_20251030_20260723
```

## Results

| Metric | Soft360Pool | Hard Similar60 | Latest60 |
| --- | ---: | ---: | ---: |
| Strategy total return | 8.11% | 15.99% | 4.37% |
| Strategy Sharpe | 0.989 | 1.653 | 0.528 |
| Max drawdown | -7.26% | -7.09% | -9.48% |
| Excess return | 3.22% | 10.84% | -0.27% |
| Excess Sharpe | 0.615 | 1.791 | -0.008 |
| Average turnover | 80.60% | 78.30% | 77.78% |
| Model daily rank-IC mean | 0.00322 | 0.00533 | -0.00598 |

- Soft360Pool minus Hard Similar60: -4.07 bp/day over 176 common days (79 wins, 93 losses, 4 ties; paired t=-1.129, two-sided p=0.261).
- Soft360Pool minus Latest60: +1.95 bp/day (82 wins, 90 losses, 4 ties; paired t=0.518, two-sided p=0.605).
- Soft360Pool shares 34.03% of Hard Similar60 Top20 names on average (6.81 names); it is materially different, but not a better version of Hard Similar60.
- The results reject the first hypothesis that smoothing all candidate days improves this mechanism. The Hard60 boundary carries useful information that the Gaussian tails dilute.

## Audit and Runtime

- 176 score days: 172 normal kernel days and four explicit state-gap rolling fallbacks (`2026-05-12`, `2026-05-15`, `2026-05-19`, `2026-06-17`).
- Two common no-score dates (`2026-06-11`, `2026-06-12`) were skipped by all three strict backtests.
- `rolling_similar_days.csv` has 61,924 audit rows. Every selected day is strictly earlier than its target day.
- Normal kernel days have raw day ESS of 60.0 by construction. After retaining equal-row model weighting, final day ESS has median 60.82, IQR 57.45--64.17, and range 50.52--97.40. Median maximum single-day mass is 5.29%; the maximum is 11.61%.
- Full cold walk-forward score run: approximately 23m03s (`22:46:41` to `23:09:44`), excluding the preliminary four-day smoke. This is materially more expensive than Hard Similar60 and not live-feasible without a separate runtime design.

## Artifacts

- Full model audit: `D:\cbond_on\results\models\lgbm_screened_no_winsor_neutral_tminus1_soft360pool_pathfull360_ess60_cold_v1_20260727\2025-10-30_2026-07-23\20260727_224654`
- Score root: `D:\cbond_on\results\experiments\soft360pool_pathfull360_ess60_20260727\scores\soft360pool_ess60_cold`
- Strict backtest: `D:\cbond_on\results\backtest\2025-10-30_2026-07-23\Research_Soft360Pool_pathfull360_ess60_cold_20251030_20260723\20260727_231254`
- Backtest plot: `backtest_report.png` under the strict backtest directory.
- Background stdout/stderr: `D:\cbond_on\results\experiments\soft360pool_pathfull360_ess60_20260727\soft360_full_20260727_retry1_stdout.log` and `soft360_full_20260727_retry1_stderr.log`.

## Caveats and Promotion Status

- The state CSV includes information through 14:30 whereas the live factor chain uses a 14:29 cutoff. This remains an offline mechanism study, not a live-ready candidate.
- This is not a fresh, isolated comparison with current production Regsim; it must not replace the Champion.
- Soft360Pool loses to the aligned Hard Similar60 mechanism and costs substantially more. Do not promote it or combine it with other weight schemes based on this result.
