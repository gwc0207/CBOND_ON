# Task State: R88 CVaR Hyperparameter Study

## Objective

- Tune the research-only CVaR score-fusion rule for frozen P6, P3, and Regsim scores.
- Evaluate a 54-point raw-relative-return grid and a 16-point residual-alpha grid through real Top20 backtests.

## Risk Level

- Medium. Live configuration, model state, factors, production DB, scheduler, production scores, and production outputs are out of scope.

## Locked Contract

- Common current-data period: 399 dates, 2025-01-02 through 2026-08-27.
- Frozen P6/P3/Regsim scores; no retraining or scoring.
- Existing o_0005 universe, own-universe percentile rank, neutral missing score rank 0.5, Top20, 5 percent name cap, full turnover, fees, TWAP, benchmark, and market mask.
- Raw grid: lookback {60,120,180}, tail {5%,10%}, lambda {0,0.75,1.5}, equal shrinkage {25%,50%,75%}: 54 labels.
- Residual-alpha grid: lookback {120,180}, tail {5%,10%}, lambda {0.75,1.5}, equal shrinkage {50%,75%}: 16 labels.
- All signals at date t use only returns strictly before t. 2025 is development; 2026 is observed reporting only.

## Current Verified Facts

- Existing fixed CVaR point is lookback 120, tail 5%, lambda 0.75, equal shrinkage 50%, and relative-to-benchmark input. It was not a hyperparameter search.
- The completed full-method study has an as-run current-data audit covering 403 source dates / 1,202 raw files / 1,209 publish evidence files. Its input digest will be independently rechecked in this study.
- Full generic score fusion is non-linear through score rank and Top20. No sleeve-return proxy is accepted as score-fusion evidence.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/context/source_of_truth.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `docs/experiment_records/r88_trio_combination_methods_20260908.md`
- `harness/tools/r88_trio_combination_methods.py`

## Files Changed

- `harness/tools/r88_cvar_hyperopt.py`
- `tests/test_r88_cvar_hyperopt.py`
- `docs/experiment_records/r88_cvar_hyperopt_20260909.md`
- This task-state record.

## Commands Run

- `py -3 -B harness/tools/agent_preflight.py --mode research-experiment`
- `py -3 -X utf8 -B harness/tools/r88_cvar_hyperopt.py --stage all --resume`
- `py -3 -X utf8 -m pytest tests/test_r88_cvar_hyperopt.py -q` (`6 passed`)

## Artifacts

- Research root: `D:/cbond_on/research_scratch/r88_cvar_hyperopt_20260909_r1/full_grid_20260909_r1/`
- Parameter ledger: `grid/parameter_registry.csv` (70 labels, 61 canonical weight sequences).
- Complete aligned return ledger: `grid/aligned_daily_returns.csv` (29,526 rows = 74 methods x 399 dates).
- Formal validation: `validation/cvar_hyperopt_union/`.
- Human-readable result: `RESULTS.md`.

## Final Result

- Completed 2026-09-09 22:35:30 CST after 1:58:17. The grid contains 60 newly executed strict Top20 backtests and one fixed-CVaR output reused only after exact 399-day score/weight parity verification.
- Both pre-grid and pre-validation input gates passed: 2,411 raw/manifest files checked each time, zero mismatches.
- The best 2025 point estimate is raw `lookback=180`, `tail=5%`, `lambda=1.5`, `shrinkage=25%`: cumulative return 50.00%, Sharpe 4.386, HAC alpha t 4.476, MDD -6.67%. Equal-rank is 46.64%, 4.122, 3.991, and -6.44%, respectively.
- This is not a validated upgrade. Its unadjusted paired uplift versus equal-rank is +0.931 bp/day (HAC p=0.221); family White RC p=0.674 and Hansen SPA-consistent p=0.603. The 10% MCS retains all 65 exact-return sequences.
- CSCV/PBO is 25.4% with selected OOS median rank 17/65. This is a diagnostic only, not forward OOS confirmation.
- 2026 is reporting-only: the 2025 point-estimate winner has 33.66% cumulative return / Sharpe 3.671 versus equal-rank 34.71% / 3.920. It cannot be used to retune the grid, but it lowers confidence in the development-period uplift.
- Residual-alpha CVaR does not improve the baseline in this matrix. The best residual configuration has 2025 Sharpe 4.120 versus equal-rank 4.122.

## Open Risks

- 2026 has already been observed and cannot become a clean validation set.
- A large grid can overfit 2025. This run did not establish a family-level improvement; any future candidate requires a frozen configuration and fresh forward shadow.
- Current DataHub inputs are as-run, not immutable historical bytes.
- The generated `RESULTS.md` embeds a stale `status=running` snapshot due to report-write ordering. The authoritative `run_status.json` is `completed`; this is metadata-only and did not affect calculation or validation.

## Next Action

- Keep equal-rank as the defensible research baseline. Do not change live from this result.
- If a CVaR rule is still desired, freeze a pre-declared candidate and collect fresh forward-shadow evidence; do not use the already observed 2026 period to select another grid point.

## Handoff Summary

- Owner confirmed the 54 + 16 pre-registered CVaR matrix on 2026-09-09. Research completed without live action; no live action is authorized by this experiment.
