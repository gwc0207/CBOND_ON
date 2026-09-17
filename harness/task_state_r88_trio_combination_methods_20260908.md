# Task State: R88 Trio Combination Methods

## Objective

- Compare pre-registered combination methods for frozen `P6 + Full88 P3 + Regsim`
  under the current DataHub execution contract.
- Keep single-Top20 score fusion and three-sleeve allocation as separate
  research contracts.

## Risk Level

- Medium. Research-only model-combination work; live configuration, model state,
  factors, production DB, scheduler, and production outputs are out of scope.

## Current Verified Facts

- B2 is the accepted current-data baseline: all four B2 comparison series have
  399 common dates from 2025-01-02 through 2026-08-27.
- P6/P3/Regsim scores are frozen inputs. The required score policy is own-score-
  universe percentile rank, unchanged `o_0005` target universe, and neutral
  `0.5` contribution for an absent score.
- Current execution contract is Top20, 5% per name, full turnover, buy
  `twap_1442_1457`, next-day sell `twap_0930_0939`, buy cost 1.0bp, sell cost
  1.2bp, and the shared strict benchmark.
- The 403 current DataHub source dates needed by B2 have valid raw/clean/publish
  gates. The completed as-run audit records 1,202 raw input files, 1,209
  publish-evidence files, and digest
  `e06605a6bfdd2d3dd9cfd175f840a1ef754a2418a94dd6e96ef5505544b3aea5`.
- 2025 is development; 2026 is already observed reporting only and will never
  be used to choose a method or hyperparameter.
- All 27 single-Top20 score-fusion methods and all eight sleeve-only controls
  completed with 399 aligned dates. No live, DB, scheduler, training, or
  re-scoring call occurred.
- The fixed equal-rank B2 baseline remains the defensible score-fusion result.
  Hedge and CVaR improve development metrics slightly but lose that increment
  in the already-observed 2026 period; their full-period deltas versus equal
  rank are `+0.024bp/day (p=0.903)` and `+0.094bp/day (p=0.913)`.
- Static score selection is not robust: 2025 CSCV PBO is `50.0%`, with selected
  OOS median rank `9.5/17`. The full score ledger has 31 method labels but 28
  unique daily-return sequences; DSR, RC/SPA, and MCS were re-run on that
  de-duplicated formal family and do not establish a method superior to the
  equal-rank baseline or Regsim.
- The strongly regularised LGBM allocator predicts the same value for all three
  experts on every post-warmup day and therefore remains exactly equal weight;
  it is a valid conservative null result, not an improvement.

## Confirmed Experiment Matrix

- Score fusion: equal rank baseline, static simplex grid, normal-score fusion,
  cross-sectional Ridge and NNLS stacking, fixed-share Hedge, Bayesian expert
  weighting, EWMA/MVO-style expert weighting, CVaR/CDaR/downside score
  weighting, and a strongly regularized rolling LGBM meta-allocator.
- Sleeve-only controls: equal weight, GMV, shrinkage MVO, ERC, CVaR, CDaR,
  robust MVO, and Bayesian shrinkage allocation. These are not presented as a
  single Top20 strategy.
- Validation: rolling OOS ledger, Sharpe/alpha/risk stability, PSR/DSR,
  CSCV/PBO for static family, White RC/Hansen SPA, MCS, beta/residual alpha,
  and block-bootstrap diagnostics.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/context/source_of_truth.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `docs/experiment_records/r88_trio_score_fusion_plan_20260908.md`
- `docs/experiment_records/model_switch_score_level_fusion_20260811.md`
- `docs/experiment_records/model_switch_dynamic_weight_optimization_20260728.md`
- `docs/experiment_records/model_switch_downside_risk_fusion_20260811.md`

## Files Changed

- `harness/tools/r88_trio_combination_methods.py`
- `tests/test_r88_trio_combination_methods.py`
- `harness/task_state_r88_trio_combination_methods_20260908.md`
- `docs/experiment_records/r88_trio_combination_methods_20260908.md`

## Commands Run

- `py -3 -B harness/tools/agent_preflight.py --mode research-experiment`
- `py -3 -m pytest tests/test_r88_trio_score_fusion_b1.py
  tests/test_r88_trio_score_fusion_b2_current_data.py
  tests/test_r88_trio_combination_methods.py -q` -> `11 passed`.
- `py -3 -m ruff check harness/tools/r88_trio_combination_methods.py
  tests/test_r88_trio_combination_methods.py` -> passed.
- Full research run: 27 generic strict Top20 replays, eight sleeve controls,
  and family validation completed under the dedicated root.

## Artifacts

- Completed root:
  `D:/cbond_on/research_scratch/r88_trio_combination_methods_20260908_r1/`
  - `full_methods_20260908_r1/as_run_input_audit/`
  - `full_methods_20260908_r1/score_fusion/`
  - `full_methods_20260908_r1/sleeve_allocation/`
  - `full_methods_20260908_r1/validation/`
  - `full_methods_20260908_r1/RESULTS.md`

## Open Risks

- This is an as-run current-data baseline, not an immutable historical replay.
- Weight-to-Top20-to-return is non-linear; only a full generic backtest proves
  a score-fusion result.
- Current three-strategy returns are highly correlated, so sleeve diversification
  claims require special scrutiny.
- 2026 was necessarily inspected to report this completed comparison. It cannot
  be reused to tune a revised method, weight grid, or LGBM regularisation.
- Sleeve allocation is a separate multi-book contract. It has not replayed
  merged holdings, netting, or portfolio-layer execution costs and cannot be
  proposed as a single-Top20 live replacement.

## Next Action

- No further selection or live action is authorized. Any follow-up must either
  define a new pre-registered method family using 2025 only or freeze a method
  now and collect fresh forward shadow evidence.

## Handoff Summary

- Owner-confirmed full comparison completed on 2026-09-08. The result supports
  retaining equal-rank fusion as the research baseline, not replacing it with a
  newly selected dynamic rule. No live action is authorized by this task.
