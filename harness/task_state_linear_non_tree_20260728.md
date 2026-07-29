# Task State

## Objective

- Run an isolated, causal T1430 cross-sectional linear-model experiment
  (Ridge, ElasticNet, and Huber) with a small hyperparameter grid.  Do not
  change live configuration, model state, scheduling, database outputs, or
  Champion selection.

## Risk Level

- medium: research-only code-path repair plus backtests; no live artifact may
  be reused or overwritten.

## Current Verified Facts

- The current linear scorer derives target days from label files and joins the
  target label before scoring, so it cannot score a live day without a label.
- `LinearAdapter` currently discards `label_cutoff`.
- The current 27-factor T1430 contract is available through 2026-07-27;
  labels are available through 2026-07-24.
- The experiment contract is 60 labelled trading-day lookback, daily refit,
  T1430 factors, 14:42 buy label, next-day 09:30--09:39 sell, Top20, and the
  existing strict backtest/allowlist configuration.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/context/source_of_truth.md`
- `cbond_on/infra/model/impl/linear/linear_score.py`
- `cbond_on/infra/model/runners/train_linear.py`
- `cbond_on/infra/model/adapters.py`

## Files Changed

- `cbond_on/infra/model/impl/linear/linear_score.py`
- `cbond_on/infra/model/runners/train_linear.py`
- `cbond_on/infra/model/adapters.py`
- `cbond_on/app/usecases/backtest_runtime.py`
- isolated linear and backtest configs under `cbond_on/config/`
- `tests/test_linear_causal_score.py`
- `tests/test_backtest_output_root.py`
- experiment records.

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- causal unit tests and strict-cycle regression test: `11 passed`
- `py -m cbond_on.common.architecture_guard` (`ok`)
- nine model-score runs over 2025-10-30..2026-07-23
- nine isolated strict backtests plus read-only HL20/Regsim/Ensemble baselines.

## Artifacts

- `D:/cbond_on/results/experiments/linear_non_tree_grid_20260728/`
- `docs/experiment_records/linear_non_tree_grid_20260728.md`

## Open Risks

- Historical T1430 is marked 14:30 while live input cutoff is 14:29; these
  results are not deployable without a separate as-of-14:29 rebuild.
- The preselected ElasticNet and every other linear variant have negative
  holdout Sharpe; no linear candidate is eligible for promotion.

## Next Action

- Do not promote this family.  If revisited, use a longer locked holdout or
  rolling multi-fold selection and a separately rebuilt 14:29 feature panel.

## Handoff Summary

- Research is complete and isolated.  No live model/config/state/DB/scheduler
  artifact was modified.
