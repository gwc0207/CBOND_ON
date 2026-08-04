# Task State

## Objective

- Run an isolated causal label/execution-lag experiment for the current Regsim
  LGBM contract.  The primary variant uses factor day `T` to predict the
  complete strict return from buying on `T+1` and selling on `T+2`.

## Risk Level

- medium: research-only changes touch the shared generic LGBM/backtest paths,
  but their default (`lag=0`) must remain unchanged.  No live config, DB,
  scheduler, Champion, live score root, or live model-state root may be used.

## Current Verified Facts

- Standard label file `L[D]` means buy `D 14:42--14:57`, then sell on the
  next trading day `09:30--09:39`.
- A lag-one sample is `factor[T] -> L[next_trading_day(T)]`; scoring at `T`
  may train only through feature day `T-2`, because `L[T]` has not begun its
  buy leg at the 14:29 decision cutoff.
- Current rolling Regsim uses 60 calendar-trading slots and 59 labelled
  history slots.  The lag-one variant will use 61 slots so it retains the
  same 59 usable historical feature days plus a one-day embargo.
- The available data supports a complete final lag-one score day of
  2026-07-28 (buy 2026-07-29, sell 2026-07-30).
- The full lag-one score run completed with 179 score days
  (`2025-10-30..2026-07-28`) and 179 matching alignment rows. All rows pass
  the feature/label embargo check; no live path was used.
- Each final backtest has 179 actual days. The lagged arms skip execution
  days 2026-06-12 and 2026-06-15 because score dates 2026-06-11 and
  2026-06-12 have factor files lacking required columns; the same-day control
  instead skips 2026-06-11 and 2026-06-12. Pairing against the control uses
  their exact 178-day intersection, without filling either gap.
- Final research result: full label+execution lag-one is 18.935% total return
  and 1.862 Sharpe, versus 17.851% / 1.805 for same-day current Regsim. The
  paired difference versus the current control is not statistically
  significant (0.132bp/day, t=0.040); it is not promotable.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/context/source_of_truth.md`
- `docs/experiment_records/` recent 2026-07-27--2026-07-30 records
- `cbond_on/infra/data/panel.py`
- `cbond_on/infra/model/impl/lgbm/trainer.py`
- `cbond_on/infra/model/runners/train_lgbm.py`
- `cbond_on/app/usecases/backtest_runtime.py`

## Files Changed

- Default-preserving lag-one support in the LGBM trainer/rolling runner and
  in the offline backtest runtime.
- Isolated lag-one model/backtest configs and focused tests.
- `cbond_on/app/usecases/backtest_runtime.py` also restricts IC cycle inputs
  to buy-leg fields; all three final runs were regenerated after this fix.
- `docs/experiment_records/lgbm_label_execution_lag1_20260730.md` records
  the contract, evidence, metrics, and limitations.

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- `py -m pytest tests/test_lgbm_label_lag.py tests/test_backtest_execution_lag.py`
  -> `6 passed`.
- `py -m cbond_on.common.architecture_guard` -> `architecture guard: ok`.
- Full isolated lag-one score generation plus the three final backtests
  (`current control`, `execution only`, and `full lag-one`).

## Artifacts

- Isolated root:
  `D:/cbond_on/results/experiments/label_execution_lag1_20260730/`
- Final score artifact:
  `artifacts/models/research_regsim_label_execution_lag1_20260730/2025-10-30_2026-07-28/20260730_204831/`
- Final full lag-one backtest:
  `backtest/2025-10-31_2026-07-29/Research_LabelExecutionLag1_RegsimFull_20260730/20260730_210229/`

## Open Risks

- Historical T1430 factor files are marked 14:30 rather than independently
  certified as-of 14:29.  This experiment cannot repair that pre-existing
  promotion boundary.
- A delayed execution must freeze the signal-day ranking and allowlist;
  using a later pool to re-rank would introduce a different operational
  strategy and must not be conflated with the primary result.

## Next Action

- Keep this result research-only. Do not modify live configuration, model
  routing, Champion, DB, or scheduler from it. A separately predeclared
  prospective shadow or untouched later period is required before considering
  a live decision.

## Handoff Summary

- Completed research-only lag-one experiment. No live mutation occurred; the
  result is non-promotable pending independent evidence.
