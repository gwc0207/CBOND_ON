# Task State

## Objective

- Implement and evaluate a research-only, causal two-contrast PyTorch LSTM
  for model-switch arbitration, with a predeclared small chronological grid.

## Risk Level

- medium: research-only code and experiment outputs; live routing and runtime
  boundaries are protected.

## Current Verified Facts

- Repository root is `C:\Users\BaiYang\CBOND_ON\cbond_on` at `b65a3b4`.
- The frozen input snapshot contains 538 return days and 532 complete
  state/return pairs.  The six missing-state days are all in the final holdout.
- Exact immutable split: train 2024-05-08..2025-10-15 (350 complete days),
  validation 2025-10-16..2026-02-09 (81), holdout
  2026-02-10..2026-07-27 (101 complete of 107 return days).
- The previous temporal replay exactly reproduced production Base on 538/538
  score days.  Its state provenance remains `14:30_not_strict_1429_certified`.
- PyTorch is available.  No live configuration, DB, scheduler, model state,
  `results/live`, or `results/analysis` may be changed.

## Files Read

- `AGENTS.md`, `harness/README.md`, research skill/workflow and source of truth.
- `harness/model_switch_temporal_arbitration.py`.
- `harness/tools/model_switch_temporal_arbitration_replay.py`.
- `tests/test_model_switch_temporal_arbitration.py`.
- `docs/experiment_records/model_switch_temporal_arbitration_20260729.md`.

## Files Changed

- `harness/model_switch_lstm_arbitration.py`.
- `harness/tools/model_switch_lstm_arbitration_replay.py`.
- `tests/test_model_switch_lstm_arbitration.py`.
- `docs/experiment_records/model_switch_lstm_arbitration_20260730.md`.
- This task-state file.

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`.
- Read-only Git, frozen-input, and dependency inspection.
- `py -m pytest -q tests/test_model_switch_lstm_arbitration.py` -> `9 passed`.
- `py harness/tools/model_switch_lstm_arbitration_replay.py --run-name run_20260730T_lstm_path44_v1`.
- `py -m pytest -q tests/test_model_switch_lstm_arbitration.py tests/test_model_switch_temporal_arbitration.py tests/test_live_model_switch.py` -> `37 passed`.
- Exact deterministic reproduction with run name
  `run_20260730T_lstm_path44_v1_reprocheck`; result CSVs and all checkpoints
  have matching SHA-256 values.

## Artifacts

- Canonical result:
  `D:/cbond_on/results/experiments/model_switch_lstm_20260730/run_20260730T_lstm_path44_v1/`.
- Reproducibility result:
  `D:/cbond_on/results/experiments/model_switch_lstm_20260730/run_20260730T_lstm_path44_v1_reprocheck/`.
- Selected validation trial: `lstm_seq40_hidden8_drop0p1`, validation
  two-contrast RMSE `20.7287bp`, epoch count `26`.
- Final holdout selected LSTM: `+16.5700%`, Sharpe `2.6269`, zero LCB
  overrides; exactly equal to strict Base fallback and below frozen current
  routing (`+17.0029%`, Sharpe `2.6313`).

## Open Risks

- The historical state provenance is not strict 14:29 PIT certified.
- A small validation set and fixed eight-trial grid can still overfit; final
  holdout was excluded from all trial selection and epoch selection.
- A 40-day required state sequence produced only 55 ready final-holdout days;
  missing state rows from 2026-05-12 prevent sequence recovery before the end.
- The user has not authorized promotion of any result to live.

## Next Action

- Report the negative research result and await owner direction.  Do not tune
  on the observed final holdout or make a live change.

## Handoff Summary

- Research implementation and verification are complete.  The LSTM did not
  earn an override or improve final-holdout return/Sharpe.  Live mutation
  remains unauthorized.
