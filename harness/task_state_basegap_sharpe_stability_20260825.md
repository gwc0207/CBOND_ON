# Task State: BaseGap Robust-Sharpe Stability Scorecard (2026-08-25)

## Objective

- Replace historical selection-return tuning as a research objective with
  robust cross-block Sharpe and stable relative value versus fixed Regsim.

## Risk Level

- Medium research risk; live risk controlled by a separate research root and
  no protected live path writes.

## Current Verified Facts

- The source is the completed r3 51-point frozen BaseGap replay with 1,648
  verified frozen predecessor files.
- All variants are ranked on the same 219 common active design dates.
- No variant passes the fixed absolute-Sharpe, relative-Regsim, block-stability,
  and block-bootstrap gates.
- The resulting recommendation is to keep fixed Regsim as the research
  baseline; this is not a live change.

## Files Changed

- `harness/tools/model_switch_basegap_sharpe_stability_replay.py`
- `tests/test_model_switch_basegap_sharpe_stability_replay.py`
- `docs/experiment_records/model_switch_basegap_sharpe_stability_20260825.md`
- This task-state file.

## Commands Run

- `py -3.11 -m pytest -q tests/test_model_switch_basegap_sharpe_stability_replay.py tests/test_model_switch_basegap_tuning_replay.py tests/test_live_model_switch.py`
- `py -3.11 harness/tools/model_switch_basegap_sharpe_stability_replay.py --run-name run_sharpe_stability_v1_r2_20260825`

## Artifacts

- `D:/cbond_on/research_scratch/model_switch_basegap_sharpe_stability_20260825/run_sharpe_stability_v1_r2_20260825/`

## Open Risks

- The scorecard is retrospective calibration, not a new independent OOS test.
- The source state remains unproven strict-14:29 and source prices are frozen
  historical artifacts rather than a versioned production valuation ledger.

## Next Action

- Do not retune on the same historical family.  Freeze the scorecard contract
  and validate against fixed Regsim in a new immutable-input forward-shadow
  window before considering a live change.
