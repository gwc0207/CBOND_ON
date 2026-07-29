# Task State: Robust Ridge Input Rebuild Replay (2026-07-28)

## Objective

- In a strictly offline replay, compare the current 44-feature Robust Ridge
  diagnostic against a seven-feature afternoon-dispersion Ridge and a rolling
  robust-scaled 44-feature PCA-15-whitened Ridge, including one pre-specified
  PCA-15 variant with causal model-disagreement features.  Preserve the
  current Base, Champion protection, and all non-Robust routing.

## Risk Level

- Medium research-only: the inputs originate from the live selector, but this
  task neither imports the live runtime nor writes DB, config, scheduler, or
  live artifacts.

## Current Verified Facts

- The replay baseline is `daily_current.csv` from the strict-veto experiment:
  538 aligned score days from 2024-05-08 through 2026-07-27.
- The current Robust contract is 360 prior days, minimum 120 days, pairwise
  Ridge `alpha=100`, clipped pairwise return target +/-75bp, and a 5bp
  top-two utility margin.
- The actual replaceable branch has 239 dates: `Base=margin_default` and a
  current numeric Robust diagnostic.
- Existing shadow return histories and the T1430 state CSV are read-only
  inputs.  Output is confined to `D:/cbond_on/results/experiments/`.
- All 538 aligned score days have archived Regsim/Ensemble/HL20 score CSVs;
  537 have the fixed previous-score-day Top20-overlap feature (the first day
  has no preceding aligned score day).
- `path44_standard_ridge` exactly reconstructs all 538 current decisions and
  all 239 eligible branch decisions.
- The three alternative input variants all lose to current in the aligned
  shadow-return replay: disp7 `-2.26pp`, robust-PCA15 `-1.08pp`, and
  PCA15+disagreement `-4.32pp` cumulative return differences.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/context/source_of_truth.md`
- `docs/experiment_records/model_switch_robust_strict_veto_20260728.md`
- `docs/experiment_records/model_switch_similarity_robust_20260728.md`
- `harness/tools/similarity_robust_replay.py`

## Files Changed

- `harness/task_state_ridge_robust_rebuild_20260728.md`
- `harness/tools/ridge_robust_replay.py`
- `docs/experiment_records/model_switch_ridge_rebuild_20260728.md`

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- `py -m py_compile harness/tools/ridge_robust_replay.py`
- `py harness/tools/ridge_robust_replay.py --strict-current-validation`

## Artifacts

- Planned root: `D:/cbond_on/results/experiments/model_switch_ridge_rebuild_20260728/`
- Completed run: `D:/cbond_on/results/experiments/model_switch_ridge_rebuild_20260728/run_20260728_214951/`

## Open Risks

- Shadow-return composition does not carry continuous positions or incremental
  switching costs, so it is not a live-promotion result.

## Next Action

- Keep every alternative research-only.  Do not tune their inputs on this
  same sample or modify live Robust from this result.

## Handoff Summary

- The isolated replay and experiment record are complete.  No live path,
  protected configuration, database, scheduler, or existing source file has
  been changed by this experiment.
