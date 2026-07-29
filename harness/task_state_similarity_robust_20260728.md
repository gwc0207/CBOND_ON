# Task State: Similarity-Robust Selector Experiment (2026-07-28)

## Objective

- Test whether the current Robust Ridge pairwise predictor can be replaced, within the existing live model-switch architecture, by a long-window non-parametric similarity selector: historical state Top-K days followed by model-return statistics.

## Risk Level

- Medium: selector research only. No live configuration, scheduler, model state, DB, or production output change is allowed.

## Current Verified Facts

- Current Base is already a short-window KNN selector: 60 prior days, 40 nearest days, equal-weight `trim20_lcb10` across Regsim, Ensemble, and HL20.
- Current Robust uses 360 prior days and three pairwise Ridge regressions; its individual pairwise OOS signal is not established.
- The experiment retains Base, Champion-first, Champion-third, low-confidence routing, model return histories, and all trading semantics. It replaces only the long-history Robust diagnostic in the offline replay.
- The actual replacement branch has 239 dates: Base is low-confidence and the existing Robust has a numeric diagnostic. The 59-day count is only the subset where the existing Ridge is itself high-confidence.
- The pre-specified K40 Mahalanobis kernel variant loses to current. Two K60 sensitivity variants have small full-sample gains, but neither passes conditional significance, time-stability, or nine-variant selection adjustment.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/context/source_of_truth.md`
- `cbond_on/config/live/live_config.json5`
- `cbond_on/infra/live/model_switch.py`
- `docs/experiment_records/model_switch_dynamic_weight_optimization_20260728.md`
- `docs/experiment_records/soft360pool_pathfull360_ess60_20260727.md`

## Files Changed

- `harness/task_state_similarity_robust_20260728.md`
- `harness/tools/similarity_robust_replay.py`
- `docs/experiment_records/model_switch_similarity_robust_20260728.md`

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- `py -m py_compile harness/tools/similarity_robust_replay.py`
- `py harness/tools/similarity_robust_replay.py`

## Artifacts

- `D:/cbond_on/results/analysis/model_switch_similarity_robust_20260728/run_20260728_210412/`

## Open Risks

- The long-window KNN selector remains related to Base; apparent gains can be selection noise rather than independent evidence.
- Shadow-return composition does not yet transmit continuous positions or incremental switch costs.
- Prior Soft360Pool evidence warns that soft similarity weights are not automatically superior to hard Top-K selection.

## Next Action

- Keep all Similarity-Robust variants out of live configuration. Do not continue tuning the K60 sensitivity winner on the same 538-day sample; require an independently approved time holdout and continuous-position replay before any future reconsideration.
