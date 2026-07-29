# Task State

## Objective

- Build and evaluate a research-only LGBM that trains each score day on the 60 most similar historical T1430 market-state days from an effective 360-day pool.

## Risk Level

- medium: training-code change and long research run; no live, scheduler, database, or live-config change is in scope.

## Current Verified Facts

- Current Regsim is a coarse benchmark-regime weighting model, not a nearest-day training model.
- The state source has 44 `path_full_t1430` features and enough history for a 360-day candidate pool from 2025-10-30 onward.
- Existing rolling training has 59 history days split 41/18, so it cannot express a 60-day training set without a dedicated selection layer.

## Files Read

- `AGENTS.md`
- `harness/workflows/research_experiment.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/context/source_of_truth.md`
- current LGBM, live, and backtest configs

## Files Changed

- `cbond_on/infra/model/similar_day_training.py`
- `cbond_on/infra/model/runners/train_lgbm.py`
- research-only Similar60 / Latest60 model and backtest configs
- `tests/test_similar_day_training.py`

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- focused Similar60 tests and two real-data smoke runs (Similar60 and Latest60)

## Artifacts

- Planned research score root: `D:/cbond_on/results/experiments/similar60_pathfull360_20260727/scores/`
- Research-only repaired state copy and repair manifest under `D:/cbond_on/results/experiments/similar60_pathfull360_20260727/state/`
- Successful Similar60 smoke scores: 2026-07-21 through 2026-07-24, with 60 train + 20 validation days per target.
- Full Similar60 and Latest60 scores: 176 common days from 2025-10-30 through 2026-07-23.
- Strict aligned backtests and comparison artifacts under `D:/cbond_on/results/experiments/similar60_pathfull360_20260727/`.

## Open Risks

- The historical state CSV is constructed through 14:30 while the current factor chain uses a 14:29 cutoff; this run is offline research only and cannot be promoted without point-in-time alignment work.
- The shared state CSV has missing/incomplete target-day rows, so the research copy must record deterministic repairs from the same clean snapshots instead of silently falling back.
- Four target dates in the evaluation window remain incomplete after deterministic reconstruction from clean snapshots (`2026-05-12`, `2026-05-15`, `2026-05-19`, `2026-06-17`); their only permitted behavior in this trial is an explicit rolling fallback recorded in the manifest.
- Existing working-tree changes in the LGBM runner belong to the owner and must be preserved.

## Next Action

- No live action. Next research step is a 14:29 point-in-time state rebuild and an independently isolated production-family comparison.

## Handoff Summary

- Complete as a research-only attempt. Similar60 beat its Latest60 causal control, but the result is not live-promotable because of the state cutoff mismatch, four explicit fallbacks, and lack of an aligned current-production comparison.
