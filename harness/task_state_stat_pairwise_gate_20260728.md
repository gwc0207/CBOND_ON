# Task State: Frozen Statistical Pairwise Gate Replay (2026-07-28)

## Objective

- Test a research-only, non-predictive statistical abstention gate for the
  existing Base-low-confidence Robust branch: permit a challenger to override
  Base only when frozen historical pairwise evidence is sufficient.

## Risk Level

- Medium: model-switch research only.  No live configuration, scheduler, DB,
  model state, or production output may be changed.

## Current Verified Facts

- The current aligned selector replay has 538 score days and 239 eligible
  Base-low-confidence/valid-Robust branch days.
- Existing direct long-window similarity replacement was negative as a primary
  selector, so this test uses statistics as an abstention gate rather than a
  general chooser.
- All selection inputs must be strictly earlier than the score day.
- Frozen single rule: rolling-z-score + Ledoit-Wolf Mahalanobis on the
  18-feature trend/participation and 26-feature volatility/tail blocks, with
  `sqrt(0.5*d1^2 + 0.5*d2^2)` after per-block dimension normalization; K=60
  equal weights; pairwise `trim20 - winsor10/90 std / sqrt(60)` LCB > 5bp; and
  current Kth radius <= the 75th percentile of only prior ready Kth radii.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/context/source_of_truth.md`
- `harness/task_state_similarity_robust_20260728.md`
- `docs/experiment_records/model_switch_similarity_robust_20260728.md`
- `docs/experiment_records/model_switch_robust_strict_veto_20260728.md`
- `docs/experiment_records/model_switch_dynamic_weight_optimization_20260728.md`

## Files Changed

- `harness/tools/stat_pairwise_gate_replay.py`
- `harness/task_state_stat_pairwise_gate_20260728.md`
- `docs/experiment_records/model_switch_stat_pairwise_gate_20260728.md`

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- `py -m py_compile harness/tools/stat_pairwise_gate_replay.py`
- `py harness/tools/stat_pairwise_gate_replay.py`
- independent source-return, radius-reference, gate-condition, manifest-hash,
  and repeat-run verification via `py -`

## Artifacts

- Verified main run:
  `D:/cbond_on/results/experiments/model_switch_stat_pairwise_gate_20260728/run_20260728_214957/`.
- Retained identical first run:
  `D:/cbond_on/results/experiments/model_switch_stat_pairwise_gate_20260728/run_20260728_214839/`.

## Open Risks

- The replay composes standalone model shadow returns and does not carry
  continuous previous positions or true incremental switch costs.
- This is a fixed single-rule experiment, not evidence for live promotion.
- The gate covered only 2 of 239 replacement-branch days.  It improves the
  all-Base fallback by 0.49pp but remains 2.91pp below current replay, with
  coverage limited to the final chronological third.

## Next Action

- Keep the rule out of live configuration.  Do not scan thresholds or promote
  this result without a separately approved, independent holdout design.

## Handoff Summary

- The full replay and independent recomputation passed.  The result is
  negative versus the current selector and is recorded in the experiment
  record; protected live/config/DB/scheduler paths remain untouched.
