# Task State: Model-Switch Temporal Arbitration Research (2026-07-29)

## Objective

- In a strictly offline, research-only replay, evaluate pre-specified temporal
  model-selection candidates and a Base-versus-candidate cross-arbitration
  interface. The primary goal is to improve aligned out-of-sample selector
  return and Sharpe without changing the live model-switch contract.

## Risk Level

- Medium research-only. The task consumes live-derived state and shadow-return
  artifacts, but must not write live configuration, database records,
  scheduler state, model state, `results/live`, or `results/analysis`.

## Current Verified Facts

- Current live model-switch mode is `scoreopt_t1430_fusion_gate`; the current
  Robust gate uses a 5bp internal top-two utility gap.
- The aligned baseline spans 538 score days from 2024-05-08 to 2026-07-27.
- Current full-liquidation `turnover_ratio=1.0` makes the three archived
  strict shadow-return histories valid daily selector counterfactuals for this
  research contract.
- Current Robust pairwise Ridge has near-random rolling OOS diagnostics, so a
  new gate must compare a candidate directly with the Base winner rather than
  reuse the internal Robust top-two gap as an action rule.
- The state artifact has a known 14:29/14:30 PIT provenance risk. This replay
  must preserve and disclose the existing artifact; it cannot claim PIT repair.
- A byte-for-byte copy of the mutable state, three shadow-return histories,
  current routing, and relevant frozen configs was made under the approved
  experiment root. Source and copy SHA-256 values matched before and after
  copying.
- The read-only production Base implementation reproduced all `538/538`
  frozen `base_model_id`, `base_reason`, and `base_score_gap` values exactly.
- The frozen primary intervention envelope has `239` days. Existing current
  Robust overrides Base on `33` of them, with `19` wins, `14` losses, and
  `+4.2369bp` mean realised return relative to Base; this small sample remains
  an in-sample diagnostic, not a promotion claim.
- In the first pre-specified direct-LCB comparison, none of the six temporal
  candidates earned an override. Their strict selector therefore equals the
  Base fallback on all 538 days: total return `+144.0064%`, Sharpe `3.3776`,
  versus frozen current `+147.4008%`, Sharpe `3.4146`.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/context/source_of_truth.md`
- `docs/experiment_records/model_switch_ridge_rebuild_20260728.md`
- `docs/experiment_records/model_switch_dynamic_weight_optimization_20260728.md`
- `docs/experiment_records/model_switch_stat_pairwise_gate_20260728.md`
- `docs/experiment_records/model_switch_similarity_robust_20260728.md`

## Files Changed

- `harness/task_state_model_switch_temporal_arbitration_20260729.md`
- `harness/model_switch_temporal_arbitration.py`
- `harness/tools/model_switch_temporal_arbitration_replay.py`
- `tests/test_model_switch_temporal_arbitration.py`
- `docs/experiment_records/model_switch_temporal_arbitration_20260729.md`

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- `py -m pytest -q tests/test_model_switch_temporal_arbitration.py tests/test_live_model_switch.py`
- `py harness/tools/model_switch_temporal_arbitration_replay.py --input-root D:/cbond_on/results/experiments/model_switch_temporal_arbitration_20260729/input_snapshot_20260729 --output-root D:/cbond_on/results/experiments/model_switch_temporal_arbitration_20260729`

## Artifacts

- Frozen inputs: `D:/cbond_on/results/experiments/model_switch_temporal_arbitration_20260729/input_snapshot_20260729/`
- Canonical run: `D:/cbond_on/results/experiments/model_switch_temporal_arbitration_20260729/run_20260729T105110Z_b65a3b4/`
- The canonical manifest records `database_writes=false`,
  `live_runtime_called=false`, `scheduler_called=false`, input hashes, source
  code hashes, dirty Git state, and the strict-14:29 provenance caveat.

## Open Risks

- The 538-day history has already informed prior research. Results are not a
  clean final holdout and cannot be promoted without subsequent prospective
  shadow validation.
- Any dynamic threshold requires a direct candidate-versus-Base uncertainty
  estimate; a raw top-two score gap is not calibrated evidence.
- The direct-LCB rule is deliberately conservative: the strongest temporal
  candidate-side LCB remained below zero on every eligible disagreement date.
  Reducing its uncertainty allowance after seeing this result would be a
  same-sample threshold search and is not a valid promotion basis.
- The frozen state remains `14:30_not_strict_1429_certified`; neither this
  replay nor its clean Base reproduction repairs the provenance limitation.

## Next Action

- Report the research result and keep live unchanged. A possible next research
  phase is a prospectively frozen shadow period with strict-as-of state
  provenance; no live promotion or threshold reduction follows from the
  present in-sample replay.

## Handoff Summary

- Completed the approved isolated first pass. Live configuration, DB,
  scheduler, Champion list, model state, `results/live`, and `results/analysis`
  remain untouched.
