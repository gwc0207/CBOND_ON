# Task State: No-threshold relative-utility model selector (2026-08-11)

## Objective

- Evaluate a research-only, direct three-model selector that predicts coherent relative utilities for Regsim, Ensemble, and HL20, then selects the highest predicted utility without any alpha-confidence threshold, BaseGap gate, Champion preference, Robust veto, or LCB action rule.

## Risk Level

- Medium research risk; live stability risk is controlled by an isolated output root and no modification of protected live paths.

## Current Verified Facts

- The current live `scoreopt_t1430_dispersion` selector is already a direct BaseGap argmax; its `margin` only affects the audit reason, not the selected model.
- On the current live50 return panel, direct three-model BaseGap does not beat standalone Regsim on its valid selection window.
- The current strategy has `turnover_ratio=1.0`, so aligned standalone shadow `day_return` can be used as daily selector counterfactuals under the existing execution contract.
- Earlier state-only relative-return Ridge and LSTM work did not establish usable OOS signal. This run therefore adds pre-specified, point-in-time model-score disagreement features and reports state-only as a negative-control comparator.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/context/source_of_truth.md`
- Relevant model-switch experiment records through 2026-08-06.

## Files Changed

- This task-state file only at initialization.

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`

## Artifacts

- Planned isolated root: `D:/cbond_on/research_scratch/model_switch_relative_utility_20260811/`.

## Open Risks

- The current state artifact retains a historical 14:29/14:30 provenance caveat and must be disclosed.
- The 501 valid score days are a small sample. Architecture and hyperparameters must be predeclared before reviewing holdout performance.
- No current research result authorizes a live promotion.

## Next Action

- Implement an isolated frozen-input replay and focused causal/invariant tests; run it only after input hashes and comparison contract are captured.

## Handoff Summary

- Owner explicitly authorized this research experiment while requiring no threshold-based selection design.
