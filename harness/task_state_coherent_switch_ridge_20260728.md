# Task State: Coherent Two-Contrast Switch Ridge (2026-07-28)

## Objective

- Prepare a strictly research-only selector replay that replaces three
  independently fitted pairwise Robust Ridge targets with two coherent
  orthogonal contrasts of the three-model label vector.

## Risk Level

- Medium research-only.  The tool reuses current selector-routing artefacts,
  but it must not import the live runtime, change live config, write a DB, or
  restart scheduling.

## Current Verified Facts

- Current Robust fits three separately clipped pairwise Ridge targets even
  though three model utilities have only two degrees of freedom.
- The existing 44-feature Ridge exactly reproduces the current replay but has
  near-random pairwise OOS diagnostics.
- There is no existing switch-aware label generator or labelled panel in the
  current research roots.
- The current active `strategy01_topk_turnover` config has
  `turnover_ratio=1.0`; its implementation consults `prev_positions` only
  below 1.0, while the strict cycle fully sells the next morning.  Therefore
  the three existing standalone shadow `day_return` histories are equivalent
  to daily selector labels under the current trading contract.
- The replay independently checked that equality against the current routing:
  538 aligned days, 0 mismatches, max absolute return difference 0.0.
- The coherent replay reconstructs sum-zero utilities but changes no final
  model choice (0/538 overall; 0/239 eligible); full-period return, Sharpe,
  max drawdown, and switch count equal the current replay exactly.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/context/source_of_truth.md`
- `cbond_on/infra/live/model_switch.py`
- `harness/tools/ridge_robust_replay.py`

## Files Changed

- `harness/task_state_coherent_switch_ridge_20260728.md`
- `harness/tools/coherent_switch_ridge_replay.py`
- `docs/experiment_records/model_switch_coherent_two_contrast_ridge_20260728.md`

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment` (earlier
  in this research thread)
- `py -m py_compile harness/tools/coherent_switch_ridge_replay.py`
- `py harness/tools/coherent_switch_ridge_replay.py --shadow-current-full-liquidation`

## Artifacts

- Planned root when labels are supplied:
  `D:/cbond_on/results/experiments/model_switch_coherent_ridge_20260728/`
- Completed run:
  `D:/cbond_on/results/experiments/model_switch_coherent_ridge_20260728/run_20260728_230859/`

## Open Risks

- The current shadow-label equivalence is contingent on full liquidation.  If
  `turnover_ratio<1.0`, any future partial-turnover or persistent-holdings
  strategy requires an external switch-aware label/evaluation panel instead.

## Next Action

- Keep the coherent alternative research-only and do not tune it further on
  this sample. Preserve the external label/evaluation CSV interface for a
  future partial-turnover strategy.

## Handoff Summary

- Current-trading-contract research is complete. Live code/configuration and
  production state remain untouched.
