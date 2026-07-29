# Task State: Model-Switch Cost Equivalence Check (2026-07-28)

## Objective

- Verify whether a challenger-versus-Base label under one shared previous-day
  basket differs from the existing standalone shadow-return difference under
  the current Top20 strict-cycle execution contract.

## Risk Level

- Medium, research-only. No live configuration, model state, DB, scheduler, or
  production output may be changed.

## Current Verified Facts

- The active strategy is `strategy01_topk_turnover`, configured as Top20,
  5% per name, `turnover_ratio=1.0`.
- The strategy only consults `prev_positions` if `turnover_ratio < 1.0`.
- Current strict execution is a full buy `twap_1442_1457` to next-day sell
  `twap_0930_0939` cycle with 1.0 / 1.2bp fees.
- The strict-cycle function consumes only the current target buy basket, not a
  preceding portfolio.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/context/source_of_truth.md`
- `cbond_on/domain/strategies/strategy01/strategy01_topk_turnover.py`
- `cbond_on/app/usecases/backtest_runtime.py`
- `cbond_on/infra/live/shadow_returns.py`
- `cbond_on/infra/benchmark/service.py`
- `cbond_on/domain/portfolio/service.py`
- relevant current live strategy/model-switch and fee configs, read-only

## Files Changed

- `harness/tools/switch_cost_equivalence_check.py`
- `harness/task_state_switch_cost_equivalence_20260728.md`
- `docs/experiment_records/model_switch_switch_cost_equivalence_20260728.md`

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- `py -m py_compile harness/tools/switch_cost_equivalence_check.py`
- `py harness/tools/switch_cost_equivalence_check.py`
- independent result-table and sequence recomposition verification via `py -`

## Artifacts

- Verified run:
  `D:/cbond_on/results/experiments/model_switch_switch_cost_equivalence_20260728/run_20260728_231117/`.

## Open Risks

- A persistent continuous-hold/rebalance-cost simulation would be a different
  execution contract and must not be described as the current strict-cycle
  label without separately approved scope.
- The current 538-day selector sequence has 175 model changes, but all
  1,614 active-model selections remain identical under shared or disjoint
  previous baskets; a model switch does not introduce an additional cost term
  in the present daily complete-cycle contract.

## Next Action

- Keep current Ridge/statistical research on the existing standalone pairwise
  label. Do not construct a new continuous-position label unless the owner
  separately specifies a replacement execution contract.

## Handoff Summary

- The full score-data equivalence proof and independent recomposition passed.
  The experiment record documents why a continuous-hold cost replay is out of
  scope for the current strict-cycle strategy contract.
