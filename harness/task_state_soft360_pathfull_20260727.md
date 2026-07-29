# Task State

## Objective

- Implement and evaluate a research-only `Soft360Pool` analogue to Similar60: use a 360-day effective historical `path_full_t1430` state pool, reserve distance ranks 61--80 for validation, and assign continuous Gaussian distance weights to the remaining 340 training days with target day-level ESS of 60.

## Risk Level

- medium: model-training code and a long research run; no live, scheduler, database, model-state, or live-config change is in scope.

## Current Verified Facts

- MT-014 Similar60 uses a 360-day effective state pool, ranks 1--60 for training and 61--80 for validation, daily cold-start, and no HL20 or Regsim weight.
- Soft360Pool will reuse the same repaired research-only state source, factor/label contract, score isolation, and strict backtest contract.
- The state source remains available through 14:30, whereas the live factor chain is bounded at 14:29. This trial is research-only and cannot be promoted.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/context/source_of_truth.md`
- `docs/experiment_records/similar60_pathfull360_20260727.md`
- current Similar60 model/backtest configs and training runner

## Files Changed

- `cbond_on/infra/model/similar_day_training.py`
- `cbond_on/infra/model/runners/train_lgbm.py`
- `tests/test_similar_day_training.py`
- research-only Soft360Pool model/backtest configs
- experiment record and model-tuning ledger

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- focused tests: `py -m pytest tests/test_similar_day_training.py tests/test_strict_cycle_returns.py` (`12 passed`)
- four-day real-data score smoke: 2026-07-20 to 2026-07-23
- full research-only daily cold-start score run: 2025-10-30 to 2026-07-23
- strict Soft360Pool backtest through the generic `strategy_backtest` CLI

## Artifacts

- Experiment root: `D:/cbond_on/results/experiments/soft360pool_pathfull360_ess60_20260727/`
- Full model audit: `D:/cbond_on/results/models/lgbm_screened_no_winsor_neutral_tminus1_soft360pool_pathfull360_ess60_cold_v1_20260727/2025-10-30_2026-07-23/20260727_224654/`
- Strict backtest: `D:/cbond_on/results/backtest/2025-10-30_2026-07-23/Research_Soft360Pool_pathfull360_ess60_cold_20251030_20260723/20260727_231254/`

## Open Risks

- State remains 14:30 rather than the live 14:29 cutoff; no experiment result is live-promotable.
- Soft360Pool's kernel day ESS is fixed at 60 but equal-row training yields final day ESS from 50.52 to 97.40; this is measured and disclosed, not silently corrected by a second weighting mechanism.
- Soft360Pool underperforms Hard Similar60 and has roughly 23 minutes of full historical training cost.

## Next Action

- Do not promote Soft360Pool. If further similarity research is requested, first rebuild the state history to `<=14:29`, then test a separately pre-registered hybrid or production-family comparison.

## Handoff Summary

- Complete as a research-only negative result versus Hard Similar60. No live, scheduler, database, model-state, or live-config mutation occurred.
