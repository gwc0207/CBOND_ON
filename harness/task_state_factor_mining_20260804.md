# Task State

## Objective

- Research-only factor-family expansion toward up to 100 globally admissible factors under the fixed 2025-01-01 IC contract.

## Risk Level

- medium: long CPU- and disk-intensive factor construction; all derived outputs must remain under `D:/cbond_on/research_scratch`.

## Current Verified Facts

- The fixed screen requires T1430/14:30 factors, same-day 14:42 labels, T-1 `o_0005`, `abs(mean daily Pearson IC) > 0.02`, at least 250 valid days, at least 50 days in each chronological partition, and at least 200 common redundancy days.
- Redundancy is strictly below 0.80 within a family and 0.70 across families; the selection cap is 100.
- The latest completed global baseline contains 535 factors over 381 score days, with 64 quality-eligible and exact-MIS capacity 33; it does not satisfy the requested target.
- The isolated v8 build is active at `D:/cbond_on/research_scratch/factor_mining_20260804_daily_orthogonal_batch_v8_full_r2`; it must finish naturally before another heavy full-window build is started.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/context/source_of_truth.md`
- `harness/workflows/research_experiment.md`
- `harness/workflows/long_task_context.md`
- `docs/开发规则.md`
- `docs/ai_factor_factory_dify_prompt.md`

## Files Changed

- This task-state file only. No live configuration, model, factor profile, DB, scheduler, mask, trading rule, or production output has changed.

## Commands Run

- Research and long-task harness preflights.
- Read-only process, catalogue, screen-contract, and existing-result audits.

## Artifacts

- Active isolated build: `D:/cbond_on/research_scratch/factor_mining_20260804_daily_orthogonal_batch_v8_full_r2`.
- Baseline screen: `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v7_with_joint_screen_v1`.
- Baseline exact selection: `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v7_with_joint_selection_optimal_v1`.

## Open Risks

- 100 is a strict global retained-count target, not a generated-column count; it may require several independent candidate waves.
- No full build may overlap the active v8 process.

## Next Action

- Implement and test a new strict-PIT, incremental microflow factor-family catalogue without touching active v8 source files; launch it only after v8 completes and passes immutable-root audit.

## Handoff Summary

- Research is isolated; live assets remain untouched.
