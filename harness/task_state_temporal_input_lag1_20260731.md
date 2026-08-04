# Task State

## Objective

- Test whether adding a strictly available previous-trading-day factor state
  improves the already causal label/execution-lag-one Regsim contract.

## Question and Locked Contract

- Baseline: the completed research-only model
  `research_regsim_label_execution_lag1_20260730`, whose contract is
  `F(T) -> L[next(T)]`, score at `T`, buy at `next(T)`, and sell at
  `next(next(T))`.
- Variant input: for each of the 27 frozen base factors, use raw
  `[F(T), F(P(T)), F(T)-F(P(T))]` before the existing same-day
  winsor/neutralization/standardization steps. The target, label embargo,
  61-slot rolling window, refit cadence, sample weighting, fee model,
  allowlist, frozen universe, benchmark, and execution lag remain unchanged.
- `P(T)` is the immediate previous **raw trading-calendar** day. A missing
  prior factor snapshot must not fall back to an older day; the first variant
  uses an explicit code inner join and records any resulting coverage loss.

## Risk Level

- medium research-only code change. It affects generic LGBM feature assembly,
  so the feature option must be strictly disabled by default. No live config,
  DB, scheduler, Champion, live score root, or live model-state root may be
  used.

## Baseline Evidence

- Baseline score/artifact root:
  `D:/cbond_on/results/experiments/label_execution_lag1_20260730/`.
- Final baseline backtest:
  `backtest/2025-10-31_2026-07-29/Research_LabelExecutionLag1_RegsimFull_20260730/20260730_210229/`.
- Baseline result: 179 actual days, 18.935% total return, 1.862 Sharpe,
  -6.444% maximum drawdown. Two historical factor dates have insufficient
  columns and remain explicit score gaps.

## Required Evidence

- Disabled feature option preserves old dataset/schema behavior.
- Exact raw-calendar `T -> P(T)` mapping, including a weekend test.
- No fallback across a missing prior factor day; no source label later than
  the score day.
- Variant score, model state, artifact, and backtest outputs under an
  independent experiment root.
- Aligned score/actual buy windows and metrics versus the frozen baseline.

## Planned Output Root

- `D:/cbond_on/results/experiments/label_execution_lag1_temporal_inputs_20260731/`

## Open Risks

- Historical T1430 input remains marked 14:30 rather than independently
  certified 14:29 PIT.
- 81 features increase model capacity and can overfit a historical window;
  this is not a live-promotion experiment.

## Outcome

- Completed the opt-in raw factor-state expansion and its isolated full
  rolling score/backtest. The variant uses 81 stable columns (`27 t0`,
  `27 lag1`, `27 diff1`), strict raw-calendar prior-day mapping, and an
  independent warm-start chain.
- Focused verification passed: `10 passed`; architecture guard: `ok`.
- Full score succeeded for 178 score days (`2025-10-30..2026-07-28`) and
  56,616 score rows. The final backtest completed at
  `D:/cbond_on/results/experiments/label_execution_lag1_temporal_inputs_20260731/backtest/2025-10-31_2026-07-29/Research_LabelExecutionLag1_TemporalInputs_20260731/20260731_115715/`.
- The strict common-date (178-day) comparison is adverse: baseline total
  return / Sharpe / MDD = `19.122% / 1.884 / -6.444%`; temporal-input
  variant = `11.731% / 1.182 / -6.485%`. The paired mean daily difference
  is `-3.574bp` (t=`-1.092`, two-sided p=`0.276`).
- There is one additional intentional strict score gap at factor day
  `2026-06-15` / actual trade date `2026-06-16`; its immediate raw-calendar
  predecessor fails the frozen factor schema, and no older-day bridge was
  used.

## Next Action

- Do not promote or alter live. Treat this direct 81-column state expansion
  as a negative research result. Any successor should be separately
  pre-registered and validated on new out-of-sample or prospective shadow
  data rather than tuned on this same historical window.

## Handoff Summary

- Research-only task completed after explicit owner approval. No live config,
  DB, scheduler, Champion, live score/state root, or live output was mutated.
- Full findings and exact comparison contract are recorded in
  `docs/experiment_records/lgbm_temporal_factor_inputs_label_execution_lag1_20260731.md`.
