# Task State: Diverse Cross-Sectional Atomic Models (2026-08-15)

## Objective

- Test two genuinely differentiated research-only atomic candidates using the
  existing daily rolling CBOND_ON contract: fast27 DeepSets/ListNet and
  structural23 linear/ListNet.

## Locked Contract

- Baseline: frozen live50 Regsim on the aligned window.
- Score/backtest window: 2024-07-04 through 2026-06-10; the 2026-06-11/12
  live50 input gap is fail-closed and will not be bridged.
- 60-day natural rolling history, daily refit, model-local warm start only.
- Full frozen live50 remains the raw/preprocess/admission contract: >=27
  available factors, T-1 o_0005 allowlist, no winsor, T-1 style-5 Ridge
  neutralization, daily z-score, and internal missingness masks.
- The two models only differ in `model_input_factors` after that shared
  contract.  Trading remains strict Top20 equal weight, 14:42-14:57 buy and
  next-day 09:30-09:39 sell, using the existing generic backtest contract.

## Scope and Safety

- Research-only output root:
  `D:/cbond_on/research_scratch/atomic_cross_sectional_models_20260815_r1/`.
- No live config, DB, scheduler, live score root, model state, strategy rule,
  universe, factor-store data, or production result is in scope.
- Do not reuse any r1/r2/r3 checkpoint.  Each new candidate owns an isolated
  daily state chain.

## Planned Evidence

- Focused unit tests for the subset contract and linear cross-sectional model.
- Two-date causal smoke, including output schema, 54/46 input dimensions,
  strict label boundary, and distinct output roots.
- Full contiguous daily score chain followed by the same aligned generic
  rolling strategy backtest and comparison with frozen Regsim.

## Current State

- Focused contract tests pass (`24 passed`), including the regression that a
  factor subset is selected only after full50 admission and preprocessing.
- The first smoke attempt failed before any score/state artifact because the
  runner read only the subset while attempting to z-score full50.  Its
  isolated logs remain under `.../smoke/logs/` as failure evidence; nothing
  was overwritten or deleted.
- The repaired, fresh-root `smoke_r2` completed both score dates for both
  models.  It proved 54/46 model inputs (value plus missingness mask), strict
  T-1 label boundaries, T-1 allowlist use, non-constant score files, and the
  one-step warm-start parent chain.
- Both separate full 2024-07-04 through 2026-06-10 daily rolling runs
  completed from the isolated r1 profile.  Each has 468 score days and 468
  checkpoints, with a verified model-local parent chain; no r1/r2/r3 state
  was reused.
- Both aligned generic backtests completed with 468 successful days, strict
  Top20, T-1 `o_0005`, zero fallback, and zero skips.  Neither candidate
  passes the standalone-quality admission bar versus frozen Regsim.
- Result: reject fast27 DeepSets for this configuration; retain structural23
  linear only as a research diversity diagnostic.  Do not add either to live
  models or model selection without a new owner-approved hypothesis.
- Full evidence and the no-promotion conclusion are recorded in
  `docs/experiment_records/atomic_cross_sectional_models_20260815_r1.md`.
