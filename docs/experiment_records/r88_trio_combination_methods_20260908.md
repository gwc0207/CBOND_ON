# R88 Trio Combination-Method Comparison (2026-09-08)

## Question

Compare pre-registered ways to combine frozen `P6 + Full88 P3 + Regsim`
scores without changing the `o_0005` universe, market mask, Top20, strict
cycle execution, fees, TWAP fields, or benchmark.

The objective is not the highest full-sample return.  It is a stable
research candidate with acceptable Sharpe, HAC alpha, drawdown, and
time-block behaviour.

## Locked Contract

- Current DataHub execution inputs; 399 common dates from `2025-01-02` through
  `2026-08-27`.
- No P6/P3/Regsim retraining or re-scoring.
- Each model is percentile-ranked in its own valid score universe.  For a code
  in the unchanged `o_0005` target universe without one model score, that model
  contributes neutral rank `0.5`.
- Every single-Top20 method writes one fused score and runs the same generic
  strict backtest.
- 2025 is development and rolling-training only.  2026 is observed reporting
  only and cannot select a weight, window, or hyperparameter.
- Sleeve allocation is an independent multi-book diagnostic contract.  It is
  never presented as a single Top20 strategy.

## Method Families

### Single-Top20 score fusion

1. Equal-rank baseline and a 15-point static long-only simplex grid.
2. Equal-weight normal-score fusion.
3. Strict-prior 120-day positive Ridge and NNLS cross-sectional score stacking.
4. Strict-prior 120-day fixed-share Hedge, Bayesian evidence, EWMA utility,
   shrinkage MVO, downside, CVaR, and CDaR score weights.
5. A no-threshold rolling best-expert control.
6. A strongly regularised CPU-only rolling LGBM meta allocator.

### Three-sleeve allocation controls

Equal weight, Ledoit-Wolf GMV, shrinkage MVO, ERC, CVaR, CDaR, robust MVO,
and Bayesian shrinkage allocation.

## Input and Execution Audit

- The as-run audit verifies 403 relevant DataHub source dates, 1,202 raw input
  files, and 1,209 raw/clean/publish evidence files.
- The input digest is
  `e06605a6bfdd2d3dd9cfd175f840a1ef754a2418a94dd6e96ef5505544b3aea5`.
- The audit was rechecked before score methods, before sleeves, and before
  validation.  No file or configuration drift was detected.
- All 27 single-Top20 methods completed 399 aligned dates with 20 names, full
  weight, zero cash, and an identical daily benchmark.
- No DB, live runtime, scheduler, training, or model-scoring call occurred.

## Result

The equal-rank score-fusion result remains the defensible research baseline.
No new score-fusion or sleeve method demonstrated a statistically supported
increment over it.

- Equal-rank baseline: 2025 `46.64% / Sharpe 4.122 / MDD -6.44% / alpha t
  3.991`; 2026 reporting `34.71% / 3.920 / -4.06% / 3.621`.
- Development leaders were CVaR score weights (`47.72% / Sharpe 4.234`) and
  fixed-share Hedge (`47.17% / 4.146`), but their incremental returns reversed
  in the observed 2026 period.  Their full-period deltas versus equal rank are
  only `+0.094bp/day (HAC p=0.913)` and `+0.024bp/day (p=0.903)`.
- Normal-score fusion had the strongest observed 2026 return (`37.59%`) but a
  lower 2025 Sharpe (`3.705`) and cannot be selected from 2026.
- Ridge and NNLS stacking did not improve robustly.  The rolling best-expert
  control increased drawdown.  Strong LGBM regularisation produced equal
  predictions for all experts and exactly equal `1/3` weights, so it is a null
  result rather than an independent allocator.
- Static simplex CSCV PBO is `50.0%`, with selected OOS median rank `9.5/17`.
  Static selection is therefore not robust.

## Formal Validation

- The score ledger contains 31 method labels but only 28 unique daily-return
  sequences.  LGBM equals equal rank, while two simplex endpoints equal their
  respective P6/P3 standalone controls.
- Performance ledgers retain all labels.  Formal DSR, White RC, Hansen SPA,
  MCS, and PBO use only the 28 unique-return representatives; the duplicate
  mapping is saved in `duplicate_return_sequences.csv`.
- Static and full-family White RC/Hansen SPA do not reject no-superiority.
  The full unique-return family has White RC p-values `0.414`, `0.335`, and
  `0.315` for development, reporting, and overall; corresponding consistent
  SPA p-values are `0.412`, `0.187`, and `0.287`.
- MCS at 10% retains all 28 unique score-return sequences.  It does not select
  a statistically distinguishable mean-return winner.
- PSR/DSR remains descriptive evidence that individual strategy returns are
  positive.  It does not establish incremental superiority versus equal rank.

## Sleeve Boundary and Result

Sleeve outputs linearly weight three independently traded Top20 books; the
combined book can contain roughly 20--60 names.  This is not equivalent to a
single fused score and Top20 selection.

Fixed equal sleeves produce `86.87% / Sharpe 3.604 / MDD -5.90%`.  No dynamic
sleeve improves this robustly.  CVaR adds only `+0.082bp/day` overall with HAC
`p=0.925` while worsening drawdown to `-6.36%`; Bayesian shrinkage is worse by
`-0.657bp/day` with `p=0.023`.

The sleeve calculation has not replayed merged holdings, position netting, or
portfolio-layer execution cost, so it cannot be used as a live Top20 candidate.

## Decision Boundary

- This experiment creates no live candidate and does not authorize a live
  change.
- Do not use the observed 2026 period to retune a new method.
- A next step requires either a newly pre-registered 2025-only research family
  or a frozen existing rule followed by fresh forward-shadow evidence.

## Artifacts

`D:/cbond_on/research_scratch/r88_trio_combination_methods_20260908_r1/full_methods_20260908_r1/`

- `as_run_input_audit/`
- `score_fusion/`
- `sleeve_allocation/`
- `validation/`
- `RESULTS.md`
