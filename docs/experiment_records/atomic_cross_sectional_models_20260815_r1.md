# Diverse Cross-Sectional Atomic Models r1 (2026-08-15)

## Question

Can two genuinely differentiated, cross-sectional atomic models provide
standalone quality under the frozen live50 rolling contract before any further
model-selection research?

## Locked comparison contract

- Score and backtest window: `2024-07-04` through `2026-06-10` (468 trading
  days), matching frozen Regsim exactly.
- Continuous 60-natural-trading-day rolling history, daily refit, cold start
  only once and then model-local daily warm start.
- Full frozen live50 remains the shared admission/preprocessing path:
  `>=27` available factors, T-1 `o_0005` allowlist, no winsor, T-1 style-5
  Ridge neutralization, daily z-score, and internal missingness mask.
- Model-specific input is selected only after that shared path:
  fast27 has 27 factors / 54 value-and-mask inputs; structural23 has 23
  factors / 46 inputs.  The two factor lists are mutually exclusive and their
  union is exactly live50.
- Objective: equal-day ListNet.  Execution: strict Top20 equal weight,
  `twap_1442_1457` buy and next-day `twap_0930_0939` sell, T-1 `o_0005` only,
  lag 0.

## Candidates

| Candidate | Factor input | Cross-sectional model |
|---|---:|---|
| `atomic_cs_fast27_deepsets_listnet_20260815_r1` | fast27, 54 inputs | DeepSets + ListNet |
| `atomic_cs_structural23_linear_listnet_20260815_r1` | structural23, 46 inputs | low-capacity linear + ListNet |

## Implementation correction and smoke evidence

The first two-day smoke failed before any score/state artifact because the
runner read only the model slice but attempted to z-score the complete live50
set.  This correctly fail-closed with a missing-column `KeyError`; it was not
a model or data result.  The runner was corrected to read/admit/neutralize/
z-score full50 first, then select `model_input_factors` for the model.  A
regression test locks that ordering.

The fresh `smoke_r2` subsequently passed for both models: 2024-07-04 cold
start, 2024-07-05 warm start from the preceding checkpoint, no target-day
label read, non-constant finite scores, exact T-1 validation-label boundary,
and 54/46 checkpoint input dimensions.  The failed smoke root was retained as
evidence and not reused.

Focused validation: `24 passed` for torch cross-sectional model, causal-score,
and adapter tests; `3 passed` for the prepared backtest contracts.

## Full score-chain audit

Both candidates passed a strict post-run audit:

- 468 score files and 468 checkpoint files, exactly `2024-07-04..2026-06-10`;
- no 2026-06-11/12 gap crossing;
- one stable contract fingerprint per model and continuous checkpoint parent
  chain;
- every audit row has `41` train + `18` validation days, validation ending on
  exact natural T-1, and no future label;
- finite, non-constant, duplicate-free scores; 184,196 total score rows per
  model.

Score diversity relative to frozen Regsim:

| Candidate | Mean score Pearson | Mean score Spearman | Mean Top20 overlap |
|---|---:|---:|---:|
| fast27 DeepSets | 0.094 | 0.060 | 5.15 / 20 |
| structural23 linear | 0.078 | 0.037 | 5.20 / 20 |

## Aligned strategy results

All rows below use the identical 468-day execution contract.  `IC` fields are
the equal-day averages from the generated generic-backtest `ic_summary.csv`.

| Model | Annual return | Sharpe | MDD | Excess annual | mean IC (t) | mean RankIC (t) | Daily-return corr vs Regsim | Top20 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Regsim | 60.94% | 4.067 | -4.76% | 38.66% | 0.04737 (6.73) | 0.02232 (7.01) | 1.000 | 20.00 / 20 |
| structural23 linear | 41.38% | 2.963 | -5.68% | 21.79% | 0.02322 (4.98) | -0.00139 (-0.40) | 0.805 | 5.20 / 20 |
| fast27 DeepSets | 30.51% | 1.919 | -8.42% | 12.61% | 0.02276 (3.60) | -0.02403 (-4.37) | 0.830 | 5.16 / 20 |

Both backtests have actual `2024-07-04..2026-06-10`, 468 successful trading
days, zero skips, zero missing-score days, exactly 20 picks per day,
allowlist applied on all 468 days, and zero allowlist fallback days.

## Fixed half-year OOS slices

| Period | Regsim Sharpe | structural23 Sharpe | fast27 Sharpe | Regsim annual | structural23 annual | fast27 annual |
|---|---:|---:|---:|---:|---:|---:|
| 2024H2 (122 d) | 4.89 | 3.31 | 1.96 | 94.57% | 58.35% | 40.04% |
| 2025H1 (117 d) | 3.14 | 3.70 | 2.48 | 38.81% | 40.73% | 33.49% |
| 2025H2 (126 d) | 4.07 | 2.70 | 2.19 | 44.13% | 30.34% | 27.85% |
| 2026H1 (103 d) | 4.13 | 2.37 | 1.25 | 74.05% | 37.25% | 19.99% |

## Conclusion

- The input-slice hypothesis succeeded at creating genuinely distinct score
  streams without changing the live factor, mask, or trading contracts.
- Neither candidate has adequate standalone quality to enter live selection:
  both trail frozen Regsim on full-window return, Sharpe, drawdown, raw IC,
  and rank IC.  Fast27 is decisively rejected for this configuration.
- Structural23 is the better research artifact: it has a one-half-year
  outperformance (2025H1) and much lower score/holding overlap, but its full
  performance remains materially weaker and its RankIC is effectively zero.
  Keep it as a diagnostic/diversity reference only; do not promote or use it
  in model switching.

## Evidence paths

- Score/state/model roots:
  `D:/cbond_on/research_scratch/atomic_cross_sectional_models_20260815_r1/runtime/results/`.
- structural23 backtest:
  `.../backtest/2024-07-04_2026-06-10/Research_AtomicCS_Structural23Linear_Aligned_20260815_r1/20260815_112143/`.
- fast27 backtest:
  `.../backtest/2024-07-04_2026-06-10/Research_AtomicCS_Fast27DeepSets_Aligned_20260815_r1/20260815_112345/`.
- Frozen aligned Regsim reference:
  `D:/cbond_on/research_scratch/non_tree_cross_sectional_models_20260813_r3_backtest/runtime/results/backtest/2024-07-04_2026-06-10/Research_Live50RegsimFrozen_Aligned_20260813/20260813_130035/`.

This is research-only evidence.  No live factor configuration, live model
configuration, scheduler, DB, live output, or production model state changed.
