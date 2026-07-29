# Task State: Base / Robust Soft Fusion (2026-07-28)

## Objective

- Evaluate weighted Base/Robust selection only when both existing selectors are low confidence.

## Risk Level

- Research-only. No live config, database, scheduler, model state, factor, universe, or output contract was changed.

## Current Verified Facts

- The direct gap-adaptive implementation does not improve the current 538-day replay.
- Rolling score-scale normalization makes several fixed Robust-heavy variants attractive in-sample.
- A chronological half split does not validate those weights out of sample; the weight selected in the first half loses slightly in the second half.
- The 2026-07-28 pending decision would choose Regsim for Base 30% / Robust 70%, 50% / 50%, and the gap-adaptive rolling variant.

## Files Changed

- `docs/experiment_records/model_switch_soft_fusion_20260728.md`

## Commands Run

- Read-only 538-day Base/Robust weighted-fusion replay.
- Fixed-weight, gap-adaptive, rolling-scale and chronological split sensitivity checks.

## Artifacts

- `D:/cbond_on/results/analysis/model_switch_soft_fusion_20260728/run_20260728_153456/`

## Open Risks

- Current result chaining uses each model's own shadow history; it is not a fully dynamic fused-position backtest.
- Fixed weight quality is time-varying, so full-sample winner selection would overfit.

## Next Action

- Do not add a live fusion policy yet. If authorized, build a position-continuous, walk-forward fusion replay before considering promotion.

## Handoff Summary

- Soft fusion is more promising than the rejected hard veto, but has not met the standard for live activation.
