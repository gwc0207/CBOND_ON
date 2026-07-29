# Task State: Dynamic Weight Optimization (2026-07-28)

## Objective

- Improve the rejected Base-gap / Robust-top-gap dynamic weight with causal pairwise opposition and prior-outcome reliability signals.

## Current Verified Facts

- Direct pairwise opposition against the Base winner is more informative than Robust top-two gap for the 2026-07-28 Ensemble case.
- Consensus pairwise dynamic weighting improves the full 538-day shadow composite but loses in the chronological first half.
- Prior-outcome reliability calibration reduces, but does not eliminate, that first-half loss.
- No dynamic candidate has met time-stability or position-continuity requirements for live promotion.

## Files Changed

- `docs/experiment_records/model_switch_dynamic_weight_optimization_20260728.md`

## Artifacts

- `D:/cbond_on/results/analysis/model_switch_dynamic_pairwise_fusion_20260728/run_20260728_155658/`
- `D:/cbond_on/results/analysis/model_switch_dynamic_consensus_fusion_20260728/run_20260728_155937/`
- `D:/cbond_on/results/analysis/model_switch_dynamic_reliability_fusion_20260728/run_20260728_160311/`

## Open Risks

- Current evaluation joins standalone model shadow returns, not a continuous fused holding path.
- Selecting formulas from the same 538-day sample would overfit.

## Next Action

- Keep all experimental fusion rules out of live configuration. Build a continuous-position walk-forward replay only after explicit approval.
