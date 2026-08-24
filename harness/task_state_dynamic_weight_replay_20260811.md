# Task State: No-threshold dynamic continuous-weight fusion (2026-08-11)

## Objective

- Evaluate a research-only, fixed-formula continuous allocation across the
  frozen Regsim, Ensemble, and HL20 return sleeves.  The aim is to test whether
  the existing causal relative-utility forecasts are more useful as allocation
  weights than as a daily hard winner selector.

## Locked Contract

- Source: `D:/cbond_on/research_scratch/model_switch_relative_utility_20260811/run_20260811_relative_utility_v5_with_ranker/`.
- Candidate returns: the source's frozen aligned standalone full-cycle
  `day_return` streams; all dynamic history is strictly before `score_day`.
- Variants, fixed before reviewing results: equal three-model allocation,
  causal-scale forecast softmax, 50% equal-weight shrinkage of that softmax,
  and a historical relative-return EWM softmax with half-life 20.
- No BaseGap/Robust/Fusion route, Champion role, LCB, gap, margin, veto, or
  parameter search.
- Main output is a **sleeve/portfolio mixture proxy**, not a single Top20
  score-level execution backtest.  It must be reported separately from any
  later score-rank blend study.

## State / Missing Input Rule

- The frozen relative-utility source has unavailable forecast/state warmup
  rows.  On precisely those dates, forecast-derived variants use fixed equal
  weights; this is availability handling, not an alpha threshold or Regsim
  fallback.
- Report both all 547 aligned return rows and the 541 rows with complete
  execution metadata.  The six incomplete suffix rows remain visible as a
  sensitivity, not silent production evidence.

## Safety Boundary

- New code only: `harness/tools/model_switch_dynamic_weight_replay.py` and its
  test.  All run artifacts are under
  `D:/cbond_on/research_scratch/model_switch_dynamic_weight_20260811/`.
- No live configuration, scheduler, database, model state, score root,
  trade list, `results/live`, `results/analysis`, or `results/backtest` may be
  changed.

## Verification Plan

- Verify every predecessor frozen input hash.
- Unit test simplex validity, strict no-future-outcome causality, frozen-score
  rank blend construction, and identical-score-universe enforcement.
- The full score-level generic backtest branch is explicit opt-in and is not
  part of this lightweight sleeve-proxy task unless separately requested.

## Completed Evidence

- Source input verification passed for 1,648 frozen predecessor files.
- Isolated result root:
  `D:/cbond_on/research_scratch/model_switch_dynamic_weight_20260811/run_20260811_fixed_formula_sleeve_proxy_v1/`.
- All 547 rows: Regsim `174.68% / Sharpe 3.975`; equal sleeve allocation
  `166.41% / 3.913`; forecast softmax `166.45% / 3.916`; 50% shrinkage
  `166.43% / 3.916`; historical EWM control `163.32% / 3.870`.
- Complete-metadata 541-row sensitivity preserves the conclusion: Regsim
  `172.22% / 3.968`; best fixed-formula proxy is the shrinkage softmax at
  `162.55% / 3.885`.
- The forecast softmax has mean effective model count `2.818`, mean entropy
  `1.062`, and total daily-weight turnover `56.42`; its 50% shrinkage has
  `2.948`, `1.090`, and `28.21`, respectively.  These are allocations, not
  single-book turnover estimates.
- Tests passed:
  `py -3.11 -m pytest -q tests/test_model_switch_dynamic_weight_replay.py tests/test_model_switch_relative_return_replay.py tests/test_model_switch_temporal_arbitration.py tests/test_live_model_switch.py`
  -> `42 passed`.

## Current Conclusion

- Fixed-formula continuous allocation improves the direct BaseGap sleeve proxy
  but does not beat standalone Regsim.  It is not a candidate for a live
  change.  Do not infer an executable Top20 score-level result from the sleeve
  proxy; score-level rank blending remains a separate, unrun research branch.
