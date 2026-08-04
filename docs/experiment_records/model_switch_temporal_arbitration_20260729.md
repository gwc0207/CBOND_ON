# Model-Switch Temporal Arbitration — First Frozen Replay (2026-07-29)

## Outcome

No candidate is promotable from this experiment. The frozen direct-LCB action
rule allowed zero overrides for all six pre-specified temporal forecasters, so
each candidate reverted to the Base fallback and underperformed the frozen
current routing.

This is a useful negative result: replacing the old 5bp internal Robust
top1-top2 gate with a direct, uncertainty-aware `r-b` test does not reveal a
reliable exploitable edge in this historical sample.

## Scope and safety boundary

- Research-only, offline replay.
- No change to `cbond_on/config/live`, `run/`, `liveLaunch/`, production DB,
  scheduler, Champion list, model state, `D:/cbond_on/results/live`, or
  `D:/cbond_on/results/analysis`.
- Inputs were byte-copied before replay to
  `D:/cbond_on/results/experiments/model_switch_temporal_arbitration_20260729/input_snapshot_20260729/`.
- Canonical output:
  `D:/cbond_on/results/experiments/model_switch_temporal_arbitration_20260729/run_20260729T105110Z_b65a3b4/`.
- Its manifest records `database_writes=false`, `live_runtime_called=false`,
  and `scheduler_called=false`.

## Frozen comparison contract

- Score-day range: 2024-05-08 through 2026-07-27, 538 aligned days.
- Three full-liquidation standalone shadow return streams: Regsim, Ensemble,
  and HL20. Frozen strategy configuration confirms `turnover_ratio=1.0`, so
  these are valid selector-return counterfactuals under the current contract.
- Base was rebuilt through the read-only production function
  `decide_scoreopt_t1430_dispersion()` with its actual 60-day history, 40
  nearest states, and `trim20_lcb10` score. It reproduced Base model id,
  reason, and score gap exactly on `538/538` days.
- Primary intervention envelope: only the existing 239 historical dates with
  `base_reason=margin_default` and a numeric existing Robust diagnostic. All
  other dates copy frozen current routing.
- The state input retains the known provenance limitation: it is
  `14:30_not_strict_1429_certified`. This experiment neither repairs nor
  claims to prove strict 14:29 PIT compliance.

## New arbitration contract

For each eligible date:

```text
b = actual Base top-ranked model
r = temporal forecaster's top-ranked candidate
```

The action condition is not the temporal model's internal `top1 - top2` gap.
It records and evaluates separately:

- Base view of candidate: `B_r - B_b`, candidate's Base rank, and paired
  similar-day uncertainty;
- temporal direct expectation: `E[r - b]`, Base's temporal rank, and prior
  prediction calibration/intrinsic forecast uncertainty;
- third-model evidence: `third-b` and `r-third` direct predictions;
- dynamic lower confidence bound:
  `LCB = combined_direct_mean - 1.0 * combined_scale`.

Only `LCB > 0` can override Base. The prior 5bp Robust gap remains an audit
field only. Combining uses a conservative uncertainty treatment; it does not
assume Base and temporal estimates are independent.

## Pre-specified temporal candidates

1. Frozen current independent-pair 44-feature Ridge, with only its action
   rule changed to direct `r-b` arbitration.
2. Coherent two-contrast 44-feature Ridge.
3. EWMA two-contrast expected utilities (60-day half-life).
4. Full-information fixed-share Hedge (`eta=sqrt(2 ln(3)/120)`, share 1/60),
   with EWMA direct return evidence rather than treating weight gaps as bp
   forecasts.
5. Two-contrast local-level Kalman model (`Q=0.02R`).
6. Dynamic model averaging across local-level `Q/R in {0, 0.02, 0.10}` with
   forgetting 0.99.

There was no parameter sweep or post-result threshold adjustment.

## Result

| Strategy | Total return | Sharpe | Max drawdown | Overrides vs Base | Delta return vs current | Delta Sharpe vs current |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen current routing | +147.4008% | 3.4146 | -6.5775% | 33 | 0.0000pp | 0.0000 |
| Base fallback in envelope | +144.0064% | 3.3776 | -6.3269% | 0 | -3.3943pp | -0.0370 |
| Each of the six direct-LCB candidates | +144.0064% | 3.3776 | -6.3269% | 0 | -3.3943pp | -0.0370 |

The old current routing's 33 historical Robust overrides were `19` wins and
`14` losses, with `+4.2369bp` mean realised gain versus Base. That is a small,
already-observed in-sample sample—not evidence that the old top-two gap is
calibrated or that authority should be enlarged.

### Why the new gates did not act

- The Ridge pair forecasts remain essentially uninformative: all three
  current/coherent pairwise OOS correlations are between about -0.002 and
  +0.032; OOS R-squared is negative; direction hit rates are about 48.5% to
  50.5% across 412 ready days.
- The time-series candidates did not establish a robust direct edge either.
  The most encouraging single diagnostics (EWMA Regsim–Ensemble direction
  53.8%, one-sided p about 0.065; local-level Ensemble–HL20 direction 54.8%,
  p about 0.028) have near/negative OOS R-squared and arise among multiple
  pairs/models. They do not clear a pre-declared aggregate reliability bar.
- Across eligible candidate disagreements, even the best temporal-only direct
  LCB was still below zero (current-pair Ridge maximum about -0.46bp). Once
  the Base view `B_r-B_b` and both uncertainty terms are included, every
  combined LCB remains negative.

Therefore a lower post-hoc threshold would be a same-sample search, not a
validated improvement.

## Verification

- `py -m pytest -q tests/test_model_switch_temporal_arbitration.py tests/test_live_model_switch.py`
  -> `28 passed`.
- New tests cover Helmert coherence, Hedge normalization/update order,
  local-level/DMA positive-semidefinite covariance, direct-LCB pass/reject
  cases independent of top-two gap, Base-rank audit, missing-evidence
  abstention, and a future-poison replay invariant.
- Re-running the same frozen input produced exactly identical selections,
  reasons, and LCB values across 3,228 forecaster-day rows.

## Interpretation and next boundary

This pass answers the first question clearly: Ridge is not validated as a
return-predictive arbitration engine here, and the first low-dimensional
full-feedback/state-space alternatives do not improve that conclusion under a
strict dynamic confidence rule.

Do not modify live selection, shrink the LCB allowance, or restart any
scheduler based on these results. A defensible next phase is prospective,
frozen shadow monitoring after a separately approved strict-as-of state-data
contract. Any redesign of uncertainty fusion should be pre-registered and
evaluated out of sample rather than tuned on these 538 days.
