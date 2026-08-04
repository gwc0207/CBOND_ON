# Daily relative rank-tail contradiction v1 preflight (research-only)

## Fixed contract and formula

- requested IC start: `2025-01-01`; actual score days:
  `2025-01-02..2026-07-30` (381);
- factor: T1430 / 14:30; label: same-score-day 14:42; strict T-1 `o_0005`
  is applied only by the research screen;
- daily input: `market_cbond.daily_price.exchange_code`, `prev_close_price`,
  `close_price`, and `amount`, all with `trade_date < score_date`;
- `drrq_return_amount_opposite_tail_excess60` is the 60-session mean of
  `I(return_rank <= 0.20 and amount_rank >= 0.80) - 0.04`, requiring 45 joint
  finite sessions and finite terminal ranks.  Both ranks use the full valid
  date-local daily convertible-bond market, never a future `o_0005` union.

This is a nonlinear completed adverse-flow frequency, distinct from the linear
rank-coupling family.  Its rejected single-trade-size sibling has redundancy
`0.874204` within the same family and is not implemented.

## Read-only preflight evidence

The candidate has 381 valid daily IC observations, mean daily Pearson IC
`+0.030149`, and chronological means `+0.022383 / +0.049579 / +0.033963`
(discovery / validation / holdout).  Under the canonical mean daily absolute
Pearson/Spearman maximum redundancy metric, it has:

- maximum v7 quality-eligible-factor redundancy `0.681037` with
  `twap_segment_volatility` (381 common days), below the cross-family 0.70
  gate;
- maximum redundancy `0.528123` against the current five independently
  pre-screened daily candidates (deal-sign MI, trade-size-sign MI, joint
  transition entropy, downside beta, upside beta), 381 common days;
- redundancy `0.384672` with the rank-coupling candidate.

These values are research preflight only.  A later immutable full root and
fresh global merge/screen must recompute them before any acceptance claim.

## Boundary

The module is not imported through `defs.__init__`, is not referenced by live
or model configuration or contracts, and has no Rust/live admission.  It must
pass focused tests, a no-write full preflight, and a separate scratch-only
smoke before any full root can be launched after v5 releases the resource gate.

## Static and smoke verification

Focused verification passed (`25 passed`; Ruff clean).  The generic runner's
no-write full-window preflight passed against the still-absent intended root
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_relative_rank_tail_contradiction_v1_full`.
The isolated 2026-04-28 scratch smoke wrote only
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_relative_rank_tail_contradiction_v1_smoke_20260428`:
330 unique `(dt, code)` rows, exact 14:30 timestamp, 323 finite values, seven
explicit NaNs, zero Inf, and nonconstant finite range `[-0.04, +0.493333]`.
No full build, production write, DB write, or scheduler action occurred.
