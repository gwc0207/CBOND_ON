# Daily relative-rank coupling v1 preflight (research-only)

## Candidate and fixed contract

- factor: `drrc_return_amount_rank_spearman60` in family
  `prior_relative_return_amount_rank_coupling`;
- requested IC start: `2025-01-01`; actual score calendar:
  `2025-01-02..2026-07-30` (381 days);
- factor visibility: T1430 / 14:30, using only `market_cbond.daily_price`
  rows with `trade_date < score_date`; label remains same-score-day 14:42;
- evaluation universe: only the external screen applies strict previous-session
  `quant_factor_dev.researcher_xuvb.o_0005`.  The factor itself never reads a
  mask or labels;
- fixed acceptance gates: absolute mean daily Pearson IC strictly above `0.02`,
  at least 250 valid days and 50 days per chronological 60/20/20 partition,
  at least 200 common redundancy days, within-family redundancy `<0.80`, and
  cross-family redundancy `<0.70`.

## Formula and data visibility

For every historical daily-price session, calculate the percentile ranks over
all valid observable convertible-bond rows for:

1. `log(close_price / prev_close_price)`;
2. `log(amount)`.

For each output bond, calculate their Pearson correlation across the latest 60
strict-prior source sessions (at least 45 finite pairs).  This is the rank
coupling, not a raw amount/return regression.  The factor requires a finite
row at the latest strict-prior source session; it otherwise returns `NaN` and
does not carry an older value forward.  No score-day row, future row, label,
mask, PnL, database, or file access is used by the factor implementation.

## Zero-write exploratory evidence

An in-memory read-only screen used the fixed 381 score days, same-day 14:42
labels, and strict T-1 `o_0005` only at scoring time.  Cross-sectional ranks
were corrected to use the full historical daily convertible-bond market, not
the future union of `o_0005` codes.  The candidate's raw mean daily Pearson IC
was `+0.023629`, with 381 valid days and chronological split means
`+0.018622 / +0.033116 / +0.029092` (discovery / validation / holdout).

A corrected zero-write pairwise audit of the implemented formula against all
64 v7 quality-eligible factors reported maximum cross-family redundancy
`0.475931` (`dliq_volume_return_corr20`), under the canonical daily
Pearson/Spearman maximum metric and with 381 common valid days.  This is
preflight evidence only: the final value must be recomputed by the global
screen after an immutable full root is built and merged.

## Boundary and next gate

The implementation is an explicit research module only.  It is not imported by
`defs.__init__`, is not referenced by any live/model config or factor contract,
and has no Rust/live admission.  Focused unit tests and a scratch-only smoke
are required before a new, previously absent full build root may be considered;
while aggregate v5 is active, do not launch that full build.

## Static and smoke verification

Focused verification passed:

```text
27 passed
ruff check passed
```

The no-write full-window launcher preflight passed with an absent intended root
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_relative_rank_coupling_v1_full`.
The isolated 2026-04-28 generic-runner smoke wrote only
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_relative_rank_coupling_v1_smoke_20260428`.
Its FactorStore has 330 unique `(dt, code)` rows, the exact 14:30 timestamp,
323 finite values, seven explicit `NaN` values, zero `Inf`, and a nonconstant
finite range `[-0.303581, +0.637746]`.  Its run manifest is research-only and
records only the one-day scratch range.  No full build was started.
