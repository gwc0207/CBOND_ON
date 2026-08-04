# Daily observable seasoning v1 — 2026-08-03

## Status

Research-only candidate family.  It is not in `defs.__init__`, live/model
configuration, factor contracts, a database, scheduler, or the production
FactorStore.  The evidence below is a full-window pre-screen plus a one-day
mechanical smoke; it is not final global admission.

## Fixed contract

- Requested IC start: `2025-01-01`; actual score calendar:
  `2025-01-02..2026-07-30` (381 sessions).
- Factor timestamp: same-day T1430/`14:30`; label: same-day `14:42`.
- Feature inputs: only `market_cbond.daily_price` rows strictly before the
  score date.
- Screening universe: strict T-1 `quant_factor_dev.researcher_xuvb.o_0005`.
- Gate: `abs(mean daily Pearson IC) > 0.02`, 250 valid days, 50 valid days in
  each chronological 60/20/20 partition, 200 common redundancy days;
  same-family redundancy `< 0.80`, cross-family redundancy `< 0.70`.

## Mechanism

The family measures observable market seasoning, not an issuer's unobserved
listing age or a static maturity proxy:

- `osa_terminal_amount_streak60`: terminal run length, capped at 60, of
  completed strict-prior sessions with positive amount.  A valid-close,
  zero-amount terminal session has formula value zero; a missing/stale terminal
  daily-price row fails closed to `NaN`.
- `osa_observation_density60`: fraction of the latest 60 strict-prior market
  sessions where the security has a valid positive close, conditional on a
  valid latest strict-prior close.

No `o_0005` code enters either formula; it is used only in later evaluation.
Missing sessions stay missing after reindexing, rather than being filled.

## Zero-write full-window evidence

The initial ten-member observable-seasoning pre-screen retained the following
two mutually admissible members:

| factor | overall daily Pearson IC | discovery | validation | holdout | valid days | maximum v7 cross-family redundancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `osa_terminal_amount_streak60` | -0.026728369 | -0.012239782 | -0.038246878 | -0.058260721 | 381 | 0.452114966 |
| `osa_observation_density60` | -0.020690000 | -0.004749000 | -0.029832094 | -0.058868556 | 381 | 0.629500881 |

Their same-family daily redundancy is `0.782062123` over 381 common days and
therefore is below the required `< 0.80` threshold.  Neither had a cross-v7
conflict at `>= 0.70` in the provisional audit.

Rejected or omitted siblings were not added by renaming/window duplication:

- `osa_amount_reactivation_count60` had IC `+0.024080300` and low v7
  redundancy (`0.352791485`), but redundancy with the terminal streak was
  `0.991134717`; it cannot coexist with the selected internal member.
- `osa_positive_amount_density60` had IC `-0.022140663`, but redundancy with
  the terminal streak was `0.999675192` and with observation density
  `0.982394028`.
- The deal-density sibling was numerically identical to amount density in the
  pre-screen.  No duplicate was implemented.

Separately, a market-flow-share idea was rejected despite IC
`+0.038408043`: it had ten v7 cross-family conflicts at `>= 0.70`, maximum
`0.816436678`, so no module was created for it.

## Implementation and verification

New research-only files:

- `cbond_on/domain/factors/defs/research_factor_mining_daily_observable_seasoning_v1.py`
- `tests/test_research_factor_mining_daily_observable_seasoning_v1.py`
- `cbond_on/domain/factors/defs/research_factor_mining_daily_orthogonal_batch_v7.py`
- `tests/test_research_factor_mining_daily_orthogonal_batch_v7.py`

Focused verification:

```powershell
py -3.11 -B -m pytest -q tests/test_research_factor_mining_daily_observable_seasoning_v1.py tests/test_research_factor_mining_daily_orthogonal_batch_v5.py tests/test_research_factor_mining_daily_orthogonal_batch_v6.py tests/test_research_factor_mining_daily_orthogonal_batch_v7.py tests/test_run_factor_mining_expansion.py -p no:cacheprovider
py -3.11 -m ruff check cbond_on/domain/factors/defs/research_factor_mining_daily_observable_seasoning_v1.py cbond_on/domain/factors/defs/research_factor_mining_daily_orthogonal_batch_v7.py tests/test_research_factor_mining_daily_observable_seasoning_v1.py tests/test_research_factor_mining_daily_orthogonal_batch_v7.py
```

Result: `16 passed`; Ruff clean; scoped `git diff --check` clean.

The generic runner passed a no-write preflight, then completed a real isolated
one-day Python-engine smoke at:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_observable_seasoning_v1_smoke_20260428
```

Its T1430 output had 330 unique `(dt, code)` rows, both factor columns had 330
finite values, zero `Inf`, and exact `14:30` timestamps.  A separate raw-data
recomputation from the strict-prior 60 daily-price sessions matched both
columns exactly (maximum absolute difference `0.0`).

## Next action

`daily_orthogonal_batch_v7` is a future catalogue only.  Do not start its
full root while the independent aggregate-v5 full build is active.  After v5
has completed naturally and passed its immutable-root audit, run a fresh,
serial full build, audit it, outer-union compose all complete roots, run the
fixed screen, and then exact MIS.  The final all-pool screen can still reject
these provisional candidates if later complete roots introduce a conflict.
