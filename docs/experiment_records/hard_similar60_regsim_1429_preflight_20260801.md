# Hard Similar60-Regsim strict-14:29 preflight — 2026-08-01

## Question

Can a Hard Similar60 training selector improve the frozen current Regsim model
without changing the 27-factor input, T-1 neutralization, HL20/regime weights,
`o_0005`, execution windows, fees, or `strategy01_topk_turnover`?

## Locked candidate contract

- Baseline: `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708`.
- Candidate-only difference: select nearest 60 training days and ranks 61--80
  validation days from an exact 360-prior-trading-day market-state window.
- State cutoff: `<= 14:29:00`; no 14:29:30 or 14:30 observations.
- Score construction: no score-day 14:42 label read; existing causal T-1
  `o_0005` applies before scores are frozen.
- Any state/training gap is explicit and aborts this candidate before a score
  can be written.  A separate explicit composer can make a byte-identical
  Regsim copy for a separately labelled fallback artifact; it is never a
  Similar60 observation.

## Feasibility result

Read-only manifest analysis covered the 624 clean/publish trading-day records
through 2026-07-31.  The former *minimum reconstruction* check required both
the clean manifest and publish `.done` to be successful, carry the same run
id, and have `produced_at` on the score day at or after 14:29.  It is not a
forward point-in-time certificate: a final manifest written after the cutoff
does not prove that the exact immutable source version was available at the
cutoff.

| Availability class | Dates |
|---|---:|
| Same-day minimum reconstruction evidence | 76 |
| Rejected by backfill / intraday timing | 548 |
| First same-day evidence | 2026-04-08 |
| Last same-day evidence at preflight | 2026-07-31 |

For example, 2025-10-30 has clean/publish run id
`backfill_20250101_20260407_114330` and `produced_at=2026-04-09`; it cannot be
claimed as point-in-time availability on 2025-10-30.  Even under the weaker
reconstruction check, there are only 76 days, so 284 additional days would be
needed to fill a 360-day pool.

The current DataHub V1 manifest contract is stricter still: only four dates
(2026-07-28 through 2026-07-31) contain the full V1/profile/validation/asset
metadata, and their final post-cutoff files are **historical reconstructions**,
not forward-certified state observations.  The forward-certified count is
therefore zero.  A valid forward study needs a cutoff-time immutable manifest
or append-only source watermark before new days can accumulate toward 360.

## Decision

No historical or live Similar60 score/backtest was run from this preflight.
The strict study is blocked: it has zero forward-certified state days (and,
even under the weaker reconstruction count, remains 284 days short).  It must
not silently bootstrap the missing history from later backfills or treat a
regular rolling fallback as Similar60 evidence.

An owner could explicitly choose a different research question: a **forward
shadow with a frozen, presently reconstructed historical state bootstrap**.
That would be labelled non-historical-PIT and could not be reported as a
strict historical Similar60 validation.  It is not authorized by this record.

## Research-only implementation prepared

- `harness/tools/build_strict_t1429_state_history.py`: audited source/state
  generator; writes no FactorStore, label, DB, scheduler, or live artifact.
- `harness/tools/similar60_compose_score_root.py`: byte-level Regsim fallback
  composition into a separate root.
- `harness/tools/ic_uplift_score_pair.py`: freezes pair score/code evidence and
  validates source/pool/frozen hashes before labels are opened.

The implementation is intentionally not a promotion path.  A future forward
study still needs a fresh 120 completed-score-day window and its predeclared
paired IC/return gates.
