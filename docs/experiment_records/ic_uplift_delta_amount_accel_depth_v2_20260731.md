# Delta-amount acceleration-depth v2 OOS feasibility diagnostic (2026-07-31)

## Decision

Reject the frozen `delta_amount_v2_r1` experiment without running
`score/evaluate`. The factor is correctly defined as missing on multiple
structurally unusable T1430 windows, but those missing dates make a complete
strict-120-calendar-day final OOS comparison impossible. A validation-only
number would be incomplete and is not reported as an IC result.

This is research-only work. No strategy rule, mask, execution window,
turnover, fee, benchmark, live config, Champion, DB, scheduler, model state,
or live output was changed.

## Fixed question and implementation

Question: after correcting the legacy Wave80 misuse of cumulative `amount`,
can a T1430 amount-flow acceleration times recent L1-depth factor improve
Regsim through a causal residual Ridge correction?

The independent research-only factor
`t1430_amount_accel_depth_delta_v2` uses only same-day T1430 panel columns:

- it differences cumulative `amount` by `(dt, code)` before forming flows;
- it uses early `(14:19, 14:24]` and recent `(14:24, 14:29]` intervals,
  strict `trade_time <= 14:29`, and a 90-second boundary freshness guard;
- it requires a pre-14:19 seed, retains bounded signed vendor corrections,
  and continues to fail fast on material counter resets, negative counters,
  invalid timestamps, and negative depth;
- a structurally unavailable whole-market window returns its original
  `(dt, code)` index with all values `NaN`. It is not filled with zero or an
  invented seed.

The only frozen candidate plans were anchored residual Ridge with exactly 120
immediately preceding Regsim score-calendar days and `alpha=20`:

| Arm | Inputs |
| --- | --- |
| Delta | delta-amount factor |
| Delta + VWAP | delta-amount factor + `vwap_30m` |

## Scratch build and data audit

All derived outputs were confined to
`D:\cbond_on\research_scratch\ic_uplift_oos_20260731\delta_amount_v2`.
The completed FactorStore has 543 daily parquet files from 2024-05-08 through
2026-07-30, matching the 543 available clean snapshot dates. Across 241,116
rows, 233,165 (96.7024%) are finite, 7,951 are `NaN`, and there are no
infinities, duplicate `(dt, code)` entries, schema mismatches, or date/file
mismatches. Median daily finite coverage is 99.277%.

The following 17 dates are wholly unavailable for this factor:

```text
2026-03-26, 03-27, 03-30, 04-01, 04-03, 04-07, 04-10, 04-16,
04-17, 05-11, 05-12, 05-13, 05-14, 05-18, 05-19, 05-29, 06-17
```

Several further dates have low but nonzero coverage, including 2026-04-02
(37.54%), 04-09 (56.64%), 05-06 (72.59%), and 05-15 (29.62%). On 2026-03-26
the clean-direct panel itself loaded 1.78 million rows for 356 codes; the
failure is specifically a tail-window structure problem, not an absent day or
bad cumulative amount. There are only three rows in each of the
14:17:30--14:19 and 14:19--14:24 periods, while 14:24--14:29 has 7,399 rows.
Thus loosening the seed/freshness rule would fabricate an unobserved early
flow and is not a valid fix.

The production T1430 FactorStore was scanned separately: its 623 files contain
the v2 target column in zero files. This verifies that this candidate did not
enter the production store; it does not make claims about unrelated live writes.

## Strict OOS feasibility preflight

The Regsim calendar has 541 score dates and 59 final-reporting dates from
2026-05-06 through 2026-07-30. With the required exact 120 immediately prior
calendar days, the first unavailable date (2026-03-26) blocks every later
window through the current calendar end. The latest unavailable date
(2026-05-29) would require 80 additional Regsim score-calendar dates before a
valid window could resume.

| Item | Result |
| --- | ---: |
| Candidate OOF days possible under frozen contract | 337 |
| Validation OOF days | 112 / 137 |
| Final OOF days | **0 / 59** |
| Final paired Regsim days | **0** |

The current shared two-arm section builder also reads the union of Delta and
VWAP features. Missing `vwap_30m` on 2026-06-15 through 2026-07-30 would
therefore additionally exclude even the Delta-only arm. Splitting sections by
arm or shrinking/bridging the training window would be a new experiment, not a
repair of this frozen one.

## Integrity, limits, and follow-up

The factor reads no label, pool, mask, daily, stock, DB, or live output, and
score was intentionally not run. Historical clean snapshots have event
timestamps but lack immutable as-of/ingest versions, so even a future positive
historical result would remain timestamp-clipped replay evidence rather than
proof of live availability.

The next candidate must first pass a full-history tail-window coverage audit
and should avoid the unavailable early-window comparison. It will be a new,
pre-registered research candidate, not a tuned continuation of this rejected
one.

## Verification

```powershell
py -m pytest tests\test_t1430_amount_accel_depth_delta_v2.py -p no:cacheprovider -q
# 8 passed
```
