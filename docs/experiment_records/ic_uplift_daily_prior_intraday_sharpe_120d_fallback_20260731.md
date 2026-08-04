# Daily prior-intraday Sharpe residual-Ridge OOS experiment (2026-07-31)

## Decision

Reject `prior_intraday_sharpe_120d_fallback_r1` as an IC-uplift candidate.
It does not deliver a material or statistically supported improvement over
Regsim, and its final-reporting RankIC and Top20 mean label are lower.  Do not
tune its Ridge alpha, 120-session window, coverage threshold, transform, or
blend weight against these observed results.  No strategy backtest or live
promotion follows from this experiment.

This was research-only.  `strategy01_topk_turnover`, every existing mask
including `o_0005`, buy/sell TWAP windows, turnover, fees, benchmark,
universe, live config, scheduler, Champion, DB, model state, and
`results/live` were not changed.

## Frozen hypothesis and point-in-time contract

For a score day `T` and bond `i`, the scratch-only factor is:

```text
r[i, d] = twap_1442_1457[i, d] / twap_0930_0935[i, d] - 1
f[i, T] = mean(r[i, d]) / sample_std(r[i, d])
          over exactly the last 120 source sessions d < T
```

- The factor code explicitly filters `trade_date < T` even though the daily
  loader supplies the T-day file.  It uses no label, pool, mask, trade list,
  DB, or live output.
- Each bond requires all 120 finite, positive-price historical observations
  and a nonzero sample standard deviation; missing values remain missing and
  are never zero-filled.
- The frozen score arm was anchored residual Ridge, `alpha=20`, using the
  exact 120 immediately prior **Regsim score-calendar slots**.  At least 96
  usable factor/label slots were required; unavailable slots were not bridged
  with older dates.
- A current day needed at least 30 matched codes and 80% factor coverage.  On
  a non-usable day, or for a currently unavailable code, the output was raw
  Regsim exactly.  This preserves the full score-day/code universe without
  imputing a feature.

Historical raw snapshots do not prove immutable vendor as-of versions.  The
code-level `d < T` boundary is enforced, but all historical conclusions remain
timestamp-clipped research evidence rather than live PIT certification.

## Scratch build and audit

The build was confined to:

```text
D:/cbond_on/research_scratch/ic_uplift_oos_20260731/prior_intraday_sharpe_120d/
```

Configuration used Python/CPU, `workers=1`, `refresh=false`,
`overwrite=false`, `backtest_enabled=false`, with only the new factor.  The
completed sidecar FactorStore has 543 files from 2024-05-08 through
2026-07-30, 241,116 rows, 204,181 finite values, 36,935 explicit NaNs, no
duplicate `(dt, code)` rows, no schema/index mismatch, no nonnumeric value,
and no `+/-inf`.

- The first 40 globally under-warmed days were all-NaN as required.  The first
  day with 120 earlier daily source sessions was 2024-07-04 and had 496
  finite values.
- Regsim has 541 score days in the study range; the two factor-only days are
  2026-06-11 and 2026-06-12.  There are no score-only days.
- All 59 final-reporting score days passed the frozen 80% threshold.  Their
  matched-factor coverage ranged from 91.441% to 93.973% (mean 93.973%).
- A full physical-schema scan of the 623-file production FactorStore found
  the new factor column in zero files.

## OOF score integrity

Canonical experiment output:

```text
D:/cbond_on/results/experiments/ic_uplift_oos_20260731/
  prior_intraday_sharpe_120d_fallback_r1_20260731_001/
```

The score output contains all 541 Regsim dates and 219,936 unique `(score_day,
code)` rows, exactly matching the raw Regsim universe.  Both Regsim and the
candidate are finite on every row.  `availability_audit.csv` and
`fallback_code_audit.csv` record every day/code decision.

| Audit item | Result |
| --- | ---: |
| Usable factor days | 501 / 541 |
| Ridge-fitted score days | 405 / 541 |
| Whole-day fallback days | 136 / 541 |
| Overall raw-Regsim fallback code share | 35.377% |
| Final-reporting fitted days | 59 / 59 |
| Final-reporting raw-Regsim fallback code share | 6.026% |
| Same-day label boundary violations | 0 |
| Training calendar slots exact/no bridge | 541 / 541 audit rows |

The large all-history fallback share is primarily the declared 120-slot
warm-up.  It is not hidden by intersecting away early days; the final segment
has a fitted model on every day.

## Fixed-split IC result

The primary metric is daily cross-sectional Pearson IC against the same
score-day 14:42 label, equal-weighted by day.  The output code universe and
`n` are identical for candidate and Regsim on every evaluated day.

| Segment | Days | Candidate Pearson IC | Regsim Pearson IC | Paired delta | Paired t | RankIC delta | Top20 mean-label delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation 2025-10-08--2026-05-05 | 137 | 0.004797 | 0.004765 | +0.000032 | +0.513 | -0.000184 | +0.000034 |
| Final reporting 2026-05-06--2026-07-30 | 59 | 0.027338 | 0.027247 | +0.000091 | +0.185 | -0.001046 | -0.000134 |

The predeclared sign condition for paired Pearson delta is technically
positive in both segments, but the effect is economically negligible and
statistically uninformative.  It does not move the IC toward the 0.10 research
goal, and the two secondary metrics deteriorate in the final segment.  This
is therefore a rejection, not a positive result.

The final-reporting segment is chronological but not prospective live shadow
evidence; it has also been viewed by earlier exploratory IC candidates in this
research program.  It must not be used for further tuning or promotion claims.

## Verification

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
py -m pytest tests/test_ic_uplift_oos_residual_ridge.py `
  tests/test_daily_prior_intraday_sharpe_v1.py `
  tests/test_t1430_amount_accel_depth_delta_v2.py `
  -p no:cacheprovider -q
# 31 passed
```

Focused tests cover strict legacy-path preservation, mixed-policy refusal,
96/120 and 80% thresholds, no date bridging, full-code universe preservation,
code-level raw-Regsim fallback, same-day-label invariance, score/evaluate
artifact boundaries, and fallback evaluation `n` alignment.

## Next boundary

Choose a new, independent hypothesis using development/validation information
only.  Do not mutate this candidate or use its final-reporting result to tune
parameters.  Any later strategy assessment must retain the unchanged masks and
execution contract and requires a separately justified score result first.
