# Prior-intraday Sharpe residual-Ridge OOS experiment (2026-07-31)

## Decision

Reject `daily_prior_intraday_sharpe_120d_fallback_r1` as an IC-uplift
candidate.  The feature is causal, high-coverage, and not a duplicate of the
current 27-factor pack, but its frozen residual-Ridge correction produces only
tiny, statistically insignificant Pearson-IC changes and worsens final RankIC
and Top20 mean label.  Do not tune its window, Ridge alpha, fallback threshold,
or blend weights against these validation/final segments.

This is research-only.  No strategy rule, existing mask (including `o_0005`),
execution window, turnover, fee, benchmark, live factor/configuration,
Champion, DB, scheduler, live score/state, or trade list changed.  No strategy
backtest was run because the pre-registered score IC gate did not pass.

## Frozen question and contract

Question: can a strictly prior-session daily intraday state improve the current
Regsim score through a causal residual Ridge correction while retaining the
entire Regsim code universe?

For score date `T` and bond `i`, the candidate is

```text
r(i,d) = twap_1442_1457(i,d) / twap_0930_0935(i,d) - 1
f(i,T) = mean(r(i,d)) / sample_std(r(i,d))
         over exactly the latest 120 daily_twap sessions d < T
```

The score plan was fixed before the final reporting segment was opened:

- anchored residual Ridge on `z(y) - z(Regsim)`;
- Ridge `alpha=20`;
- exactly 120 immediately preceding Regsim score-calendar slots, with no date
  bridging;
- at least 96 usable feature/label slots to fit;
- current-day feature coverage at least 80%; and
- unavailable days or codes retain their raw Regsim scores exactly, rather
  than being imputed, dropped, or causing a mask/universe change.

The score stage opens only labels dated strictly before `T`; same-day 14:42
labels are opened only by the separate evaluation stage after OOF scores exist.

## Factor build and integrity audit

The Python-only scratch build used `market_cbond.daily_twap` through the
existing daily context and wrote only to:

```text
D:/cbond_on/research_scratch/ic_uplift_oos_20260731/prior_intraday_sharpe_120d/
```

The completed T1430 sidecar FactorStore has all 543 expected snapshot dates
from 2024-05-08 through 2026-07-30.  It has 241,116 rows: 204,181 finite and
36,935 explicit `NaN`, with zero infinities.  Every file has exactly the
candidate column plus unique `(dt, code)` index, with `dt=14:30:00` on its
file date.  `daily_twap` and cbond snapshot calendars both have all 623
sessions from 2024-01-02 through 2026-07-30; the first 120-prior-session
signal day is 2024-07-04, which is also the first day with a finite candidate
value.  No pre-warm-up finite value occurred.

The Regsim calendar contains 541 score days.  Its 59 final-reporting dates
(2026-05-06 through 2026-07-30) all have sidecar files; mean candidate/Regsim
coverage is 93.9727%, minimum 91.4414%, and every final day clears the frozen
30-code / 80% gate.  The production T1430 FactorStore was separately scanned:
the candidate column appears in zero of 623 files.

## Score/evaluation artifacts and causal audit

Artifact root:

```text
D:/cbond_on/results/experiments/ic_uplift_oos_20260731/
  prior_intraday_sharpe_120d_fallback_r1_20260731_001/
```

The score artifact contains 541 days and 219,936 code-day rows, exactly equal
to the source Regsim universe and raw baseline score on every code-day.

- 405 score days fitted the correction; 136 cold-start/unavailable days fell
  back entirely to Regsim.
- 77,807 code-day rows (35.3771%) used raw-Regsim fallback; every such output
  equals the baseline exactly.
- Every audit row marks the fixed calendar slots exact; every training label
  maximum is strictly before the score day.  Fitted days use 96--120 usable
  training days.
- On the final 59 days the model fitted all days.  It changed 93.97% of codes,
  but the average absolute raw-score adjustment was only `0.00007187`; final
  Top20 overlap with Regsim averaged 19.763/20 (minimum 18).

Focused verification:

```powershell
py -m pytest tests/test_ic_uplift_oos_residual_ridge.py `
  tests/test_daily_prior_intraday_sharpe_v1.py `
  tests/test_t1430_amount_accel_depth_delta_v2.py -p no:cacheprovider -q
# 31 passed
```

## Aligned OOS results

Daily IC is equal-weighted cross-sectional correlation against same-score-day
14:42 `y`; candidate and Regsim use identical code counts on every day.

| Segment | Regsim Pearson IC | Candidate delta | t(delta) | Candidate RankIC delta | Candidate Top20-y delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development, 345 days | 0.04940 | +0.000015 | 0.251 | +0.000321 | -0.000012 |
| Validation, 137 days | 0.004765 | +0.000032 | 0.513 | -0.000184 | +0.000034 |
| Final reporting, 59 days | 0.027247 | +0.000091 | 0.185 | -0.001046 | -0.000134 |

The final candidate Pearson IC is 0.027338, versus 0.027247 for Regsim.  The
small nominal increase is neither practically nor statistically persuasive;
the companion rank and Top20 diagnostics are adverse.  This therefore does
not support a strategy test or any live promotion.

## Duplicate check and interpretation

Using the project factor-selection convention (mean daily cross-sectional
Pearson and Spearman correlations), the largest
`max(abs(Pearson), abs(Rank))` against the live 27-factor pack is 0.21888
(`range_30m`); no pair reaches the 0.85 duplicate threshold.  The rejection
is therefore about absent stable incremental predictive information, not
factor duplication, coverage, or a fallback artifact.

Historical daily raw files do not supply immutable vendor as-of versions, so
this remains timestamp-clipped historical research evidence even apart from
the failed IC gate.
