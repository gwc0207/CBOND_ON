# Daily debt-floor wedge source-coverage audit (2026-08-03)

## Question and scope

Can the research-only
`research_factor_mining_daily_debt_floor_wedge_dynamics_v1` catalogue satisfy
the fixed factor-mining validity contract over the requested `2025-01-01`
start (`2025-01-02..2026-07-30`, 381 score days) without a full FactorStore
build?

This was a DataHub-input feasibility audit only.  It did not run a full-window
factor build, screen, backtest, model, score, label calculation, DB write, live
configuration change, or scheduler action.

## Fixed contract audited

- T1430 / 14:30 factor and same-score-day 14:42 label contract;
- strict `trade_date < score_date` for both daily inputs;
- fixed prior-trading-day `quant_factor_dev.researcher_xuvb.o_0005` pool,
  using the immutable v6 pool audit;
- `market_cbond.daily_price`: `exchange_code`, `close_price`;
- `market_cbond.daily_base`: `exchange_code`, `debt_puredebt_ratio`,
  `puredebt_prem_ratio`, `ytm`, `duration`;
- 66 visible source-file lookback; a valid input tail must be consecutive in
  the daily-price session calendar and have finite required fields with
  positive duration;
- screen minima: 30 pooled observations per day, 250 valid days overall, and
  50 valid days in each chronological discovery / validation / holdout slice.

The audit read 446 required historical files from each declared daily source.
It reproduced the module's strict-prior anchor, date-key join, and tail-input
conditions, but deliberately did not calculate factor values or IC.  Counts
below are therefore source-input upper bounds, not IC results.

## Input-feasible score days

| Signal group / required tail | Overall days with >=30 pool-eligible inputs | Discovery | Validation | Holdout | 250 overall | 50 per partition |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `dfw_wedge_*` change signals / 6 | 331 | 228 | 56 | 47 | yes | **no** |
| persistence signals and OOS residual / 21 | 316 | 228 | 56 | 32 | yes | **no** |
| residual acceleration / 26 | 311 | 228 | 56 | 27 | yes | **no** |
| residual z-score / 41 | 296 | 228 | 56 | 12 | yes | **no** |

Thus the global 250-day count alone is achievable, but no signal family can
meet the full fixed quality gate because holdout input feasibility is below 50
days in every case.  Actual usable-factor/IC coverage can only be lower than
these input bounds.

## Root cause

`daily_price` files remain present and complete around the observed failure.
The declared `daily_base` fields `debt_puredebt_ratio` and
`puredebt_prem_ratio` are both entirely null for these source-file sequences:

- 2026-03-10 to 2026-03-20 (9 file days)
- 2026-03-24 to 2026-03-25 (2)
- 2026-03-27 (1)
- 2026-03-31 to 2026-04-02 (3)
- 2026-04-07 to 2026-04-13 (5)
- 2026-04-15 to 2026-04-17 (3)
- 2026-04-21 to 2026-04-29 (7)
- 2026-05-06 to 2026-05-15 (8)

Strict consecutive-tail requirements extend the effective zero-input periods:

- tail 6: 2026-03-11 to 2026-05-25;
- tail 21: 2026-03-11 to 2026-06-15;
- tail 26: 2026-03-11 to 2026-06-23;
- tail 41: 2026-03-11 to 2026-07-14.

No missing-field fallback, zero fill, or stale carry-forward was used or is
permitted.

## Layered strict-PIT smoke confirmation

| Score day | Rows | Finite cells across 9 signals | Empty signals | Constant signals | Inf cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-04-02 (existing healthy baseline) | 477 | 4,235 | 0 | 0 | 0 |
| 2026-04-01 | 349 | 0 | 9 | 0 | 0 |
| 2026-04-02 (existing break smoke) | 349 | 0 | 9 | 0 | 0 |
| 2026-04-03 | 348 | 0 | 9 | 0 | 0 |

Each smoke used the generic scratch-isolated expansion runner, had one
`research_only` manifest, and used a distinct fresh root.  The two 2026
neighbouring roots were created specifically for this audit; the 2025-04-02
and 2026-04-02 rows are explicitly reused evidence from the preceding
diagnostic, not newly rerun results.

## Decision

Do not start a full-window DFW build and do not add this module to the next
aggregate under the current DataHub input contract.  A repair of the historical
`daily_base` fields, or an owner-approved new factor design with independent
PIT validation, is required before a new feasibility audit.  This is not an
IC-quality conclusion and does not modify any production path.

## Evidence

- `D:/cbond_on/research_scratch/factor_mining_20260803_dfw_daily_coverage_feasibility_audit_v1/audit_summary.json`
- `D:/cbond_on/research_scratch/factor_mining_20260803_dfw_daily_coverage_feasibility_audit_v1/daily_source_file_quality.csv`
- `D:/cbond_on/research_scratch/factor_mining_20260803_dfw_daily_coverage_feasibility_audit_v1/score_day_input_feasibility.csv`
- `D:/cbond_on/research_scratch/factor_mining_20260803_dfw_daily_coverage_feasibility_audit_v1/coverage_gap_runs.csv`
- `D:/cbond_on/research_scratch/factor_mining_20260803_dfw_daily_coverage_feasibility_audit_v1/smoke_output_quality.csv`
