# Task State: R88 P6 + Full88 P3 + Regsim current-data score-fusion baseline

## Objective

- Establish a research-only current-data, single-Top20 score-fusion baseline
  for the frozen `P6 + Full88 P3 + Regsim` trio.
- Keep all existing score trees fixed and execute each standalone plus the
  equal-rank fusion under one generic execution contract.

## Risk Level

- Medium. The work is research-only, but it concerns model combination and
  backtesting. Live configuration, model state, factor stores, database,
  scheduler, and live result roots are out of scope.

## Current Verified Facts

- P6 and Full88 P3 each have 401 frozen R88 score/return days from
  2025-01-02 through 2026-08-27.
- Regsim has an overlapping return history, but the R88/Regsim comparison is
  valid only on raw `day_return` common dates because historical benchmark
  values differ on known dates.
- Earlier read-only work found non-identical daily score universes. The
  existing generic score-fusion helper correctly refuses non-identical
  universes, so its behavior must not be bypassed silently.
- The user authorized only A0 preparation on 2026-09-08. B-stage score fusion,
  static-weight runs, dynamic weights, sleeve allocation, and all live actions
  remain unapproved.
- A0 completed under the isolated scratch root. It found 399 common score days
  from 2025-01-02 through 2026-08-27, but only 1/399 days has an identical
  three-model score universe. The mean code intersection is 345.40 versus a
  mean union of 383.57; the minimum intersection is 202.
- P6/P3 each expose 401 score files and Regsim exposes 569. The three return
  histories have 399 common days; Regsim lacks 2026-06-11 and 2026-06-12, and
  18 common dates have a non-identical benchmark value versus P6.
- A0 made no fused scores and did not invoke a generic backtest. Its contract
  therefore blocks B-stage until an owner-approved missing-score policy and a
  fresh generic Regsim identity replay are in place.
- Owner confirmed the neutral-rank policy and B1 on 2026-09-08. B1 invoked the
  generic runtime only for Regsim identity and then halted correctly before
  score fusion. On the corrected 399-day comparison, the current generic
  return differs from frozen Regsim by a maximum of 0.0024989529687471004 on
  2026-08-12. This is a genuine current-versus-frozen identity mismatch; the
  underlying cause is not inferred by this task.
- Read-only root-cause tracing now locates that mismatch at the mutable
  execution-input layer rather than the warm-start/training layer. The Regsim
  score file for 2026-08-12 was created at 2026-08-12 14:31, and the 2026-08-13
  scheduler log records the historical shadow-return append for the same score
  source through 2026-08-12. In contrast, the current local DataHub pool file
  for 2026-08-11 was created on 2026-08-17, and current daily TWAP/price files
  for 2026-08-12/13 were created on 2026-08-18/19, after the historical shadow
  row was calculated.
- Re-running the live shadow-return builder read-only with today's inputs gives
  exactly the same 2026-08-12 strategy return as generic B1 (`0.0151291039553838`),
  not the historical frozen value (`0.0126301509866367`). The historical row's
  benchmark count is 149 versus today's 295; the buy-leg difference is about
  4.26bp and sell-leg difference about 20.52bp. No immutable historic raw/pool
  byte snapshot remains, so the task cannot isolate whether the revised output
  is caused by pool membership, market-price/TWAP revision, or both.

- Owner then authorized the current-data baseline. B2 completed with frozen
  P6/P3/Regsim scores and the same current DataHub/raw/pool inputs, `o_0005`,
  Top20, fees, benchmark, and strict cycle contract on 399 common dates.
- B2 made no database, live-runtime, scheduler, training, or scoring call.
  The 399-day aligned results are P6 `94.50% / Sharpe 3.475`, P3
  `86.42% / 3.404`, Regsim `79.29% / 3.285`, and equal-rank fusion
  `97.54% / 4.005`. Fusion's paired difference versus current Regsim is
  `+2.409bp/day`, HAC `t=1.421`, `p=0.155`.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/research_experiment.md`
- `harness/workflows/long_task_context.md`
- `harness/context/source_of_truth.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/skills/cbond-long-task-memory/SKILL.md`
- `docs/experiment_records/r88_single_model_validation_phase2_20260902.md`
- `docs/experiment_records/r88_single_model_health_20260902.md`

## Files Changed

- `harness/tools/r88_trio_score_fusion_preflight.py`
- `harness/tools/r88_trio_score_fusion_b1.py`
- `harness/tools/r88_trio_score_fusion_b2_current_data.py`
- `harness/task_state_r88_trio_score_fusion_20260908.md`
- `docs/experiment_records/r88_trio_score_fusion_plan_20260908.md`
- `tests/test_r88_trio_score_fusion_b1.py`
- `tests/test_r88_trio_score_fusion_b2_current_data.py`

## Commands Run

- `py -3 -B harness/tools/agent_preflight.py --mode research-experiment`
- `py -3 -B harness/tools/agent_preflight.py --mode long-task`
- `py -3 -B -m py_compile harness/tools/r88_trio_score_fusion_preflight.py`
- `py -3 -B harness/tools/r88_trio_score_fusion_preflight.py --help`
- `py -3 -B harness/tools/r88_trio_score_fusion_preflight.py --output-root
  D:\\cbond_on\\research_scratch\\r88_trio_score_fusion_20260908_r1 --run-name
  a0_preflight_20260908_r1`
- Read-only A0 verifier: 1,375 frozen input hashes rechecked; 399 score rows,
  399 aligned return rows, all no-live/no-DB status flags and A0 block status
  asserted.
- `py -3 -B -m ruff check harness/tools/r88_trio_score_fusion_preflight.py`
  -> passed.
- `py -3 -B -m py_compile harness/tools/r88_trio_score_fusion_preflight.py`
  -> passed.
- `py -3 -B -m pytest -q tests/test_r88_trio_score_fusion_b1.py` -> `3 passed`.
- `py -3 -B -m ruff check harness/tools/r88_trio_score_fusion_b1.py
  tests/test_r88_trio_score_fusion_b1.py` -> passed.
- B1 generic Regsim identity under the explicit live50 input profile completed
  with `blocked_nonparity`; no fusion score or fusion backtest was allowed.
- Corrected-window read-only verifier: expected/generic/aligned `399/399/399`,
  maximum absolute return difference `0.0024989529687471004` on `2026-08-12`.
- B2 completed four generic replays. The aligned output has 399 unique dates
  per strategy; every retained day has 20 positions, full weight, and zero
  cash weight.
- `py -3 -m pytest tests/test_r88_trio_score_fusion_b1.py
  tests/test_r88_trio_score_fusion_b2_current_data.py -q` -> `6 passed`.
- `py -3 -m py_compile harness/tools/r88_trio_score_fusion_b2_current_data.py`
  and `py -3 -m ruff check ...` -> passed.

## Artifacts

- Completed A0 output:
  `D:/cbond_on/research_scratch/r88_trio_score_fusion_20260908_r1/a0_preflight_20260908_r1/`
  - `study_contract.json`
  - `input_manifest.json`
  - `a0_daily_score_universe.csv`
  - `a0_common_return_alignment.csv`
  - `a0_score_calendar_missing_by_model.json`
  - `a0_return_calendar_audit.json`
  - `run_status.json`
- B1 stopped output:
  `D:/cbond_on/research_scratch/r88_trio_score_fusion_20260908_r1/b1_equal_rank_neutral_20260908_r1/`
  - `run_manifest.json`
  - `identity_regsim_parity.csv`
  - `generic_backtests/.../r88_trio_b1_regsim_identity/.../`
  - `run_status.json` (`blocked_nonparity`)
- B2 completed output:
  `D:/cbond_on/research_scratch/r88_trio_score_fusion_20260908_r1/b2_current_data_equal_rank_20260908_r1/`
  - `run_manifest.json`, `run_status.json`, `RESULTS.md`
  - `summary_metrics.csv`, `paired_vs_regsim.csv`,
    `current_data_aligned_returns.csv`, `daily_fusion_coverage.csv`
  - four complete generic-backtest outputs under `generic_backtests/`

## Open Risks

- A rank fusion cannot use the score-file intersection as a replacement for the
  existing `o_0005` universe.
- B2 is an as-run current-data baseline, not a byte-for-byte replay of frozen
  historical execution: the raw/pool/TWAP inputs are mutable and no immutable
  as-of snapshot remains.
- P6/P3 generic outputs have 401 raw days before common-date restriction;
  Regsim and fusion have 399. This does not alter aligned 399-day metrics, but
  reporting must state the common-date alignment.
- Neutral fill is material: mean target-pool fills are 11.63 codes each for
  P6/P3 and 47.56 for Regsim. It is part of the fixed strategy definition.
- The 2026 reporting period is already observed and cannot be reused to tune a
  later variant.

## Next Action

- Await owner direction. Static-weight search, dynamic weights, sleeve
  allocation, retraining, and all live actions remain unapproved.

## Handoff Summary

- B2 establishes the requested current-data unified baseline. It is complete,
  research-only, and does not authorize a live change.
