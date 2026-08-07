# Live 50-factor cutover — 2026-08-05

## Objective

Make the 2026-08-06 and subsequent CBOND_ON live runs use the frozen
`live-27 + screened-23` factor set (50 model features) across Regsim,
baseline, HL20, labeltop20, and the rank-average Ensemble.  Preserve the
already completed 2026-08-05 run and verify both base-feature identity and
end-to-end timing before the controlled scheduler cutover.

## Owner-approved scope

- Do not rewrite today's live output (`2026-08-06` target), its DB partition,
  or existing 27-factor states.
- Productionize the 23 frozen r5 candidates, use a Rust live path, and use
  new versioned model ids/state/score roots.
- Change the next live run to 50 factors only after an isolated no-DB
  end-to-end verification proves the original 27 columns unchanged and
  records stage/full-runtime timing.
- Preserve strategy, `o_0005`, neutralization, DB table/mode, and selector
  semantics; do not introduce a new `run/` entrypoint.

## Frozen baseline

- Current live factor pack: `live_screened_no_winsor_27.json5`, SHA-256
  `1862929DFEE680B0810964152B490EC94E30B9E0FF8E1397AD0331AF7DF3AD32`.
- Current live config SHA-256:
  `71F98A23048F68BB44BA6BA0AAE7D598F8A0D34C6B2863BCFF89E7263F00DFEA`.
- 2026-08-05 live FactorStore file:
  `D:/cbond_on/factor_data/factors/T1430/2026-08/20260805.parquet`,
  SHA-256 `52E2BBD2BBCB2CA66E95AB8776402FED6785716052A5F9E1F1101E6F6B8C3992`.
- Completed target output:
  `D:/cbond_on/results/live/2026-08-06/trade_list.csv`, SHA-256
  `44C58284E6B76C7C94535C903F4A4B7B1E2F2733F095D55AEC01867660BD005F`.
- Scheduler was healthy after the completed run; no current-day rerun is
  permitted.

## Open risks

- All 23 selected candidates are currently research-only and require 13 Rust
  kernels plus production contracts and registry admission.
- Existing FactorStore has 27 features only; new 50-feature model states must
  be independent.
- The r5 uplift is a selection-leakage diagnostic.  The owner has nevertheless
  explicitly requested the controlled next-day production migration.

## Verification gates

1. Rust/Python factor parity and production registry/contract checks pass.
2. Isolated 50-factor build on the completed score day has exactly the frozen
   `(dt, code)` universe and byte/value-identical original 27 factor columns.
3. All 23 new columns exist, have no Inf, and meet documented missing-value
   semantics.
4. Isolated no-DB full live-chain replay records per-stage and wall-clock time
   without writing to `results/live`, production DB, current FactorStore, or
   existing model state.
5. Controlled cutover preserves the old 27 configuration/state paths for
   rollback and verifies the restarted scheduler's PID, config fingerprint,
   and next-day loaded 50-factor set.

## Completed cutover (2026-08-05 19:23 CST)

- The live configuration now resolves the isolated 50-factor inputs:
  data/paths_live50_20260805, live/live_factors_50_20260805, and
  versioned 50-feature Regsim/baseline/HL20/labeltop20/Ensemble state and
  score roots.  Strategy, o_0005, neutralization, DB table/mode, schedule,
  and TWAP settings remain unchanged.
- The new scheduler PID is 28756 (Dashboard reports start time
  2026-08-05T19:23:06).  Its state is idle_after_run, with
  last_target_run=2026-08-06; it will not rerun or overwrite the already
  completed 2026-08-06 target.  The next scheduled execution is the
  2026-08-06 14:29 run for the 2026-08-07 target.
- The protected completed artifacts remain unchanged:
  trade_list.csv SHA-256
  44C58284E6B76C7C94535C903F4A4B7B1E2F2733F095D55AEC01867660BD005F,
  and the original 2026-08-05 27-factor Store SHA-256
  52E2BBD2BBCB2CA66E95AB8776402FED6785716052A5F9E1F1101E6F6B8C3992.
- The isolated no-DB full-chain replay passed for score days 2026-08-03,
  2026-08-04, and 2026-08-05.  The most representative latest run took
  84.015s wall-clock: factor build 28.464s, model-switch 43.558s, and
  strategy selection 0.062s.  Individual model-score timings are nested
  within the selector timing and therefore are not additive to the wall
  time.  The measured 2026-08-04 run was 85.941s.
- FactorStore verification covered 626 matching trade-day files / 269,866
  rows.  Every live50 file has the same 50-column order and the identical
  (dt, code) index.  For every original current-pack factor where the legacy
  Store contains it (624--626 days by factor), values including NaNs are
  exactly identical; all 27 current factors are present in every new file;
  all 23 added columns have zero Inf values.  Legacy Store files have
  historical extra-column schemas (19/27/121/137/202/230/310), which are not
  part of the frozen 27-factor live pack and do not indicate a parity failure.
