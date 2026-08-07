# Rust23 shadow migration — 2026-08-06

## Objective

Implement all frozen live50 screened-23 factor kernels behind the opt-in
`compute_live23_factor_frame` shadow API, prove Python/Rust exact parity from
an isolated wheel, and leave every active live route unchanged.

## Risk Level

- High: the 23 columns are live50 inputs.  The completed work remains source
  and scratch-wheel only; it has no production dispatch, configuration, DB,
  scheduler, model-state, or result-artifact change.

## Current Verified Facts

- P1 (7), information (4), paths (4), rank/state (2), cross-asset (2), and
  intraday (4) all have isolated Rust kernels: 23/23.  The opt-in API now
  covers all 23, including QED3 + LRD1, through the separately built r14
  scratch wheel only.
- The final r10 isolated wheel passed all five daily Python shadow suites:
  `74 passed` (including all 20 random seeds for the newest four factors).
  Each numerical suite now compares finite `float64` values by their
  `uint64` IEEE bit view; NaN and finite masks, index, column order, and
  dtypes are independently exact.
- The final r14 Python shadow regression passed `114 passed`: the previous
  daily 74 plus 40 intraday checks.  Every finite value is compared through
  its `float64.uint64` bit view, with separate exact index/column/dtype and
  NaN/finite-mask assertions.  The intraday suite includes 20 random seeds,
  QED missing-field all-NaN, LRD missing-book KeyError, per-spec labelled vs
  physical key unions, local wall-clock/timezone comparisons, sub-microsecond
  cutoff, QED lunch adjacency/duplicate/reset, LRD lunch/nonfinite/full-book,
  and missing-`trade_time` precedence over both invalid and multi-date
  build-day cases.
- The r14 Cargo suite passed 46 tests (including the now-linked 7 intraday
  unit tests). The separately compiled r13 intraday executable also passed
  7/7.
- The adapter materialises Pandas' precise labelled and physical score-day
  booleans before the Rust boundary.  This preserves tz-aware versus naive
  Timestamp non-equality rather than substituting a local date-string rule.
- R14 preserves the LRD reference's `ensure_trade_time` error precedence
  before score-day resolution and evaluates QED's centered square through
  NumPy `power`, exactly matching Python `(location - center) ** 2`.
- The four rank/state and cross-asset signals preserve per-spec source contracts and strict-exchange
  aliases.  BSSRC now matches Python's inner merge before tail selection and
  validates global inconsistent-underlying input before an invalid output code
  can return NaN.  Empty output universes skip daily-source parsing as Python
  does; legal `params.signal` whitespace is stripped before dispatch.
- The active `pipeline.py`, `rust_python_hybrid.py`, and
  `live_config.json5` have no `rust_live23_shadow` or
  `compute_live23_factor_frame` reference.  The production extension SHA-256
  remained `053A8F740B6F5FFA6002A5D8305EDA781E7D564167FFB917970005B5F8D08ABC`
  after final r14 verification.

## Files Read

- `AGENTS.md`, `harness/README.md`, `harness/context/source_of_truth.md`,
  `harness/workflows/long_task_context.md`, and the long-task skill/template.
- Every Python reference kernel for the completed daily groups, plus QED/LRD
  source-contract and time-semantics audit; corresponding shadow Rust source
  and tests.

## Files Changed

- `rust/factor_engine/src/lib.rs`
- `rust/factor_engine/src/live23_shadow.rs`
- `rust/factor_engine/src/live23_daily_cross_asset.rs`
- `rust/factor_engine/tests/live23_daily_cross_asset.rs`
- `rust/factor_engine/src/live23_intraday.rs`
- `cbond_on/infra/factors/rust_live23_shadow.py`
- `tests/test_rust_live23_shadow_adapter.py`
- `tests/test_rust_live23_shadow_daily_p1.py`
- `tests/test_rust_live23_shadow_daily_information.py`
- `tests/test_rust_live23_shadow_daily_paths.py`
- `tests/test_rust_live23_shadow_daily_rank_state_cross_asset.py`
- `tests/test_rust_live23_shadow_intraday.py`

## Commands Run

- `py harness/tools/agent_preflight.py --mode long-task`
- isolated `maturin build --release --locked` plus `pip --target` for r8-r14
  (every build has its own `CARGO_TARGET_DIR` and target site)
- six-file Python shadow regressions; final r14 result `114 passed`
- `cargo fmt --check`, `cargo test --locked` (r14: 46 passed), and isolated
  standalone `live23_intraday.rs` test (r13: 7 passed)
- closing re-run against the r14 isolated site: six Python shadow suites
  `114 passed`; `cargo fmt --check`, `cargo test --locked` `46 passed`,
  `git diff --check`, production extension hash, and active-route reference
  audit all passed (Cargo used the r14 scratch target and Python 3.11).

## Artifacts

- r8 regression evidence, including the pre-fix BSSRC tail failure:
  `D:/cbond_on/research_scratch/rust23_shadow_20260806/daily_rank_cross_r8_20260806_20260806_134600_1859`
- r9 intermediate evidence, including the empty-index dtype mismatch:
  `D:/cbond_on/research_scratch/rust23_shadow_20260806/daily_rank_cross_r9_final_20260806_20260806_140257_3689`
- final r10 isolated wheel/site/cargo target and intraday executable:
  `D:/cbond_on/research_scratch/rust23_shadow_20260806/daily_rank_cross_r10_20260806_20260806_140625_8854`
- r11 first intraday API regression evidence (36-test suite: 34 passed / 2
  failed; timezone contract and fixture precision findings retained):
  `D:/cbond_on/research_scratch/rust23_shadow_20260806/intraday_r11_20260806_20260806_142809_580`
- r12 intermediate passing intraday wheel/site/cargo target (36 passed; later
  superseded by the stricter labelled-timezone contract):
  `D:/cbond_on/research_scratch/rust23_shadow_20260806/intraday_r12_20260806_20260806_143435_501`
- final r13 isolated wheel/site/cargo target and standalone executable:
  `D:/cbond_on/research_scratch/rust23_shadow_20260806/intraday_r13_20260806_20260806_143904_413`
- final r14 isolated wheel/site/cargo target, including the LRD error-order
  and QED `np.power` parity fixes:
  `D:/cbond_on/research_scratch/rust23_shadow_20260806/intraday_r14_20260806_20260806_145425_466`

## Open Risks

- Invalid/NaT `panel.attrs["__build_day__"]` still follows the pre-existing
  adapter fallback rather than Python's fail-closed exception.  It is an
  out-of-scope abnormal-input behaviour and was deliberately not changed.
- Exact parity is proven over the listed deterministic, random, source
  contract, error, empty-output, and intraday time fixtures, not yet over
  historical FactorStore data.  Do not activate the shadow route without a
  separate owner decision and an approved historical/performance validation
  plan.

## Next Action

- Await an owner-approved historical/performance validation plan or an explicit
  activation decision.  Neither production routing nor the production `.pyd`
  may be changed implicitly.

## Handoff Summary

All 23 frozen live50 factors now have literal Python/Rust parity in the opt-in
shadow API under r14.  Earlier r8/r9/r11 evidence and r12's superseded
intermediate wheel were retained; no scratch artifact was copied into
`cbond_on_rust/`.  No live state was touched and activation remains out of
scope.
