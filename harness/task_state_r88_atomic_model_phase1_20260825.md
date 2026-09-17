# Task State: R88 Atomic-Model Phase 1 (2026-08-25)

## Objective

- Build a research-only, Rust-first 88-factor input contract and use it to
  test standalone cross-sectional atomic models for higher and more stable
  rolling OOS Sharpe.

## Locked Contract

- Factor universe: `R88 = legacy27 + screened61`, with all 88 names retained
  even where the owner permits correlated or economically overlapping inputs.
- Model inputs: each atomic model may select any subset of R88, but may not
  introduce a factor outside R88.
- No ensemble, BaseGap, or model-switching logic belongs to this phase.
- Keep the existing strategy rule, T-1 `o_0005` universe, label/sell window,
  costs, neutralization mode, and benchmark unchanged.
- Research outputs are isolated below `D:/cbond_on/research_scratch`.
- Live factor packs, live configs, DB, scheduler, live FactorStore, live
  model state, and live score/result roots are out of scope.

## Current Verified Facts

- The current live chain has one Rust-first 50-factor contract: legacy27 plus
  23 of the screened61 factors.
- The screened research result contains 61 selected factor instances over 381
  score days (`2025-01-02..2026-07-30`); 38 of them are not in live50.
- R88 therefore contains 88 unique factor instances.
- The R88 profile now has 88 exact Rust contracts: 50 frozen `live50_r5/*`
  instances and 38 research `research_r88_20260825/*` instances. Its exact
  `specs_sha256` is
  `e8f7ab3bbbd15e300876c49b9751f5438a28d98bf12d6020f6db101c6f5e6d44`.
- The isolated wheel at
  `D:/cbond_on/research_scratch/r88_rust_wheel_20260825_r3/site` advertises
  exactly those 88 contracts, uses `compute_factor_frame`, and declares
  `python_fallback=false`. It is not the local/live `.pyd`.
- The `2026-07-30` R88 scratch smoke completed at
  `D:/cbond_on/research_scratch/r88_factor_backfill_smoke_20260825_r2`:
  302 rows, exact R50/R38/R88 index equality, 38 new columns and 88 final
  columns in profile order, unchanged R50 values, no Inf, and no nonempty
  constant final factor column. 274/302 rows met the explicit 66-of-88
  availability gate.
- R38 dispatch now uses score-day caches for all remaining signals and direct
  intraday metric bundles. Source validation passed: `cargo test` 82 passed;
  isolated-wheel R88 Python checks 23 passed.
- Final research wheel policy: R38-only cache paths are limited to the two
  research BSSRC tail signals; the existing live50 BSSRC correlation retains
  its direct execution path. The current source validation after this boundary
  is `cargo test` 94 passed and the focused R88 Python suite 41 passed.
- Current clean snapshot coverage audit found 134 source-zero-row code-days
  across 97 of 622 score days (0.0499% of 268,655 frozen R50 code-days). All
  have at most 44 valid frozen R50 factors, below the explicit 66-of-88
  admission floor. The R88 builder retains those R50 rows and writes the 38
  non-reconstructible R38 values as NaN with a per-day ledger record.
- Full backfill is running at
  `D:/cbond_on/research_scratch/r88_factor_backfill_20260825_r2` using the
  isolated r11 wheel. It has no completion manifest yet, so every model entry
  remains fail-closed.
- DataHub raw and clean inputs physically cover the 2024 onward period, but
  the existing research-factor launcher is intentionally restricted to 2025
  onward and cannot be reused unchanged for a 2024 backfill.

## Open Risks

- A completed full-history R88 FactorStore does not exist yet. It must use a
  fresh scratch root, then be audited against the frozen R50 source before
  model input.
- The current 61-factor research evidence is not an independent 2024 rolling
  input history. A fresh, versioned backfill is required before aligned model
  comparison.
- The current local `cbond_on_rust` binary may be used by live processes. Any
  research build must load an isolated scratch wheel and must never replace
  the live/local `.pyd`.
- Several historical factor-screen sources admitted physical observations
  through 14:30. The new R88 research contract uses a strict physical 14:29
  cutoff for newly ported intraday factors. Their old 61-factor IC result is
  therefore provenance for candidate names, not proof that their values or IC
  reproduce under the stricter contract; R88 must be freshly re-screened.

## Next Action

- Monitor the full fresh R88 backfill across frozen R50 coverage. Audit its
  completed manifest and all FactorStore days before model input.
- First model baseline must be Full88 or a pre-defined no-label family slice.
  A full-period static factor screen may describe factor quality, but cannot
  be used to select a fixed subset and then claim 2024 OOS; later
  result-driven selection needs a nested rolling selector or an explicit
  train-cutoff/OOS split.
- Prepared-but-unexecuted Full88 and screen configurations live below
  `D:/cbond_on/research_scratch/r88_full88_rolling_20260825_r1/configs` and
  `D:/cbond_on/research_scratch/r88_screen_20260825_r1/configs`.
