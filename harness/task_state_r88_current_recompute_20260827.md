# R88 current-contract full recompute and unified LGBM replay

## Objective

Create one new, research-only R88 materialisation by recomputing all 88 exact Rust
contracts over the current DataHub snapshot calendar, then rerun the fixed
factor-set × LGBM study on that one data version.  No live configuration,
production FactorStore, DB, scheduler, model state, or result root is in
scope.

## Locked contracts

- Factor profile: `research_r88_rust88_20260825`, 88 ordered factor IDs,
  `specs_sha256=e8f7ab3bbbd15e300876c49b9751f5438a28d98bf12d6020f6db101c6f5e6d44`.
- Compute: current isolated Rust wheel only, Rust-first CPU pipeline; no Python
  fallback.
- Calendar: DataHub raw cbond calendar must equal clean cbond and clean stock
  snapshot inventories for `2024-01-03..2026-08-27`, exactly 642 days.
- Panel construction: `clean_direct`, matching the active live factor runtime.
  Each score day builds one in-memory cbond/stock T1430 panel from clean
  snapshots with four-day lookback, 5,000 points per code, and strict 14:29
  physical cutoff. No persistent PanelStore is read or written.
- Admission: retain 66-of-88 availability.  The full-current route never
  reads, merges, or uses the frozen R50 FactorStore as data or calendar.
- Model study: rolling daily refit, fixed execution/mask/cost/benchmark;
  2025 is the only selection period and 2026 is report-only.

## Current facts

- The legacy R88 root had a frozen-R50 coverage break on 2026-06-11/12.
- Current-contract smoke on 2026-06-11 wrote 88 ordered columns, zero Inf,
  and 308/317 rows met 66/88.  The manifest records isolated wheel extension
  hash `72312b9b967d3b70c5e40f42428cc9456e0dd061d488dee15e58ea201a2dc974`.
- The isolated wheel is below
  `D:/cbond_on/research_scratch/r88_rust_wheel_20260827_r12/site` and advertises
  exactly 88 contracts.
- The full-current launcher now has a no-write preflight that binds the exact
  clean-direct panel contract and rejects a non-`clean_direct` source.
- The partial current-contract factor scratch root remains at
  `D:/cbond_on/research_scratch/r88_full_current_20260827_r1` with 39 durable
  factor partitions and no completion manifest.  Its worker and downstream
  study watcher were stopped after the shared-panel boundary was clarified.

## Open risks

- The full manifest must be `completed_rust88_current_recompute` with all 642
  durable experiment-table commits, identical DataHub calendars, matching
  file/index hashes, direct DataHub manifest/.done evidence, and
  `no_frozen_factorstore_read=true` before any model run is admitted.
- The follow-on study must remain labelled
  `exploratory_conditioned_on_preexisting_r88_screen`; it is not live evidence.

## Next action

Owner confirmation is recorded. Run the fresh R88 clean-direct recompute into
`experiment/generations/r88_clean_direct_20260828`, verify its 642 committed
days, then atomically activate that verified experiment version. Only after a
fresh normal reader admits the active version may the fixed 30-candidate
rolling LGBM research matrix begin in its new scratch study root.

## Execution status (2026-08-28)

- Full preflight passed against the exact 642-day DataHub clean calendar:
  `2024-01-03..2026-08-27`, calendar SHA-256
  `a7a0e9219a87a7b582952c6cf3023f3f833de4112650560cd51a85d270b4facd`.
- Full repository regression passed before launch: `1069 passed, 117 skipped`.
- The approved single-worker research-only recompute started with generation
  `r88_clean_direct_20260828`.  It may write only below the experiment table's
  internal generation and its declared scratch root; it cannot activate the
  table, touch live, write a DB, or restart a scheduler.
- Activation and the 30-candidate study remain pending the completed 642-day
  generation's manifest, `.done`, exact coverage checks, and a fresh normal
  experiment-table reader admission.

## Completion and next research stage (2026-08-31)

- The clean-direct generation completed all `642/642` committed score days
  (`2024-01-03..2026-08-27`), with one 88-column current-Catalog contract,
  no Inf cells, and a finalized `generation.done`.  The full DataHub calendar
  hash is `a7a0e9219a87a7b582952c6cf3023f3f833de4112650560cd51a85d270b4facd`.
- Fresh canonical-reader admission passed after atomic activation of
  `r88_clean_direct_20260828`; the reader is pinned to the generation-local
  manifest and resolves 642 days.  No live table, DB, or scheduler was
  modified.
- The immutable research plan is prepared below
  `D:/cbond_on/research_scratch/r88_joint_factor_lgbm_20260828_r1/runs/rolling_20250102_20260827`.
  It contains exactly 30 warm-start candidates (five frozen factor masks by
  six LGBM profiles), ranks only `2025-01-02..2025-12-31`, and reports
  `2026-01-01..2026-08-27` without using that period for selection.
- Serial `run-all` is running research-only.  It creates scores, backtests,
  warm-start audits, coverage audits, and metrics only below that run root.
