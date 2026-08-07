# Rust-first unified 50-factor contract — 2026-08-06

## Objective

Make normal CBOND_ON factor execution Rust-only and make the production live
chain one ordered 50-factor contract.  The historical origin of a feature must
not create a second compute API, fallback, FactorStore, model feature path, or
model-switch source path.

## Risk Level

- High: factor admission and model feature admission are production boundaries.

## Current Verified Facts

- `live/live_config` resolves to `live/live_factors_50_20260805` and the
  Regsim 50-feature model source.
- The active pack contains exactly 50 unique ordered output columns, all with
  exact `live50_r5/...` Rust contract IDs.  The loaded extension declares ABI
  `rust_factor_contracts_20260806_r1`, `compute_factor_frame`,
  `python_fallback=false`, and all 50 contracts.
- Live factor admission loads/validates the full metadata surface and one
  ordered 50-feature contract.  The normal factor pipeline calls only
  `build_factor_frame_rust` and rejects Python/hybrid fields before panel or
  FactorStore I/O.
- The live-50 FactorStore can only be written with an admission-issued permit;
  generic batch/CLI use is rejected.
- Primary Regsim and each model-switch score source are now required to resolve
  to the same ordered 50-feature model contract.  A stale 19/27-feature model
  source fails before scoring.
- AI Factor Factory emits Rust drafts and an exact contract requirement only;
  legacy Python payloads are diagnostic-only and fail review.  A factor family
  is ideation, not an executable candidate.
- No scheduler restart, DB write, live rerun, model-state change, or live
  artifact mutation was performed for this closure.  The existing target
  `2026-08-07` output remains the scheduler's completed run.

## Files Changed

- Rust-first app/workflow/pipeline policy and exact capability admission.
- Live 50-factor factor/model admission and protected FactorStore permit.
- Rust factor engine contracts/manifest and 50-feature live configs/model
  contracts.
- AI Factor Factory Rust-only draft, schema, docs, and tests.
- Factor-development and README documentation.

## Commands Run

- `py -3.11 -m pytest` targeted Rust/live/AI policy suites:
  `51 passed, 113 skipped` (skipped tests require the intentionally isolated
  shadow wheel site).
- Live admission/store/path focused recheck: `18 passed`.
- `cargo fmt --check` and `cargo test --locked` with `PYO3_PYTHON` set to the
  project Python 3.11: all Rust tests passed.
- `py -3.11 -m cbond_on.common.architecture_guard`: passed.
- Read-only live config/model/capability handshake: passed.

## Artifacts

- No new live, DB, FactorStore, or model-state artifacts were created.
- Existing isolated parity/reference artifacts remain under
  `D:/cbond_on/research_scratch` and are not normal execution inputs.

## Open Risks

- The long-lived scheduler PID was started before the newest admission code;
  Python does not hot-reload it.  Its current live configuration was already
  the compatible single 50-factor profile, so no restart is required to retain
  the active factor set.  A separately owner-approved controlled restart is
  required only to load the newest fail-closed guards into that resident
  process.
- Historical Python formula modules remain isolated parity/reference material.
  They cannot execute through normal config, batch, FactorStore, model, live,
  or AI-factory candidate paths.  Any future executable experiment needs a
  compiled Rust kernel and loaded exact capability contract.

## Next Action

- Keep the existing scheduler and completed 2026-08-07 live output untouched.
- At an owner-approved maintenance window, restart the scheduler and verify
  PID/config fingerprint/loaded live50 admission before the next run.

## Handoff Summary

The normal execution architecture is now Rust-first and unified at the 50
feature contract.  There is no operational old/new factor branch: all live
features share one Rust computation call, Store, admission, model feature
order, and model-switch validation.
