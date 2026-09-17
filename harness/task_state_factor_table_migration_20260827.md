# Task State

## Objective

- Route every normal CBOND_ON factor reader and designated writer through the
  three canonical local tables under `D:/cbond_on/factor_store`.

## Risk Level

- High: canonical live-path cutover and scheduled supplement-writer change.
  The confirmed scope excludes DB writes, scheduler restart/execution, and
  deletion of historical roots.

## Current Verified Facts

- All canonical tables are `full_verified` and consumer-ready.
- Active live paths resolve the admitted `live` table at
  `D:/cbond_on/factor_store/live`.
- The standard live factor build is the sole canonical `live` writer.
- The 23:59 supplement reads canonical live, computes non-live factors only
  in ephemeral scratch staging, then publishes one full canonical library day.
- Normal R88 model research reads canonical `experiment`; direct historical
  factor roots are limited to migration/audit evidence.

## Files Changed

- Canonical reader/writer adapters, active/default paths, live runtime,
  Dashboard/readiness, 23:59 supplement, and R88 research paths.

## Commands Run

- `py -3 -B harness/tools/agent_preflight.py --mode live-change`
- Two fresh no-DB live full-chain replays using the canonical live input.
- Focused path/governance suite: `147 passed`.

## Artifacts

- Canonical live no-DB replays:
  `D:/cbond_on/research_scratch/canonical_live_factor_writer_dryrun_20260828_r1`
  and `_r2`.

## Open Risks

- Recomputing 2026-08-25 with current DataHub/current computation differs in
  six fields from its frozen historical materialization. The canonical store
  correctly rejected overwrite; no historical partition was changed.
- Historical roots remain for migration/audit and are not deletion-authorized.

## Next Action

- Let the next normal live and 23:59 supplement cycles publish new canonical
  days, then inspect their manifests and source parity. Do not restart either
  scheduler manually.

## Full-Repository Route Closure (2026-08-28)

### Completed

- Every normal `paths` profile now declares `factor_table`; profiles that retain
  a legacy root are explicit `lifecycle.status=audit_only` exceptions and
  cannot enter a normal consumer.
- External paths files passed through normal CLI/bootstrap loading are now
  validated by `load_paths_profile`; a filename/location can no longer bypass
  the canonical route contract.
- `build_factor_reader` rejects direct roots for normal use. The only direct
  reader exception is the environment-gated, fresh research-scratch no-DB
  verifier profile.
- Active live writing is bound to the exact active
  `data/paths_live50_20260805` profile and `D:/cbond_on/factor_store`; the
  prior candidate writer profile is now read-only.
- Canonical wide writers require a designated authority: admitted live or
  explicit experiment publisher. The 23:59 supplement remains the designated
  factor-library publisher.
- Normal direct FactorStore writes through the common pipeline fail closed;
  only permit-bound research staging or explicit no-DB scratch staging can
  reach the legacy implementation.
- Dashboard binds its paths profile to the active live config. The scheduler
  now fails closed if its paths profile changes after startup, preventing its
  calendar/journal roots from diverging from runtime computation.
- Added `cbond_on.common.factor_route_governance_guard` and the precise
  `harness/policies/factor_route_legacy_allowlist.json` policy. The guard
  validates config routes, direct constructor allowlists, legacy literals, and
  required governance documentation.

### Verification

- `py -3 -m cbond_on.common.factor_route_governance_guard`: passed.
- Focused route/canonical/live/R88/supplement/dashboard suite: passed.
- Focused route/canonical/live/R88/supplement/dashboard suites passed; no DB,
  scheduler execution/restart, or formal-table data write.
- No-DB full-chain preflight for `2026-08-27 -> 2026-08-28`: passed, produced
  no scratch directory and no writes.
- `py_compile` for all changed route entry points: passed.
- Full repository `pytest -q`: `1057 passed, 117 skipped` in 136.83 seconds.
  The 22 historical tests were migrated from the retired `defs` namespace to
  the current `operators` namespace; no retired source module was restored.

### Remaining Operational Check

- After the next normal live cycle and 23:59 supplement, perform read-only
  verification of new day parquet/manifest/.done, DataHub readiness evidence,
  scheduler journal, and absence of model/factor/DB drift. Do not execute or
  restart either scheduler without separate owner authorization.

### Runtime Incident Observed During Closure (2026-08-28 14:29)

- The already-running scheduler (PID `21332`, started `2026-08-26 14:59:34`)
  naturally reached its cutoff and exited failed before DataHub readiness.
- Attempt journal `results/live/scheduler/attempts/2026-08-28.jsonl` records
  `ValueError: read_only_input_roots ... missing ['factor_data_root']` at
  `ready_gate`; it used the scheduler process's pre-change imported config
  code, not the fresh canonical profile resolution.
- Fresh-process verification of the same active
  `paths_live50_20260805` profile resolves `factor_table.live` to
  `D:/cbond_on/factor_store/live` correctly and does not require a direct
  `read_only_input_roots.factor_data_root`.
- No trade-list, model-switch, allowlist, universe, factor-table, or DB output
  was produced for this failed attempt: `out_dir` is empty and the only files
  under `results/live/2026-08-28` are logs; the displayed trade artifacts are
  dated `2026-08-27`.
- DataHub published its 2026-08-28 clean manifest and `.done` at 14:34, after
  the configuration failure. The scheduler is now stopped. Recovery requires
  a separately authorized scheduler restart; do not restart automatically.

### Authorized No-DB Recovery (2026-08-28 14:46--14:49)

- Owner authorized a scheduler restart with today's DB write disabled. The new
  process received only the date-scoped environment
  `CBOND_ON_LIVE_NO_DB_DATE=2026-08-28`; the committed live config was not
  changed, so the override does not apply to the next trading day.
- Fresh scheduler attempt `20260828T144613_28828_f504e2f9` succeeded for target
  `2026-08-31`. It published the canonical live day `2026-08-28` as 311 rows,
  50 columns with parquet/day-manifest/`.done`, generated a 20-name trade list
  at `results/live/2026-08-31`, and returned to `idle_after_run`.
- `allowlist_summary.json` records `configured_db_write=true`,
  `effective_db_write=false`, `override_active=true`; the log records
  `skip output db write: date-scoped no-DB override active`. No production DB
  write was performed by this recovery run.

## Handoff Summary

- User confirmed path management cutover. No DB write, scheduler restart,
  manual scheduler execution, model/factor membership change, or deletion was
  performed.
