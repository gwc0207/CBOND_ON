# Task State: factor-governance live migration

## Objective

Land the full factor-governance target state:

- register the complete factor universe at instance level;
- register shared computation under the operator layer;
- retain the frozen live50 formula, ordered feature contract, Rust capability,
  model id, model state, neutralization, universe, output target and DB mode;
- compute only the admitted live release during the live run;
- add a separate post-close research supplement path for all other registered
  factor instances.

## Current verified live baseline

- live config: `live_config.json5`, SHA-256
  `b8f530062d8fbb377172d5ba66288d06b9c6cd6eb75b7dc0cdb4e5ba60356b93`;
- live factor config: `live/live_factors_50_20260805`, SHA-256
  `a71de8d3ea6ab5644d501ae1f3375b1641274258b887b4a07c245b218cca2f60`;
- factor pack: frozen ordered Rust-50 pack, SHA-256
  `7f635b3f3466d9ba6731eae5849f9ca40bd8773eb415e3fb40226dbd8572538b`;
- live factor contract profile: `live50_rust50_20260806`, SHA-256
  `cbce537e7ba3e63cb3ddbac5ff6e429843fc661760e6a2ca054e849c90e68e43`;
- model id: `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_50_20260805`;
- model state, neutralization, universe, output DB target and current model
  selection configuration remain unchanged;
- historical scheduler PID `35112` was the pre-migration daemon;
- production DB writes remain disabled for every verification in this task.

## Scope and safety boundary

This is high risk because it changes factor registration and adds a scheduled
post-close task. The production live behavior may change only after:

1. exact live50 resolver/admission parity passes in a fresh process;
2. a no-DB live dry-run completes with the frozen 50 columns and no output
   path outside the scratch root;
3. the diff and generated release binding are inspected;
4. the separate 23:59 task is independently tested and proven not to invoke
   `live_runtime`, write the live FactorStore, write score/trade outputs, or
   write the production DB.

No normal scheduler restart, live DB write, model-state deletion, or live
artifact deletion is in scope.

## Planned implementation

1. Build an immutable full Factor Catalog and Operator Catalog from the frozen
   research catalog plus existing live-only instances.
2. Generate one thin definition/contract package per registered factor and
   provide a catalog resolver.
3. Preserve old runtime APIs behind a compatibility adapter while moving new
   code to the operator vocabulary.
4. Make live admission resolve only its frozen release instances, preserving
   all existing hash/capability/permit checks.
5. Add an independent 23:59 research supplement runner, lock, ledger and task
   registration script. It remains non-live and does not change the normal
   scheduler.

## Verification plan

- catalog uniqueness, contract completeness and source mapping tests;
- fresh-process import probe: live50 only resolves its release dependencies;
- existing live50 admission, Rust contract and FactorStore permit tests;
- no-DB dry-run under a new scratch root;
- supplement dry-run, lock, manifest/.done gate and no-live-side-effect tests;
- `git diff --check` and explicit review of all protected-path changes.

## Completed implementation and evidence

- Canonical Factor Catalog: 800 instances (`research773=773`,
  `legacy_live27=27`), 266 registered operators, 174 families, and one
  explicit historical-name override. Every instance has a thin callable
  definition and contract under `factor_engine/factors/`.
- Live binding: `factor_engine/releases/live/live50_rust50_20260806.json`
  pins the existing ordered 50 instances to identity/version/contract/operator
  and Rust contract. The active live factor config now declares that immutable
  release; its frozen pack/profile/model feature order remain unchanged.
- Fresh-process live admission resolves the immutable release and only imports
  the required operator modules. It does not invoke the legacy 194-operator
  compatibility loader.
- Research supplement: `CBOND_ON_FactorSupplementV1` was registered as an
  independent Windows task at 23:59 in `execute` mode. It has `IgnoreNew`,
  `StartWhenAvailable=false`, an 8-hour limit and scratch-only stdout/stderr.
  It does not use `liveLaunch.scheduler`.
- Real supplement validation for score day 2026-08-25 completed 750/750
  non-live factors in `D:/cbond_on/research_scratch/factor_supplement_v1`.
  The latest execution manifest records zero missing columns, zero Inf values
  and 27 all-NaN coverage warnings; it is therefore completed with coverage
  gaps rather than silently labelled fully healthy.
- Isolated live50 factor-stage validation completed for 2026-08-25 below
  `D:/cbond_on/research_scratch/live50_stage_verify_20260826`: exact 50-column
  order, release ID and Rust output were verified; no live runtime, model,
  trade list or DB path was called.
- The normal scheduler was restarted only in the 01:29 non-trading window.
  Its new PID is `38196`; state is `waiting_cutoff` for target 2026-08-27.
  No live run was triggered and no production DB write occurred.

## Final verification

- focused governance/admission/supplement/Rust-contract suite: 76 passed,
  1 skipped;
- architecture guard: passed;
- catalog generator check: 800 factors, 266 operators, 50 live-released;
- `git diff --check`: passed.

## Operator-source migration follow-up (2026-08-26)

- A new canonical runtime source tree now exists at
  `cbond_on/domain/factors/operators/` with 290 Python modules: 263 operator
  implementation modules plus the required shared/aggregate support modules.
  A fresh-process scan registers the same 266 operator keys as the prior tree.
- `factor_engine/operators/<operator_id>/` now has one generated
  `definition.py` and `contract.json` for each of the 266 registered
  operators. The contracts pin runtime path, source hash, migration provenance
  and explicit audit-status fields; the 800 factor contracts are regenerated
  against these operator contracts.
- The active factor admission is now pinned to
  `live50_rust50_operator_source_20260826`. The old
  `live50_rust50_20260806` release is preserved under
  `factor_engine/releases/history/`; the Rust profile name remains unchanged.
- No-DB full-chain replay under the active migrated admission completed in
  `D:/cbond_on/research_scratch/catalog_live50_operator_source_fullchain_20260826_r3`.
  It was deterministic against the immediately preceding migrated replay:
  309x50 frame, columns, index, NaN mask and all values equal; trade-list hash
  `656bd2f895f721262763847c2799479d4b7e7f46da4dd11b8613695ba4bb790b`.
  A difference from the earlier baseline was isolated to a later DataHub
  `daily_price` source revision, not the source-path migration.
- The production scheduler completed its pre-cutover 2026-08-26 run successfully
  (target 2026-08-27) before the active release was switched. It is now
  `idle_after_run`, PID `38196`, and has not been restarted.

## Completed operator-source cutover

- Owner-approved scheduler restart completed in the idle window: pre-cutover
  PID `38196` exited and new PID `21332` started with
  `pythonw -m liveLaunch.scheduler`. The new process is `idle_after_run` for
  the already-completed target `2026-08-27`; its restart created no new live
  attempt, trade list, decision, or DB write.
- Fresh-process no-DB full-chain replay completed under the active release in
  `D:/cbond_on/research_scratch/catalog_live50_operator_source_fullchain_20260826_r4`.
  It matches the preceding current-input replay exactly at the 309x50 factor
  frame and final trade-list SHA-256.
- The retired source root `cbond_on/domain/factors/defs/` was deleted only
  after replacement count=290 and active-reference count=0 checks passed.
  Historical source-path evidence remains in the migration manifest and old
  release history; no active runtime/code/config/test imports that namespace.
- Final validation: focused factor governance/admission/Rust/supplement suite
  `54 passed`; catalog check `800 factors / 266 operators / 50 live released`;
  architecture guard and diff whitespace check passed. Generated Python caches
  below the factor roots were removed after validation.

## Known unrelated condition

`cbond_on/infra/factors/rust_python_hybrid.py` has a pre-existing duplicate
module docstring before its `from __future__` import, so broad `compileall` of
the entire historical factor directory reports a syntax error there. The file
is not modified by this task and is excluded from every normal Rust-first
runtime; all changed and active paths compile and their focused suites pass.
