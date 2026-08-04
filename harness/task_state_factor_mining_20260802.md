# Task State

## Objective

- Build a research-only, factor-family-first candidate catalogue and screen for the first 100 factors with absolute mean daily Pearson IC above 0.02 under the fixed T-1 `o_0005` universe.

## 2026-08-03 active execution checkpoint (current)

- The immutable IC contract remains: requested start `2025-01-01` (first score
  day `2025-01-02`), T1430/14:30 factor, same-day 14:42 label, strict T-1
  `quant_factor_dev.researcher_xuvb.o_0005`, absolute mean daily Pearson IC
  strictly above `0.02`, at least 250 valid days, at least 50 valid days in
  each 60/20/20 chronological partition, 200 common redundancy days,
  within-family absolute correlation below `0.80`, cross-family below `0.70`,
  and at most 100 selected.  Existing masks and trading rules are unchanged.
- The latest complete baseline is v7: 535 signals / 77 families / 381 days,
  64 IC-and-validity eligible, 31 canonical-greedy selections, and an exact
  maximum-independent-set selection of 33 (summed absolute IC `0.873409`).
  This is a research baseline, not the requested 100-factor result.
- `research_factor_mining_aggregate_catalog_v3.py` is a new research-only
  healthy successor to aggregate v2.  It has 97 signals / 35 families: it
  removes the three cross-period persistent-empty modules, keeps the
  coverage-fractured debt-floor module isolated, and replaces the original
  state-gated source with its nine-signal healthy wrapper.  Focused tests,
  Ruff, and full-window no-write preflight passed.
- v3 has a completed strict-T1430 2026-04-28..2026-04-29 scratch smoke at
  `D:/cbond_on/research_scratch/factor_mining_20260803_aggregate_catalog_v3_smoke_20260428_20260429`.
  It has two date files (330/324 rows), stable 97-column schema, unique
  `(dt, code)`, zero Inf, and zero all-empty or cross-sectionally constant
  signals on either day.  It is smoke evidence only; no v3 full-window build
  has started yet.
- Fresh independently smoked additions currently ready for the next aggregate:
  `daily_price_base_relations_v1` (9/3),
  `cross_asset_book_event_transmission_v1` (9/3; only 95--118 finite names
  per smoke day, so retain the full-window coverage caveat),
  `historical_execution_to_open_transition_state_v1` (9/3),
  `intraday_trade_grid_topology_v1` (9/3), and the 7-signal / 3-family
  `daily_asymmetric_state_transitions_healthy_catalog_v1` wrapper.  All have
  no v7 signal/family name collision.  The original asymmetric module and the
  depth-dominance module are not admissible unchanged because their smoke
  revealed empty signals.
- Do not create a full build from partial, one-day, or non-manifest roots.
  Finish the remaining smoke/coverage evidence first, then compose one fresh
  aggregate successor and launch only that fresh isolated full root.  No live
  config, DB, scheduler, production FactorStore, mask, trading rule, or
  `defs.__init__` has changed.

## 2026-08-03 current continuation (supersedes older runtime snapshots below)

- The fixed contract is unchanged: requested IC start `2025-01-01` (first
  trading score day `2025-01-02`), T1430/14:30 factor, same-score-day 14:42
  label, strict T-1 `o_0005`, `abs(mean daily Pearson IC) > 0.02`, at least
  250 valid days and at least 50 per chronological partition, 200 common
  redundancy days, within-family correlation `<0.80`, cross-family `<0.70`,
  and max 100 selections.
- `orderbook_repricing_v1_full` completed naturally and passed a full
  read-only root audit: 381 date files (`2025-01-02..2026-07-30`), one
  research-only manifest and family catalogue, 155,910 rows, 6 signals / 2
  families, exact stable schema, unique `(dt, code)`, exact 14:30 timestamps,
  and zero Inf.  Its Python process is no longer running.
- Fresh v6 composition/outer-union merge/screen added that complete orderbook
  root to the v5 input base without filling source differences.  The canonical
  v6 greedy screen at
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v6_with_orderbook_screen_v1`
  covers 493 signals / 70 families / 381 days: 61 IC-and-validity eligible and
  28 greedily selected.  Two new orderbook factors survive all gates:
  `price_ladder_reprice_direction/lrd_cross_side_reprice_symmetry` (mean
  Pearson IC `+0.022386`) and
  `reprice_conditioned_depth_relocation/rdm_joint_reprice_depth_retention`
  (`-0.021022`).  The other four have full coverage but fail the 0.02 IC gate.
- A new research-only optimizer
  `harness/tools/optimize_factor_mining_selection.py` preserves every source
  screen rule and treats unavailable redundancy evidence as a fail-closed
  conflict.  Focused tests (`3 passed`) and Ruff passed.  On v6 it proves the
  maximum-cardinality, maximum-absolute-IC independent set is 30 (61 eligible,
  154 conflict edges, summed abs IC `0.797533`), versus the immutable greedy
  baseline's 28 / `0.739736`.  Its output is a separate scratch artifact, not
  a model/live admission or replacement of the canonical screen:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v6_with_orderbook_selection_optimal_v1`.
- `intraday_joint_state_v1_full` remains an actively writing, partial
  research scratch build and is ineligible for any merge/screen until it
  naturally produces all 381 files and its immutable manifest.  No third full
  build has been launched.
- `intraday_state_interaction_v1` is launch-ready but queued: its 36 signals /
  6 families have no name collision with v6 or the aggregate, its focused
  tests (`10 passed`), Ruff, and no-write full-root preflight passed, and its
  intended fresh root remains absent.  It must wait for the active joint-state
  build to release capacity.
- `structural_neighborhood_v1` (12 signals / 4 families) and
  `underlying_cohort_distribution_v1` (6 / 2) have passed focused tests
  together (`15 passed`), Ruff, and no-write preflights.  They still need real
  strict-PIT smoke evidence before either can be added to an aggregate/full
  build.  Their shared T-1 similar-stock neighborhood makes their economic
  correlation nontrivial despite no exact signal/family-name collision.
- Three research-only healthy-subset catalogues now isolate known-degenerate
  entries without changing their original modules: rank lattice exposes 9
  healthy barrier/optionality/liquidity signals and excludes its three
  floor-source-gap signals; queue state exposes 9 non-lock signals and excludes
  three cross-sectionally constant `lqls_*`; cross-asset event clock exposes 5
  signals and excludes near-zero `xca_lull_overlap_excess`.  Their focused
  tests, Ruff, and no-write preflights passed.  All three remain smoke-only
  candidates and are not yet imported by the aggregate or any live path.
- Current explicit aggregate source is 16 modules / 151 signals / 53 families
  after adding the tested research-only state-gated microstructure module,
  `daily_mark_barrier_dynamics_v1` (official-close versus completed execution
  TWAP, prior-only phase-conditioned final-mark residuals, and barrier
  occupancy hysteresis), and `daily_interday_topology_v1` (prior-only
  final-mark-to-next-opening transmission, stock-conditioned mark residuals,
  and call/put-floor payoff topology).  The first daily module's focused
  module/aggregate tests passed (`9 passed`); the second's passed (`10
  passed`); Ruff and diff checks passed.  The first module and its then-current
  aggregate passed no-write preflight; the current 151/53 aggregate has not
  run smoke or full build because joint-state is still consuming the resource
  gate.  None of these research modules is in `defs.__init__`, live config,
  model config, contracts, DB, scheduler, or production FactorStore.

## Risk Level

- medium: a long, compute-heavy research batch; all derived data must stay outside production runtime roots.

## Current Verified Facts

- **Current P1 checkpoint (research-only):** static and focused validation for
  the new queue-state and cross-asset event-clock modules passed together
  (`18 passed`; Ruff clean).  Neither module is imported into a production
  pack or config.
- Queue-state has a completed one-day smoke at
  `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_queue_state_v1_smoke_20260428`
  (329 unique `(dt, code)` rows, all 12 catalog columns, zero Inf) and a
  complete six-score-day smoke at
  `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_queue_state_v1_smoke_20250102_20250109`
  (six date files, one research-only manifest).  Nine queue/trade/depth
  signals are nonconstant with broad finite coverage in those samples.  The
  three `limit_queue_lock_state_machine` signals are cross-sectionally
  constant zero on all six early sample days and all-NaN on the 2026-04-28
  sample, so they have no credible path through the 250-day IC gate unless a
  new, distinct replacement hypothesis is designed; no threshold or fill was
  changed.
- Cross-asset event-clock has a complete one-day smoke at
  `D:/cbond_on/research_scratch/factor_mining_20260803_cross_asset_event_clock_v1_smoke_20260428`
  and a complete six-score-day smoke at
  `D:/cbond_on/research_scratch/factor_mining_20260803_cross_asset_event_clock_v1_smoke_20250102_20250109_6d`.
  The latter has six date files, unique `(dt, code)`, exact 6-column schema,
  zero Inf, and a research-only manifest.  Five signals are nonconstant and
  broadly finite.  `xca_lull_overlap_excess` is 99.867% zero over the six-day
  smoke because exact simultaneous zero-trade bins are nearly absent; retain
  the v1 artifact for audit but do not treat that signal as a viable full-run
  candidate.  A separate activity-calendar family is being designed rather
  than mutating the evidence-backed v1 module.
- `intraday_joint_state_v1_full` remains the only active compute-heavy full
  build (Python PID `4964`, scratch root
  `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_joint_state_v1_full`).
  It has progressed beyond 126 immutable day files under the same fixed
  contract; it is still ineligible for any merge until 381 files and a
  complete manifest exist.  A transient free-memory observation of 2.88 GiB
  caused all new full builds to remain paused; no existing job was stopped.
- **2026-08-03 current checkpoint (this section supersedes stale "active" facts below):** a fresh v4 catalogue composed every complete immutable root available at the time: 475 signals in 64 families.  Its 381-day fixed-pool screen completed under the requested `2025-01-01` start, T1430/14:30 factor, same-day 14:42 label, T-1 `o_0005`, `abs(mean daily Pearson IC)>0.02`, `min_valid_days=250`, 60/20/20 partitions with at least 50 valid days each, and 200-common-day redundancy gates.  It selected **25** signals (58 quality-eligible), not 100; 399 failed IC, 10 failed 250-day coverage, and the rest were redundancy conflicts.  See `docs/experiment_records/factor_mining_unified_v4_screen_20260803.md` and `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v4_all_complete_screen_v1`.
- The v4 merge preserved genuine cross-root historical row differences through a per-day outer union with missing-source values as `NaN`; it did not change the fixed T-1 pool or fill missing values.  The merged store and manifest are at `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v4_all_complete_merged_v1`.
- `intraday_joint_state_v1_full` is still running and incomplete; it must not enter a catalogue/merge/screen until it has all 381 files and a complete immutable research manifest.  All smoke and partial roots remain ineligible.
- The real Git root is `C:\Users\BaiYang\CBOND_ON\cbond_on`; the worktree is intentionally dirty and must not be reset, cleaned, checked out, or broadly modified.
- The unified requested IC start date is `2025-01-01`; it is a non-trading day, so the frozen evaluation calendar begins on `2025-01-02`.
- Current usable fixed-universe range is `2025-01-02..2026-07-30`, with 381 factor/label calendar days.  Every score day must use the prior trading day's `quant_factor_dev.researcher_xuvb.o_0005` file, with no pool fallback.
- Strict-PIT v3 has completed and its fixed-gate screen accepted only 13 of 193 vetted signals.  The number is a baseline fact, not evidence that the requested 100 target has been achieved.
- Two independent, research-only expansion builds are active: 84 intraday/path signals in `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_combined_full_v1` and 64 T-1 daily signals in `D:/cbond_on/research_scratch/factor_mining_20260803_daily_full_v1`.  Both use the same `2025-01-01..2026-07-30`, T1430/14:30, label 14:42 contract, DataHub-only sources, Python engine, and no backtest/report stage.
- Daily contract/stock has 46 signals in seven families. Its daily-data contract is strict-prior, with an independent price T-1 anchor and a per-security anti-stale base-row guard. Its full scratch build is active under `factor_mining_20260803_daily_contract_stock_v1_full`.
- Its real single-day scratch output has a unique (dt, code) FactorStore with 330 rows, 46 catalog-matching columns, 14,272 finite cells, 908 NaN, and zero Inf; it did not touch live, DB, scheduler, or production FactorStore.
- The intraday IOPV family is expected to fail closed on current DataHub data: IOPV is physically present but zero-only in the audited running prefix, so its six signals are all NaN. Two trade-size candidates are non-null but have materially inadequate coverage. This is a data-availability limitation, never a reason to fill values or weaken the screen.
- Uncombined microstructure v2 has passed a real scratch smoke: 24 columns over 329 unique rows, 7,575 finite cells, 321 NaN, and zero Inf. Quote-path and depth-centroid are fully covered; event-clock is mostly covered and has one lower-coverage candidate, so all remain research candidates pending the unified screen.
- Daily incremental v1 has independently passed focused tests and no-write preflight; its code-matching 2026-04-28 smoke has 330 rows, 28 columns, 8,797 finite cells, and zero Inf. Its correct full scratch build is active under `factor_mining_20260803_daily_incremental_v1_full`.
- Intraday microstructure v2 is an independent 24-signal / three-family research catalogue: quote/trade-path asynchrony, event-clock price discovery, and depth-centroid relocation. Its focused module and generic runner tests passed (11 tests), its dedicated one-day scratch manifest is research-only, and its 2025-01-01..2026-07-30 no-write preflight passed against a previously absent root.
- Daily state events v1 r2 is a queued independent 24-signal / four-family catalogue (`capital_supply_transition`, `call_put_activation_state`, `conversion_adjustment_state`, and `bond_stock_quantity_regime`). It has passed its focused tests plus real smoke according to its experiment record; the intended full root `D:/cbond_on/research_scratch/factor_mining_20260803_daily_state_events_v1_r2_full` is absent. It must not start while it would contend with the four active full builds.
- Daily state events v1 r2 is an independent 24-signal / four-family catalogue (`capital_supply_transition`, `call_put_activation_state`, `conversion_adjustment_state`, and `bond_stock_quantity_regime`). It passed its focused tests plus real smoke and, after residual r1 completed its audit under a safe-memory observation, was launched in its fresh intended full root `D:/cbond_on/research_scratch/factor_mining_20260803_daily_state_events_v1_r2_full`. It then completed naturally (launcher PID `19192` exited) and independently passed the complete root audit: 381 days / 156,178 rows, one research-only 2025-01-01..2026-07-30 manifest, 24 signals / four families, exact stable catalog schema, unique `(dt, code)` index with file-day alignment, and zero Inf cells. It is excluded from the already waiting PID `29248` first-screen composition but is eligible for the later fresh all-complete-roots composer.
- Intraday joint state v1 is an independent 42-signal / seven-family catalogue. Its audited smoke has seven focused tests, a 330x42 real scratch frame with zero Inf, strict 14:29 intraday visibility, and T-1 daily mapping. It was subsequently launched in its fresh intended full root `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_joint_state_v1_full`: launcher PID `23200`, Python child `4964`, and parent-scratch `.launch.*` logs. It is outside the immutable PID `29248` first screen and must be included in a later fresh all-complete-roots composer.
- Intraday state interaction v1 is a queued independent 36-signal / six-family catalogue. It has 10 focused tests plus ruff, an r5 330x36 smoke with zero Inf and no constants, and strict PIT validation. Its intended full root `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_state_interaction_v1_full` must remain absent until its conditional launch.
- Intraday cross-sectional residual v1 r1 has 36 candidates in six families.  Its first smoke was all-empty because any local negative ``amount`` or ``volume`` cumulative-counter delta invalidated every summary for that bond, leaving only 27 rows below the Ridge minimum of 30.  The r1 contract keeps the pre-existing bounded signed ``amount`` correction semantics (absolute <=100 and relative <=1e-5), keeps ``volume``/``num_trades`` strict, and isolates each source field to only the signals that require it.  Focused tests and the fresh 2026-04-28 scratch smoke passed: 330 rows, all 36 columns nonempty/nonconstant, 10,892 finite cells, and zero Inf.  A full-window preflight is clean; it remains queued until the two active full builds finish.
- Existing production FactorStore has 202 raw (non-tail-transform) columns on the same contract; only 23 currently show absolute mean daily Pearson IC above 0.02.  This confirms that relabeling existing factors is not an acceptable route to the requested 100 candidates.
- Default factor batch writes to production `D:/cbond_on/factor_data`; this task requires an explicit environment-pinned scratch paths profile and a pre-run path assertion.

## 2026-08-03 Continuation Checkpoint (current)

- The canonical complete-root screen remains
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v4_all_complete_screen_v1`:
  475 signals / 64 families / 381 score days, 58 quality-eligible, and 25
  canonical greedy selections.  A read-only maximum-independent-set audit
  found an upper bound of 27 on the same candidates; the requested 100 has
  not been reached and thresholds were not relaxed.
- The fixed contract remains `2025-01-01` requested start (first score day
  `2025-01-02`), T1430/14:30 factor, same-day 14:42 label, strictly T-1
  `o_0005` universe, `abs(mean daily Pearson IC) > 0.02`, at least 250 valid
  days and 50 per chronological partition, 200 redundancy days, within-family
  correlation `<0.80`, and cross-family correlation `<0.70`.
- Joint-state remains a mutable scratch-only build (child PID `4964`) and
  must not be merged or screened until it naturally exits with all 381 date
  files and exactly one run manifest plus family catalogue.  It is the only
  still-active full-window factor build at this checkpoint.
- Daily-TWAP microstructure completed naturally and passed a full immutable
  root audit: research-only manifest, 12 signals / four families, 381 files
  (`2025-01-02..2026-07-30`), 156,178 rows, stable schema, unique `(dt, code)`,
  zero Inf, and nonzero finite coverage for every signal.  It has been
  combined only through a fresh v5 catalogue/outer-union merge, never appended
  to the prior 25-factor selection.
- The v5 fixed-pool screen completed naturally at
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v5_with_daily_twap_screen_v1`
  over 487 signals / 68 families.  It accepted 26 (59 quality-eligible): the
  only new accepted signal was
  `dtwm_session_afternoon_late_log_slope` with absolute mean daily Pearson IC
  `0.0208896185`.  This is research evidence only and does not meet the 100
  target.
- The newly added family modules remain research-only and are not imported by
  `defs.__init__` or referenced by a live/model config.  The earlier eleven
  modules passed 78 focused tests; the current twelve-module expansion (109
  candidates / 39 families) has focused-test evidence per module and an
  aggregate-catalogue test.  Ruff was clean.  The aggregate catalogue
  `research_factor_mining_aggregate_catalog_v1` passed a no-write expansion
  preflight under the same paths and temporal contract; its smoke root remains
  absent until enough memory is released.
- Orderbook repricing has an audited 2026-04-28 smoke root at
  `D:/cbond_on/research_scratch/factor_mining_20260803_orderbook_repricing_v1_smoke_20260428`:
  329 unique `(dt, code)` rows, six nonconstant factor columns, and zero Inf.
  This confirms mechanics only, not IC or correlation eligibility.
- Orderbook repricing full build is active in its fresh root
  `D:/cbond_on/research_scratch/factor_mining_20260803_orderbook_repricing_v1_full`;
  it must follow the same natural-completion and full-root audit before any
  composition or screen.
- Current next action: await the v5 screen and the still-running joint-state
  build.  Audit every future immutable root before it enters a fresh
  all-complete-roots catalogue, outer-union merge, and fixed-pool screen.  Do
  not append any later result to the 25-factor list.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/skills/cbond-factor-backtest-protocol/SKILL.md`
- `harness/workflows/research_experiment.md`
- `harness/workflows/factor_backtest_protocol.md`
- `harness/context/source_of_truth.md`
- `docs/开发规则.md`
- `docs/ai_factor_factory_dify_prompt.md`
- Current factor pipeline, storage, config resolver, and strict `o_0005` audit helpers.

## Files Changed

- `harness/task_state_factor_mining_20260802.md`
- `cbond_on/domain/factors/defs/research_factor_mining_intraday_cross_section_residual_v1.py`
- `tests/test_research_factor_mining_intraday_cross_section_residual_v1.py`

## Commands Run

- Research-experiment and factor-backtest read-only preflights.
- Read-only fixed-universe baseline scan over existing FactorStore columns.
- `py -3.11 -m pytest -q tests/test_research_factor_mining_catalog_v1.py tests/test_factor_mining_screen.py` (14 passed).
- `PYTHONDONTWRITEBYTECODE=1 py -3.11 -m pytest -q tests/test_research_factor_mining_daily_contract_stock_v1.py` (8 passed).
- `py -3.11 -m pytest -q tests/test_research_factor_mining_intraday_microstructure_v2.py` (11 passed).
- `py -3.11 -m pytest -q tests/test_research_factor_mining_daily_incremental_v1.py tests/test_run_factor_mining_expansion.py` (12 passed).
- `py -3.11 -m pytest -q tests/test_research_factor_mining_intraday_cross_section_residual_v1.py` (6 passed; real smoke coverage failure remains open).
- The daily contract/stock real single-day scratch smoke completed successfully on 2026-04-28 through the safe expansion runner.
- Strict-PIT v2 real smoke build for `2026-04-28..2026-04-29` (238 columns/day, scratch-only).
- A v2 full-build attempt was stopped after confirming a repeated output-index
  scan bottleneck.  It remained scratch-only and its partial output is retained
  for audit; v3 will restart from a fresh root after the cache regression test.
- v3 cache regression and focused tests: `15 passed`; v3 real smoke ran in
  about 19 seconds per day.  Its `2026-04-28` and `2026-04-29` FactorStores
  match v2 cell-for-cell exactly (238 fields, no Inf).
- Full strict-PIT v3 build started in a hidden process on 2026-08-02 23:35 CST:
  PID `17136`, stdout/stderr under `D:/cbond_on/research_scratch/factor_mining_20260802_strict_pit_v3/logs/`.
- The post-v3 screen ran at `D:/cbond_on/research_scratch/factor_mining_20260802_screen_v3_full_vetted_20260803_0227/`, writing `accepted_factors.csv`, `rejected_factors.csv`, daily metrics, redundancy evidence, and an input manifest.
- The first-wave 2026-08-03 intraday/path and daily builds completed with immutable manifests and 381 T1430 days. Residual r1 completed naturally (launcher PID `14104` exited) and independently passed the full root audit: 381 days / 156,178 rows, one research-only 2025-01-01..2026-07-30 manifest, 36 signals / six families, exact stable catalog schema, unique `(dt, code)` index with file-day alignment, and zero Inf cells. Daily incremental v1 also completed naturally (launcher PID `3492` exited) and passed the same audit: 381 days / 156,178 rows, one research-only manifest, 28 signals / five families, exact stable catalog schema, unique date-aligned index, and zero Inf cells. Daily contract/stock launcher PID `34420` (Python child `11124`) remains active. The guarded post-processing orchestrator is PowerShell PID `29248`; it must wait for all three original launchers, then enforce 381 days, exactly one manifest, research-only status, expected catalogue counts, and date contract before catalog composition, merge, or screening. Do not reuse, restart, or stop any scratch root merely to refresh progress.
- After a resource review found approximately 13.9GB available memory, the independent microstructure v2 full build was explicitly authorized and launched through the same safe runner: launcher PID `25924` (Python child `9868`), scratch root `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_microstructure_v2_full_r1`, and hidden parent-scratch launch logs with the matching `.launch.stdout.log` / `.launch.stderr.log` names. It was intentionally outside the immutable in-flight v3 screen.
- Early microstructure v2 runtime evidence was healthy but compute-heavy: its first two of 381 days completed at about 124 seconds per day, with no stderr error and more than 10GB free memory at the observation. A later concurrent-runtime observation fell from 4.53GB to 1.64GB free while residual r1 was at 188 days, contract/stock 62, incremental 117, and microstructure 4.
- Under an explicit immediate red-line instruction, the microstructure Python child PID `9868` and launcher PID `25924` were command-line verified against the exact microstructure module/root, then terminated. The partial root and both launch logs were preserved; no file was removed or reused. A subsequent instruction that would have kept it alive arrived after termination, so it cannot be treated as an active job or restarted in place. Memory measured 9.53GB free immediately after the stop. Any later microstructure full rerun requires a newly named root and explicit direction.
- Intraday residual r1 validation: `py -3.11 -m pytest -q tests/test_research_factor_mining_intraday_cross_section_residual_v1.py tests/test_t1430_amount_accel_depth_delta_v2.py` (`17 passed`); its dedicated one-day clean-direct build wrote only `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_cross_section_residual_v1_r1_smoke_20260428`.

## Artifacts

- v1 single-day scratch output is structurally complete (238 factors across 27
  families) and remains preserved for audit only.  It is not eligible for IC
  because clean-direct rolling paths can relabel prior `trade_time` snapshots
  with the score-day index.
- v2 strict-PIT smoke is preserved for audit.  v3 retains the same PIT logic
  but caches the physical output index once per build day, eliminating repeated
  scans for all 238 concrete specs.
- Daily contract/stock smoke FactorStore is under its dedicated factor_mining_20260803_daily_contract_stock_v1_smoke_20260428 scratch root.
- Microstructure v2 successful smoke FactorStore is under a dedicated factor_mining_20260803_intraday_microstructure_v2_smoke_20260428 GUID scratch root.
- Daily incremental code-matching smoke is preserved under its prior dedicated scratch root. The active correct full root is `D:/cbond_on/research_scratch/factor_mining_20260803_daily_incremental_v1_full`; it is incomplete until it has 381 date files and one immutable manifest. The accidentally started `factor_mining_20260803_daily_incremental_full_v1` partial root is audit-only (three date files, no manifest) and must never be an input.
- Active strict-PIT scratch root: `D:/cbond_on/research_scratch/factor_mining_20260802_strict_pit_v3/`.

## Open Risks

- The user-requested count is a target, not a result: no candidate may be claimed accepted until it passes the fixed-pool IC, validity, chronology, and correlation gates.
- The v1 one-day sample found 36 all-empty flow-derived candidates.  Diagnose
  the raw counter semantics or let the validity gate reject them; do not fill
  missing flow values with zero.
- Cross-asset mappings are accepted only when `daily_base` has a mapping on
  the prior market session independently anchored by `daily_price`; an older
  base row must produce `NaN`, never a stale stock match.
- Daily context contains the score-day source file by design; all daily factor implementations must explicitly filter to dates strictly before the score date.
- Cross-family redundancy will be measured pairwise on common fixed-universe observations, not inferred from economic names.
- Do not stop, restart, or reuse either active first-wave expansion root merely to omit unusable IOPV signals; the completed immutable root is evidence. Candidate pruning belongs only to a later fresh composition/screen stage.
- The current second-wave roots are partial while their processes are alive; no partial root, partial parquet set, or absent-manifest root may enter the merge. If any job fails its postcondition, the orchestrator must fail closed before screen rather than fill, resume in place, relax thresholds, or substitute the audit-only incremental root.
- Do not launch another compute-heavy batch while the three current full builds are active. The microstructure partial root is audit-only. If available memory falls below 4GB or any launcher/process becomes abnormal, record the evidence and halt only subsequent launches; do not interrupt the three existing jobs without explicit owner direction.
- After joint-state launch, it is the newest and sole sacrificial task. If memory is below 2GB for two consecutive samples or there is OOM evidence, verify its exact command line and terminate only joint state; preserve its partial root and launch logs. Do not interrupt daily contract/stock, daily state events, the completed roots, or PID `29248`.
- The fixed screen contract is non-negotiable: IC start `2025-01-01` (calendar starts `2025-01-02`), fixed T-1 `o_0005`, `abs(mean daily Pearson IC) > 0.02`, within-family absolute correlation `<0.80`, cross-family absolute correlation `<0.70`, `min_valid_days=250`, each chronological partition at least 50 days, redundancy evidence at least 200 common days, and `max_selected=100`.
- The first unified v3 screen is diagnostic only, never the final list. Whenever any later module completes, compose a fresh catalog from *all complete immutable roots*, write a new merged FactorStore, and run a new full-pool screen with global IC ordering and global pairwise redundancy. Never append a later factor to the first-screen accepted list incrementally.

## Next Action

- Monitor daily contract/stock, daily state events, and intraday joint state full builds without restarting them. The original orchestrator (PID `29248`) may compose the initial unified family catalog, merge the six immutable source stores, and run the fixed `o_0005` IC/correlation screen only after every listed original-screen input root passes its postcondition. Daily state events and intraday joint state are deliberately excluded from that in-flight merge. The microstructure partial root is also excluded and may only be recomputed later in a new root if needed.
- Audit the resulting `screen_summary.json`, `accepted_factors.csv`, `rejected_factors.csv`, and `factor_pair_redundancy.csv`. If fewer than 100 candidates survive, report accepted/eligible/redundancy distributions and add only low-related new information families, each through a fresh smoke root then a fresh full root. Do not lower IC, correlation, chronology, validity, or mask gates.
- Treat that first screen strictly as a diagnostic expansion decision. If any later root is built, produce a new catalog, new merged root, and new screen root encompassing every complete root, then use only that latest global screen as the candidate list source.
- Daily state events v1 r2 is the next conditional queue entry. Only after at least one current full build naturally succeeds, available memory remains safely above 4GB, and the unified screen has not accepted 100 factors, run its no-write preflight and then launch the exact absent full root. It remains outside the in-flight v3 unified screen.
- Intraday joint state v1 is also a conditional queue entry under the same resource and accepted-count gates. First run a no-write preflight against `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_joint_state_v1_full`; start only if it remains a fresh root and does not contend with active full builds.
- Intraday state interaction v1 is also a conditional queue entry: release capacity first, require the global screen to be below 100, run no-write preflight against `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_state_interaction_v1_full`, and then use a new full root. It must be included in the next fresh all-complete-roots global merge/screen, never incrementally appended.
- Microstructure v2 may be reconsidered only after one of residual r1, daily contract/stock, or daily incremental naturally succeeds and passes its complete audit, available memory is at least 8GB, and the unified screen has accepted fewer than 100 factors. Its only permitted rerun target is the new absent root `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_microstructure_v2_full_r2`, after a new no-write preflight; r1 must never be restarted or reused.

## Handoff Summary

- No live configuration, model, database, scheduler, strategy, mask, or production factor output has been changed.

## 2026-08-03 latest rank-family extension checkpoint

- Aggregate v5 remains the sole active full-window build under
  `D:/cbond_on/research_scratch/factor_mining_20260803_aggregate_catalog_v5_full`
  (Python PID `1572` at the latest observation).  It must complete naturally;
  no second full build has been started or queued to overlap it.
- `research_factor_mining_daily_relative_rank_coupling_v1.py` is a new
  research-only one-signal module.  Its all-market daily rank construction is
  strict-prior and never uses the future union of `o_0005` codes; `o_0005` is
  used only by the later fixed screen.  The candidate
  `drrc_return_amount_rank_spearman60` had a read-only corrected pre-screen
  mean daily Pearson IC `+0.023629` over 381 score days and passed a v7
  eligible-pool redundancy audit (maximum `0.475931`, 381 common days).
- The module passed 27 focused tests plus Ruff, an absent-root full-window
  no-write preflight, and an isolated 2026-04-28 scratch smoke with 330 unique
  rows, 323 finite values, seven NaNs, zero Inf, and exact 14:30 timestamps.
  It is not imported by `defs.__init__`, and is absent from all live/model
  config, contracts, DB, scheduler, and production FactorStore paths.
- A small auditable scratch pre-screen panel at
  `D:/cbond_on/research_scratch/factor_mining_20260803_daily_relative_rank_tail_prescreen_v1`
  contains only candidate daily values and a research-only manifest.  It is
  not a FactorStore and cannot enter merge/screen inputs.  Its amount-tail
  candidate is under strict v7 pairwise audit; the trade-size-tail sibling is
  already rejected from the same family because their redundancy is `0.874204`
  (>= 0.80).  No family threshold has been relaxed.
- `research_factor_mining_daily_relative_rank_tail_contradiction_v1.py` has
  now been added as a separate nonlinear rank-tail family after its current
  v7/new-candidate redundancy checks.  It passed 25 focused tests, Ruff,
  no-write full preflight, and a 2026-04-28 scratch smoke: 330 unique rows,
  323 finite values, seven NaNs, zero Inf, and exact 14:30 timestamps.  The
  intended full root remains absent while v5 is active.

## 2026-08-03 OHLC / rank-flow continuation checkpoint

- Aggregate v5 remains the sole active full-window build.  At the latest
  independent read-only audit its Python PID was `1572`, it had 56 of 381
  date files, and it had not produced a run manifest, family catalogue, or
  `.done`; it must not be composed, merged, screened, stopped, or restarted.
- The future-build successor
  `research_factor_mining_daily_relative_rank_flow_coupling_v2.py` now has
  two same-family rank-flow signals.  Its no-write full preflight, focused
  `10 passed` test set, Ruff, and isolated 2026-04-28 scratch smoke passed:
  330 unique rows, 323 finite values and seven NaNs per signal, zero Inf, and
  exact 14:30 timestamps.  Its intended full root remains absent.
- A no-write nine-candidate daily rank-lag transmission batch did not find a
  viable extension: its best absolute mean daily Pearson IC was `0.009831`.
  No code or FactorStore was created from that rejected batch.
- A separate pre-registered 19-candidate strict-prior daily-OHLC path batch
  retained two signals in the genuinely new research-only family
  `prior_daily_ohlc_wick_path_asymmetry`.  Their preliminary full-window ICs
  were `+0.025656` and `+0.022637`; each had 381 valid days, v7-only maximum
  redundancy `0.335210` / `0.509127`, and mutual redundancy `0.298079`.
  These are pre-screen values, not final global admission.
- The corresponding module and focused test are
  `research_factor_mining_daily_ohlc_wick_path_asymmetry_v1.py` and
  `test_research_factor_mining_daily_ohlc_wick_path_asymmetry_v1.py`.  They
  passed `26` focused tests, Ruff, full no-write preflight, and an isolated
  2026-04-28 scratch smoke.  The smoke had 330 unique rows, exact 14:30
  timestamps, two nonconstant columns, 323 finite values/seven NaNs each,
  zero Inf, and an independent raw formula reproduction with zero difference.
  It is absent from `defs.__init__`, all live/model config, contracts, DB,
  scheduler, and production FactorStore.
- Full builds remain serial: no new complete root is launched until v5 exits
  naturally and passes immutable root audit.  Every later root must be folded
  into a fresh all-complete-roots catalogue, outer-union merge, fixed T-1
  screen, and exact MIS; no incremental accepted-list append is permitted.

## 2026-08-03 daily orthogonal batch readiness

- Individually pre-screened and mechanically smoked daily candidates are now
  assembled in the research-only
  `research_factor_mining_daily_orthogonal_batch_v1` catalogue: 10 signals,
  six genuine families, and five source kernels. It deliberately uses
  rank-flow v2 rather than its obsolete single-signal v1 predecessor, so no
  same-signal duplicate is possible. The batch has no signal/family collision
  with active v5's 214-signal/75-family catalogue.
- Focused aggregate/member validation passed `38` tests with Ruff clean. Its
  full no-write preflight proves the new full root is absent and all derived
  outputs stay below `D:/cbond_on/research_scratch`; inputs are pinned to local
  DataHub and reports/backtests stay disabled.
- Combined 2026-04-28 scratch smoke completed in its own root: 330 unique
  T1430 `(dt, code)` rows, exact 14:30 time, 10 columns, no constant column,
  zero Inf, and 323--324 finite values per signal. The smoke manifest is
  research-only. This confirms co-execution/cache behavior only, not final
  IC/redundancy admission.
- The fresh full root is intentionally still absent. It remains queued behind
  v5; launching it before v5 naturally exits would violate the serial full
  build/resource gate.

## 2026-08-03 daily orthogonal v4 continuation

- Aggregate v5 remains the sole active full-window build.  At the latest
  read-only observation it had 77 of 381 parquet days (latest score day
  `2025-04-29`), Python PID `1572` remained alive, and no
  `factor_mining_run_manifest.json` existed.  It must complete naturally;
  do not compose, screen, stop, restart, or reuse this partial root.
- `research_factor_mining_daily_bond_stock_copula_tail_dependence_v1` adds a
  strict-prior empirical-copula family.  Of its three pre-registered
  mechanisms, only `bsct_upper_tail_dependence60` passed the 381-day fixed
  T-1 pre-screen (`+0.022812`; discovery/validation/holdout
  `+0.023074/+0.023443/+0.021414`) and v7 provisional redundancy (`0.503544`,
  381 common days).  The qualified one-signal healthy wrapper is used by
  later batches; the two IC-failing siblings are not queued.
- `research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1`
  adds three strict-prior cross-sectional bond/underlying-stock rank signals.
  Their 381-day pre-screen ICs are `+0.027487`, `+0.024546`, and `+0.021430`;
  all partition signs remain positive.  Their v7 maximum redundancies are
  `0.610472`, `0.579420`, and `0.631020`, while their maximum same-family
  redundancy is `0.773341`.  These are provisional only until v5 and every
  later root are globally screened.
- A seven-candidate capacity-normalized return-rank tail batch was explicitly
  pre-registered and evaluated zero-write.  Six expressions cleared the IC
  gate, but every one conflicted with the v7 pool across-family (`0.707198`
  to `0.836788`, largely `base_float_adjusted_amount`); the entire family is
  rejected rather than weakening the 0.70 rule or adding code.
- The successor `research_factor_mining_daily_orthogonal_batch_v4` preserves
  v3 and combines 15 signals / nine families / eight kernels.  Focused
  aggregate/member verification passed `25` tests with Ruff and diff checks
  clean.  Its full no-write preflight proves
  `D:/cbond_on/research_scratch/factor_mining_20260803_daily_orthogonal_batch_v4_full`
  is absent and every derived path remains under research scratch.
- The combined v4 `2026-04-28` scratch smoke has 330 unique T1430 `(dt,code)`
  rows at exact 14:30, 15 nonconstant columns, 290--324 finite values per
  signal, 6--40 explicit NaNs, and zero Inf.  The lower 290 coverage of the
  cross-sectional-rank family is intentional fail-closed behavior for missing
  mapped-stock/rank anchors.  See the two dedicated experiment records for
  raw-formula and pre-screen evidence.
- Next action: after v5 immutable-root audit, launch only the fresh v4 full
  root (not older v1/v2/v3 roots), audit it, then compose v7+v5+v4 and run the
  fixed 2025-01-01/T-1 global screen plus exact MIS.  Do not append any
  pre-screened signal to an accepted list.

## 2026-08-03 observable-seasoning continuation checkpoint

- The full-window market-supply/absorption candidate
  `msa_market_flow_share50` had a strong strict-T-1 pre-screen IC
  (`+0.038408043`, 381 days), but the v7 full-catalogue correlation audit
  found ten cross-family conflicts at `>= 0.70`, maximum `0.816436678`.
  It is rejected and no module was created; neither a family rename nor a
  threshold relaxation is permitted.
- The independently screened `prior_observable_market_seasoning` family
  retains two mutually admissible members:
  `osa_terminal_amount_streak60` (IC `-0.026728369`) and
  `osa_observation_density60` (IC `-0.020690000`).  Each has 381 valid score
  days; their mutual redundancy is `0.782062123` over 381 days, v7 maximum
  cross-family redundancy is `0.452114966` / `0.629500881`, and neither
  produces a v7 cross-family `>= 0.70` conflict.  These are provisional
  evidence only until the future all-complete-roots screen.
- `research_factor_mining_daily_observable_seasoning_v1.py` and the future
  `research_factor_mining_daily_orthogonal_batch_v7.py` have been added as
  research-only modules, deliberately outside `defs.__init__`, all live/model
  config, contracts, DB, scheduler, and production FactorStore.  Focused
  tests (`16 passed`), Ruff, and scoped diff checks are clean.
- Its isolated 2026-04-28 smoke root completed: 330 unique T1430 rows, two
  finite nonconstant columns, zero Inf, and independent strict-prior raw
  recomputation with zero maximum difference.  The future full root remains
  absent and must wait behind the active aggregate-v5 build.

## 2026-08-03 daily breadth-regime relation checkpoint

- `prior_market_breadth_regime_relation` adds one strict-prior family member,
  `brr_rank_low_high_spread60`: the difference in a security's historical
  cross-sectional return rank under locally low versus high full-market
  breadth.  It consumes only completed `market_cbond.daily_price` history;
  the full source cross-section, not `o_0005`, defines breadth and ranks.
- The zero-write 381-day pre-screen was marginal but admissible under the
  requested absolute IC gate: daily Pearson IC `-0.020012648`, partitions
  `-0.019423 / -0.031035 / -0.010878`, v7 maximum redundancy `0.540629053`,
  and no v7 cross-family conflict at `>= 0.70`.  Because the clearance is
  only `0.000012648` and the holdout magnitude is below `0.02`, it must not be
  described as finally selected.
- Its research-only module, test, and future `daily_orthogonal_batch_v8`
  catalogue now pass the focused `23`-test set and Ruff.  A pandas
  `DatetimeIndex.tail()` implementation defect was found by that test run and
  repaired with an equivalent explicit final-60-session slice before smoke.
- The isolated 2026-04-28 scratch smoke root is
  `D:/cbond_on/research_scratch/factor_mining_20260803_daily_breadth_regime_relation_v1_smoke_20260428`.
  It has 330 unique T1430 rows at exact 14:30, 323 finite values, seven
  fail-closed NaNs, zero Inf, and an independently reconstructed raw-data
  formula match with identical NaN mask and maximum absolute difference `0.0`.
- `daily_orthogonal_batch_v8` remains a future-only catalogue.  Do not launch
  any full v8 build until aggregate-v5 completes naturally and has passed the
  immutable-root audit; then use all-complete-roots outer-union, fixed global
  screen, and exact MIS rather than incrementally appending this candidate.

## 2026-08-03 v8 serial queue toward 50 exact-MIS candidates

- Owner authorized continued research-only factor generation toward a strict
  50-factor exact-MIS target.  The contract remains immutable: score range
  `2025-01-01..2026-07-30`, T1430/14:30 factor values, same-day 14:42 labels,
  fixed T-1 `o_0005`, absolute mean daily Pearson IC strictly above `0.02`,
  250 total valid days, 50 valid days per 60/20/20 partition, 200 common days,
  and correlation `<0.80` within / `<0.70` across families.
- The existing `daily_orthogonal_batch_v8` is the next serial expansion: 24
  signals / 13 families.  It passed 23 focused tests and a fresh combined
  2026-04-28 smoke at
  `D:/cbond_on/research_scratch/factor_mining_20260803_daily_orthogonal_batch_v8_smoke_20260428_r1`:
  330 unique T1430 rows, 24 columns, 7,670 finite values, 290--330 non-null
  values per signal, zero Inf, and one research-only manifest.
- Its full root
  `D:/cbond_on/research_scratch/factor_mining_20260803_daily_orthogonal_batch_v8_full_r1`
  passed no-write preflight and remains absent until aggregate-v5 completes.
  It must never overlap the active aggregate-v5 full build.
- A hidden, fail-closed serial queue is running as PowerShell PID `31320`:
  `harness/tools/queue_factor_mining_v8_to50.ps1`.  Its audit state/logs are
  under `D:/cbond_on/research_scratch/factor_mining_20260803_to50_serial_queue_r1`.
  It waits for aggregate-v5's manifest + 381-day audit, freezes 96 source,
  config, and harness hashes, requires two consecutive >=6GB free-memory
  samples, then builds v8, composes v7+v5+v8, outer-union merges their stores,
  runs the fixed global screen, and runs exact MIS.
- The no-write catalogue composition is valid: v7 + aggregate-v5 + v8 equals
  773 signals / 165 families, with no collision.  It will write only new
  scratch roots and `status.json` will report whether the exact-MIS count has
  reached 50.  Any predecessor failure, incomplete root, source drift, or
  memory failure stops it; no root is restarted, overwritten, or promoted.
