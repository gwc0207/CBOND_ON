# Healthy-wrapper source-artifact audit (research-only)

## Scope and boundary

This is a read-only audit of three already-written source-kernel smoke roots
and their current research-only healthy-catalogue views.  It is not a new
factor build, an IC screen, a backtest, a correlation result, an aggregate
composition, or a live/model admission.

Audited views:

- `research_factor_mining_intraday_queue_state_healthy_catalog_v1`;
- `research_factor_mining_cross_sectional_rank_lattice_healthy_catalog_v1`;
- `research_factor_mining_cross_asset_event_clock_healthy_v1`.

No code, aggregate catalogue, `defs.__init__`, configuration, FactorStore
root, label, mask, model, live artifact, DB, scheduler, or source smoke root
was changed.  The only file written for this audit is this record.

The project `research-experiment` preflight was run read-only:

```powershell
py -3.11 harness/tools/agent_preflight.py --mode research-experiment
```

## Static provenance and isolation checks

Each view imported successfully and returned the explicit source-entry
whitelist below.  The current source implementation SHA-256, catalogue
version, signal/family/kernel entries, and source order all exactly match the
corresponding completed source-smoke manifest.  Thus the audit evaluates the
same source formulae that produced the cited artifacts; the views themselves
only select original `CatalogEntry` objects.

| View | Healthy signals / families | Excluded source entries | Collision with v7 catalogue |
| --- | ---: | --- | --- |
| queue state | 9 / 3 | 3 lock-state signals | none |
| rank lattice | 9 / 3 | 3 floor-credit signals | none |
| cross-asset event clock | 5 / 2 | `xca_lull_overlap_excess` | none |

The current v7 catalogue at
`D:/cbond_on/research_scratch/factor_mining_20260803_unified_v7_with_joint_catalog_v1/family_catalog.json`
contains 535 unique signals in 77 families.  None of the 23 retained names or
eight retained family names collide with it.  The three views are not imported
by `cbond_on/domain/factors/defs/__init__.py`, so this audit did not alter
ordinary registry/bootstrap behavior.

Important limitation: all cited data roots were built from the immutable
*source* catalogues before the wrapper-selection views were invoked.  Their
Parquet files therefore retain the excluded source columns as well.  Since the
views return the exact same retained source entries and the source hashes
match, the stored columns are valid signal-level evidence; however, a direct
wrapper-run manifest has not been produced.  Do not describe this as generic
runner-wiring validation for any wrapper.

## Completed-root audit

Every cited root has a `research_only: true` manifest, the expected
source-catalogue family mapping, a stable factor schema, MultiIndex
`(dt, code)`, zero duplicate combined index rows, and only `14:30:00` factor
timestamps.  `Inf` and `-Inf` counts are zero for every retained signal.

| View | Existing source-smoke roots | Structural result | Coverage / quality observation | Disposition |
| --- | --- | --- | --- | --- |
| queue state | six days `2025-01-02..2025-01-09` (3,017 rows), plus `2026-04-28` (329 rows) | all nine retained signals present; no all-window-empty, per-day-empty, global-constant, or per-day-constant retained column | early: `pql_*`/`dluc_*` 3,011/3,017 finite and `taqi_*` 2,773/3,017; recent: 329/329 and 306/329 respectively | may be the only candidate considered for the next aggregate smoke; full-window build and fixed screen remain mandatory |
| rank lattice | `2026-04-28` only (330 rows) | all nine retained signals present; no empty or constant retained column on that date | three barrier signals have 300/330 finite values; `csl_barrier_call_put_moneyness_skew` is 82.7% zero among its finite values | exclude from the next aggregate: only one-date evidence and a material sparsity warning |
| cross-asset event clock | six days `2025-01-02..2025-01-09` (3,020 rows), plus `2026-04-28` (330 rows) | all five retained signals nonempty/nonconstant with no `Inf` in the early six-day root | early finite counts are 2,996--3,010 / 3,020; recent every retained signal is only 119/330 finite (36.1%), with the same 211 null names | exclude from the next aggregate until the recent common-coverage loss is diagnosed and revalidated |

The remaining `NaN` values were preserved, not filled.  This is especially
important for the cross-asset family: its implementation intentionally returns
`NaN` where the strict prior-daily mapping, supplied stock panel, or minimum
common event-bin conditions are not met.  The common 119-name recent finite
set is evidence of a shared prerequisite/coverage issue, but this audit does
not infer which upstream condition caused it.

## What this evidence does not establish

- The smoke roots have backtest/screen/report stages disabled; their plot index
  files contain only headers.  They contain no IC or performance result.
- Six (or one) smoke dates cannot satisfy the fixed full-screen redundancy
  evidence requirement of 200 common valid score days.  No pairwise
  correlation threshold is claimed passed here.
- No selected factor is implied.  The future fixed screen must begin at
  `2025-01-01`, retain the strict T-1 `o_0005` universe and T1430/14:30 to
  14:42 chronology, require `abs(mean daily Pearson IC) > 0.02`, and apply
  `max(mean daily abs Pearson, mean daily abs Spearman) < 0.80` within family
  and `< 0.70` across families.

## Follow-on boundary

For the next aggregate iteration, only the queue-state healthy view has
enough completed smoke evidence to be considered.  Rank-lattice and
event-clock must not be added merely because their wrapper selection imports
cleanly.  Any future aggregate work must be isolated under a new research
scratch root and must repeat the full-root audit, IC/validity screen, and
global redundancy gate before any candidate can enter a research-selected
list.  No live promotion is implied.
