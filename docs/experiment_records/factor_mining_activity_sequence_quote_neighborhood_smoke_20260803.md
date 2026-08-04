# Activity, event-sequence, quote-execution, and neighborhood strict-T1430 smoke (research-only)

## Scope and fixed boundary

This is a two-score-day mechanics and quality audit of four independent
research catalogues.  Every run used a fresh root below
`D:/cbond_on/research_scratch`, DataHub-only `clean_direct` inputs, T1430 /
`14:30`, same-score-day `14:42` label configuration, the Python engine, and
the physical continuous-session cutoff `14:29:00` implemented by each
module.  The runs covered `2026-04-28..2026-04-29` only.

No full-window build, IC screen, backtest, aggregate import, `defs.__init__`
change, factor contract/configuration change, live output, model, DB,
scheduler, mask, label, or trading-rule change occurred.  The scratch
manifests are all explicitly `research_only: true`; backtest/screening/report
publication stages were disabled by the expansion launcher.

## Static checks and v7 name/family de-duplication

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
py -3.11 -m pytest -q `
  tests/test_research_factor_mining_cross_asset_activity_calendar_v1.py `
  tests/test_research_factor_mining_cross_asset_event_sequence_topology_v1.py `
  tests/test_research_factor_mining_quote_execution_dynamics_v1.py `
  tests/test_research_factor_mining_cross_sectional_microstructure_neighborhood_v1.py `
  -p no:cacheprovider
```

Result: `30 passed`; Ruff was clean for all four module/test pairs.

A no-write composition against the immutable v7 catalogue completed without
signal or family ambiguity.  v7 has 535 signals in 77 families; these four
catalogues would bring a future composition to 565 signals in 87 families.
The command did not use `--execute`, and its planned scratch child does not
exist.  This is static identity evidence only, not an IC or statistical
correlation result.

```powershell
py -3.11 harness/tools/compose_research_factor_catalog.py `
  --vetted-v3-catalog D:/cbond_on/research_scratch/factor_mining_20260803_unified_v7_with_joint_catalog_v1/family_catalog.json `
  --module cbond_on.domain.factors.defs.research_factor_mining_cross_asset_activity_calendar_v1 `
  --module cbond_on.domain.factors.defs.research_factor_mining_cross_asset_event_sequence_topology_v1 `
  --module cbond_on.domain.factors.defs.research_factor_mining_quote_execution_dynamics_v1 `
  --module cbond_on.domain.factors.defs.research_factor_mining_cross_sectional_microstructure_neighborhood_v1 `
  --output-name factor_mining_20260803_v7_static_dedupe_activity_sequence_quote_neighborhood_plan
```

## Immutable two-day smoke audit

Each output has exactly two FactorStore files, one per requested date, exact
catalogue column order, a unique `(dt, code)` `MultiIndex`, date-aligned
indices, and zero `Inf` / `-Inf` values.  No nonempty signal is a
cross-sectional constant on either day or in the two-day union.

| Catalogue | Scratch root | Families / signals | Rows | Finite coverage | Smoke status |
| --- | --- | ---: | ---: | --- | --- |
| `cross_asset_activity_calendar_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_cross_asset_activity_calendar_v1_smoke_20260428_20260429` | 2 / 6 | 330 + 324 = 654 | all six: 302 + 308 = 610 / 654 | every signal nonempty, nonconstant |
| `cross_asset_event_sequence_topology_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_cross_asset_event_sequence_topology_v1_smoke_20260428_20260429` | 2 / 6 | 330 + 324 = 654 | five: 610 / 654; `xca_joint_burst_cluster_continuity`: 301 + 304 = 605 / 654 | every signal nonempty, nonconstant |
| `quote_execution_dynamics_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_quote_execution_dynamics_v1_smoke_20260428_20260429` | 2 / 6 | 330 + 324 = 654 | all six: 306 + 303 = 609 / 654 | every signal nonempty, nonconstant |
| `cross_sectional_microstructure_neighborhood_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_cross_sectional_microstructure_neighborhood_v1_smoke_20260428_20260429` | 4 / 12 | 330 + 324 = 654 | see exact partial-empty set below | partial failure; no source change made |

## Cross-sectional neighbourhood: exact partial-empty result

The following three signals are all-NaN on both dates (`0 / 654` finite):

- `csn_lql_terminal_lock_neighbor_gap`
- `csn_lql_occupancy_neighbor_gap`
- `csn_lql_transition_neighbor_gap`

They are the three entries in
`csn_limit_queue_local_dislocation`.  They are not reported as constants;
they have no finite values at all.  The existing original module and any
aggregate catalogue remain unchanged.

The other nine entries are nonempty and nonconstant:

- `csn_pql_retention_neighbor_gap`, `csn_pql_refill_neighbor_gap`, and
  `csn_pql_churn_neighbor_gap`: 329 + 321 = `650 / 654` each;
- `csn_dlc_bid_coherence_neighbor_gap`, `csn_dlc_ask_coherence_neighbor_gap`,
  and `csn_dlc_crosslag_neighbor_gap`: 329 + 321 = `650 / 654` each;
- `csn_taq_buy_initiation_neighbor_gap`,
  `csn_taq_sell_initiation_neighbor_gap`, and
  `csn_taq_imbalance_neighbor_gap`: 306 + 303 = `609 / 654` each.

This observation does not authorize filling, threshold changes, source
fallbacks, or an in-place formula modification.  A later, separately scoped
decision may decide whether a versioned healthy-subset wrapper is warranted;
this smoke neither creates nor registers one.

## Subsequent healthy-wrapper decision

Following review of the exact three-signal failure, a separate research-only,
fail-closed catalogue view was added at:

```text
cbond_on/domain/factors/defs/research_factor_mining_cross_sectional_microstructure_neighborhood_healthy_catalog_v1.py
```

It is an explicit nine-signal whitelist retaining only the three nonempty
families above.  It reuses the original `CatalogEntry` objects and source
kernel; it has no calculation, field, parameter, cache, input, or time
contract of its own.  Its source contract locks the original 12-signal order,
family assignment, and kernel identity and fails closed on a source addition,
removal, reordering, duplicate, family reassignment, or registration drift.
It remains excluded from `defs.__init__`, aggregate catalogues, configurations,
contracts, models, DB, scheduler, and live paths.

Focused source/wrapper/composer tests passed (`16 passed`), and Ruff was clean.
A no-write two-day expansion preflight reported exactly 9 signals / 3 families
with a fresh root that remains absent.  A v7 dry-run static composition found
no signal/family ambiguity (535 signals / 77 families to 544 / 80); the
planned composition root was not created.  No wrapper smoke or full build was
started in this step.

## Decision and next permitted action

The activity-calendar, event-sequence, and quote-execution catalogues have
passed this strict two-day mechanics gate and may be queued independently for
fresh full-window research roots when capacity and parent scheduling permit.
They are not IC-qualified and do not enter an aggregate, model, or live pack
from this record.

The original cross-sectional neighbourhood catalogue is not eligible for a
full aggregate admission in its current 12-signal form because its complete
limit-queue family is empty in the two-day audit.  The original remains
unchanged.  The new healthy wrapper is only a static research candidate and
must substitute for, never append to, the original source in any later
owner-approved composition.
