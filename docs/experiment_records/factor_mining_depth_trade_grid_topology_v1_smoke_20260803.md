# Depth-dominance and trade-grid topology strict-T1430 smoke (research-only)

## Scope and fixed contract

This record audits two independent research-only factor catalogues.  Neither
module is imported by `defs.__init__`, an aggregate catalogue, a factor
contract, a model configuration, a live configuration, or a scheduler.

- score window: T1430 / `14:30`, with only physical continuous-session
  observations through `14:29:00`;
- smoke dates: `2026-04-28` and `2026-04-29`;
- source mode: DataHub-only `clean_direct`, Python engine;
- derived outputs: fresh roots under `D:/cbond_on/research_scratch` only;
- backtest, screening, report publication, DB writes, live configuration, and
  scheduler actions: disabled / not performed.

The later full-window gate remains unchanged: requested IC start
`2025-01-01` (first score day `2025-01-02`), strict T-1 `o_0005`,
`abs(mean daily Pearson IC) > 0.02`, minimum coverage and chronological
partitions, and mean daily absolute Pearson/Spearman redundancy below `0.80`
within a family and below `0.70` across families.

## Static verification

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
py -3.11 -m pytest -q `
  tests/test_research_factor_mining_intraday_depth_dominance_topology_v1.py `
  tests/test_research_factor_mining_intraday_trade_grid_topology_v1.py `
  -p no:cacheprovider
py -3.11 -m ruff check `
  cbond_on/domain/factors/defs/research_factor_mining_intraday_depth_dominance_topology_v1.py `
  cbond_on/domain/factors/defs/research_factor_mining_intraday_trade_grid_topology_v1.py `
  tests/test_research_factor_mining_intraday_depth_dominance_topology_v1.py `
  tests/test_research_factor_mining_intraday_trade_grid_topology_v1.py
```

Result: `8 passed`; Ruff clean.

## Immutable smoke outputs

| Catalogue | Scratch root | Families / signals | Rows | Integrity |
| --- | --- | ---: | ---: | --- |
| `intraday_depth_dominance_topology_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_depth_dominance_topology_v1_smoke_20260428_20260429` | 3 / 9 | 329 + 322 = 651 | exact catalogue schema; unique `(dt, code)`; date-aligned files; zero Inf |
| `intraday_trade_grid_topology_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_intraday_trade_grid_topology_v1_smoke_20260428_20260429` | 3 / 9 | 329 + 322 = 651 | exact catalogue schema; unique `(dt, code)`; date-aligned files; zero Inf |

Both manifests are explicitly `research_only: true`, retain the two requested
dates, pin the DataHub source roots, and redirect all derived paths to their
respective scratch roots.

## Depth-dominance result: reject from future aggregate

All nine signals have **0 / 651 finite values**.  They are not merely
cross-sectionally constant: neither day contains one finite observation.

The cause is the documented all-or-nothing state contract.  A bond day is
rejected when any physical score-day snapshot has a non-finite/negative depth
or a tied five-level maximum on either side.  A read-only audit of the exact
smoke-output codes in the DataHub clean snapshots found:

| Date | Output codes | First rejecting condition | Bid tied-max rows | Ask tied-max rows |
| --- | ---: | --- | ---: | ---: |
| 2026-04-28 | 329 | `bid_tied_maximum`: 329 / 329 codes | 67,499 / 920,280 | 80,409 / 920,280 |
| 2026-04-29 | 322 | `bid_tied_maximum`: 321 codes; `ask_tied_maximum`: 1 code | 66,409 / 878,472 | 72,709 / 878,472 |

There were no non-finite or negative depth rows in that audit.  Thus a modest
row-level tie rate still makes every security fail because the current
definition requires every row in the complete sequence to have a unique
maximum.  This evidence is not a reason to fill zero, break ties arbitrarily,
or relax the v1 contract in place.  The v1 module remains an audit artefact and
must not enter a future aggregate or full-window build.

## Trade-grid result: candidate for a later isolated full build

Every trade-grid signal is nonempty and nonconstant on both smoke days.

| Signal group | 2026-04-28 finite | 2026-04-29 finite | Total finite / 651 |
| --- | ---: | ---: | ---: |
| `tgps_*` phase-scale topology (3) | 306 | 303 | 609 / 651 (93.55%) each |
| `tgta_*` transition amplitude (3) | 306 | 303 | 609 / 651 (93.55%) each |
| `tgze_zero_spell_exit_magnitude` | 292 | 291 | 583 / 651 (89.55%) |
| `tgze_zero_spell_exit_directional_imbalance` | 292 | 291 | 583 / 651 (89.55%) |
| `tgze_zero_spell_exit_reversal_share` | 276 | 279 | 555 / 651 (85.25%) |

This is only a mechanics and quality result.  The module is eligible for a
future fresh, isolated full-window scratch build after capacity approval, then
a new all-complete-roots composition, fixed-pool IC screen, chronological
stability checks, and global redundancy screen.  It has not passed any IC or
correlation gate and is not admitted to an aggregate yet.

## v7 static de-duplication

The current immutable v7 family catalogue contains 535 signals in 77 families.
A dry-run composition with both modules completed without family or signal
ambiguity, yielding a prospective 553 signals in 83 families.  This proves
only name/family uniqueness; it does not establish statistical independence.

```powershell
py -3.11 harness/tools/compose_research_factor_catalog.py `
  --vetted-v3-catalog D:/cbond_on/research_scratch/factor_mining_20260803_unified_v7_with_joint_catalog_v1/family_catalog.json `
  --module cbond_on.domain.factors.defs.research_factor_mining_intraday_depth_dominance_topology_v1 `
  --module cbond_on.domain.factors.defs.research_factor_mining_intraday_trade_grid_topology_v1 `
  --output-name factor_mining_20260803_v7_static_dedupe_depth_trade_grid_plan
```

The command ran without `--execute`; the displayed output path was not
created.  Because depth-dominance failed the smoke, only the trade-grid module
may be considered in a later composition after it independently completes the
full validation sequence.

## Boundary and next action

No aggregate, `defs.__init__`, configuration, live artifact, model, DB,
scheduler, mask, label, or trading rule changed.  Do not launch a full build
from this record.  The next permissible action is owner/parent scheduling of a
new, previously absent research scratch root for the trade-grid module; the
depth-dominance v1 catalogue stays excluded unless a separately designed,
versioned hypothesis changes the all-or-nothing tie semantics.
