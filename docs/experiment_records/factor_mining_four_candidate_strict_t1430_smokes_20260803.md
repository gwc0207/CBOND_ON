# Four factor-mining candidate strict-T1430 smokes (research-only)

## Scope and fixed smoke contract

This is a structural and data-quality smoke only, not an IC result, a
backtest, or a model/live admission.  The four existing research catalogues
were run independently for `2026-04-28..2026-04-29` through
`harness/tools/run_factor_mining_expansion.py` with explicit `--execute`.

- panel: clean-direct `T1430`; FactorStore output timestamps are exactly
  `14:30:00`; configured label time is `14:42`;
- engine: Python/Pandas, one worker; all backtest, screening, walk-forward,
  bad-factor-report stages disabled;
- each output has its own previously absent root below
  `D:/cbond_on/research_scratch`;
- no aggregate, `defs.__init__`, configuration, FactorStore production root,
  model, live, DB, scheduler, pool, label, mask, or trading rule changed.

## Static checks before execution

Focused validation plus generic runner tests passed:

```powershell
py -3.11 -m pytest -q \
  tests/test_research_factor_mining_daily_price_base_relations_v1.py \
  tests/test_research_factor_mining_daily_asymmetric_state_transitions_v1.py \
  tests/test_research_factor_mining_cross_asset_book_event_transmission_v1.py \
  tests/test_research_factor_mining_historical_execution_to_open_transition_state_v1.py \
  tests/test_run_factor_mining_expansion.py
```

Result: `27 passed`; Ruff passed for all four modules.

Each module contains 9 signals / 3 families.  The four modules have 36 unique
signals / 12 unique families, and have no signal or family-name collision with
the current v7 catalogue (`535` signals / `77` families).

Time visibility was checked from the source and existing focused tests:

- `daily_price_base_relations_v1` and
  `daily_asymmetric_state_transitions_v1` use the T1430 panel only for output
  keys and use exact strict-prior `daily_price` / `daily_base` histories;
- `cross_asset_book_event_transmission_v1` requires the supplied stock panel
  and bond-stock map, and restricts both current-day panels to continuous
  auction physical rows no later than `14:29`;
- `historical_execution_to_open_transition_state_v1` uses completed historical
  daily-TWAP triplets only, with a pair ending at `D+1` eligible only when
  `D+1 < T`.

## Completed-root audit

Every root has exactly two date files (`20260428`, `20260429`), one
research-only run manifest, and one family catalogue matching its source
catalogue.  All FactorStores have stable catalogue schema, MultiIndex
`(dt, code)`, unique combined index, zero `Inf`, and only `14:30:00`
timestamps.

| Module | Scratch root | Rows (Apr 28 / Apr 29) | Empty or cross-section-constant signals | Smoke disposition |
| --- | --- | ---: | --- | --- |
| `daily_price_base_relations_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_daily_price_base_relations_v1_strict_t1430_smoke_20260428_20260429` | 330 / 324 | none; all 9 nonempty and nonconstant | eligible for a future aggregate candidate, pending a fresh full-window build and fixed IC/redundancy screen |
| `daily_asymmetric_state_transitions_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_daily_asymmetric_state_transitions_v1_strict_t1430_smoke_20260428_20260429` | 330 / 324 | `svcr_relief_gap_fade_rate60` and `svcr_stress_relief_reversal_spread60` are all empty on both days; no constants among the other 7 | not eligible as the unfiltered 9-signal module; retain source for audit and use a later explicit healthy wrapper or diagnose the missing relief events |
| `cross_asset_book_event_transmission_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_cross_asset_book_event_transmission_v1_strict_t1430_smoke_20260428_20260429` | 330 / 324 | none; all 9 nonempty and nonconstant | eligible for a future aggregate candidate, but full-window coverage must be checked because strict mapping/event support yielded 95--118 finite names per day |
| `historical_execution_to_open_transition_state_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_historical_execution_to_open_transition_state_v1_strict_t1430_smoke_20260428_20260429` | 330 / 324 | none; all 9 nonempty and nonconstant | eligible for a future aggregate candidate, pending a fresh full-window build and fixed IC/redundancy screen |

The daily price/base module had 318--327 finite names per signal/day; the
healthy asymmetric signals had 148--319 (the stress event signal is about
45%, while the other six are broad); cross-asset event signals had 95--118;
and the historical execution/open signals had 307--314.  These figures are
two-day smoke evidence only, not coverage or IC evidence for the fixed
`2025-01-01` screen contract.

## Follow-on asymmetric healthy catalogue

The source implementation remains immutable.  A separate research-only view
now exists at
`cbond_on/domain/factors/defs/research_factor_mining_daily_asymmetric_state_transitions_healthy_catalog_v1.py`.
It exposes the seven nonempty/nonconstant source entries across three families
and excludes only `svcr_relief_gap_fade_rate60` and
`svcr_stress_relief_reversal_spread60`.

The wrapper returns the exact original ``CatalogEntry`` objects and the
existing `factor_mining_daily_asymmetric_state_transitions_v1` kernel; it
fail-closes if the source entry count, order, names, families, kernel, or
registry binding drift.  Its focused tests plus the source and generic runner
tests passed (`12 passed`), and Ruff passed.  Static comparison found 7
signals / 3 families and no overlap with v7.  Its full-window no-write
preflight passed against the still-absent scratch root:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_asymmetric_state_transitions_healthy_catalog_v1_full
```

The wrapper is not imported by `defs.__init__` or an aggregate.  A future
composition must substitute it for the original source catalogue, never append
both, because retained signal names are intentionally identical.

## Next step and boundary

Only the three fully healthy catalogues may be considered for a new aggregate
composition after the owner/parent task chooses a fresh full-window build.
The asymmetric source must not be appended unchanged.  Any later aggregate
must repeat global family/signal collision checks, full-root audit, fixed
T-1 `o_0005` IC/validity screen, and correlation gates.  No live promotion is
implied.
