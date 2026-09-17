# Shared T1429 PanelStore — 2026-08-27

## Objective

> Status: not a prerequisite for current factor construction. Normal live,
> experiment, and supplement factor jobs use clean-direct in-memory T1430
> panels from DataHub clean snapshots. Do not resume this proposal as an input
> dependency without a separate owner decision.

Establish one public CBOND_ON T1429 PanelStore at
`D:/cbond_on/panel_data`.  A dedicated PanelStore publisher is its only
writer.  Factor, model, research, and live-runtime consumers read the same
published panels and never write to that root.

## Fixed contract

- Logical panel: `T1430`; physical source cutoff: exactly `14:29:00`.
- Assets: `cbond` and `stock` beneath one PanelStore root.
- Construction: four-trading-day lookback, trailing 5,000 observations per
  code, `(dt, code, seq)` index, standard T1430 schema.
- Historical initial release: `2024-01-03..2026-07-30`, 622 trading days per
  asset.
- Input authority: DataHub raw calendar plus clean cbond/stock snapshots and
  their ready publish evidence.
- Consumer authority: read-only cached panels only; factor/model outputs
  retain their own designated roots.

## Current facts

- The configured public root currently does not exist.
- DataHub raw calendar, clean cbond snapshots, and clean stock snapshots agree
  on all 622 historical dates.
- Estimated initial storage is 84–90 GiB; D: has sufficient space.
- A one-worker publisher is required for the initial release because the
  current host has limited free memory while DataHub remains resident.
- No live config, scheduler, DB target, model state, factor release, or
  strategy contract has been changed.  No scheduler restart is in scope.
- A three-day scratch publication smoke passed on 2026-06-10..2026-06-12:
  both assets have the required `(dt, code, seq)` schema, 5,000 rows per code,
  maximum `trade_time=14:29:00`, and paired manifest/.done evidence.
- The full public release was launched at 2026-08-27T13:56+08:00 for
  2024-01-03..2026-08-26 (641 complete DataHub days), one worker, low process
  priority.  Its logs are below `D:/cbond_on/research_scratch/` with prefix
  `shared_t1429_panel_store_public_20260827`.
- A guarded chain watcher will audit the public release, then run R88 as a
  read-only public PanelStore consumer; it will begin the fixed model study
  only after the new R88 manifest is complete.

## Release gates

1. Publisher preflight proves exact root, fixed panel contract, input calendar,
   output scope, and absence of a conflicting active publication.
2. A small scratch smoke proves schema, cutoff, cbond/stock coverage, atomic
   publication metadata, and reader compatibility.
3. The public historical release writes only through the dedicated publisher,
   records hashes and manifest/.done evidence, and passes complete coverage
   validation.
4. R88 is changed to read the released PanelStore in a fresh scratch run.
5. A no-DB live replay validates the staged live consumer before any active live
   configuration or scheduler change is proposed.

## Next action

Finish the publisher and consumer boundaries, run the scratch smoke, then
present the public-release command, output root, capacity, and no-DB
verification evidence before initial publication.
