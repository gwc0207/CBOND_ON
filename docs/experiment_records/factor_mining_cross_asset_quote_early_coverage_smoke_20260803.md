# Cross-asset and quote-execution early-coverage strict-T1430 smoke (research-only)

## Question and scope

The later full IC contract starts at `2025-01-01`; this short smoke checks
whether three catalogues that were healthy on 2026-04-28/29 are already
mechanically usable at the beginning of that requested window. It is **not**
an IC calculation, correlation screen, backtest, model admission, or full
build.

- score dates: `2025-01-02`, `2025-01-03`, `2025-01-06`, `2025-01-07`,
  `2025-01-08`, and `2025-01-09`;
- factor contract: T1430 / `14:30`, physical continuous-session observations
  through `14:29:00`; label configuration `14:42`;
- source mode: DataHub-only `clean_direct`, Python engine;
- outputs: new, independent roots under `D:/cbond_on/research_scratch`;
- no full-window job, aggregate change, `defs.__init__` change, configuration,
  live artifact, model, DB, scheduler, mask, label, or trading-rule change.

Each run has one `research_only: true` manifest and disabled backtest,
screening, walk-forward, and bad-factor-report stages.

## Integrity result

All three outputs have six exact score-day files, exact catalogue column
schemas, a unique `(dt, code)` index on each file and in the six-day union,
date-aligned indices, and zero `Inf` / `-Inf`. No signal is all-empty or
cross-sectionally constant on any date or over the full smoke union.

| Catalogue | Scratch root | Signals | Union rows | Finite coverage | Result |
| --- | --- | ---: | ---: | --- | --- |
| `cross_asset_activity_calendar_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_cross_asset_activity_calendar_v1_smoke_20250102_20250109` | 6 | 3,020 | every signal: 3,010 / 3,020 | healthy at early-window sample |
| `cross_asset_event_sequence_topology_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_cross_asset_event_sequence_topology_v1_smoke_20250102_20250109` | 6 | 3,020 | five signals: 3,010 / 3,020; `xca_joint_burst_cluster_continuity`: 2,965 / 3,020 | healthy at early-window sample |
| `quote_execution_dynamics_v1` | `D:/cbond_on/research_scratch/factor_mining_20260803_quote_execution_dynamics_v1_smoke_20250102_20250109` | 6 | 3,020 | every signal: 2,773 / 3,020 | healthy at early-window sample |

For audit, activity-calendar finite counts by date were
`502, 502, 501, 501, 501, 503` for every signal. Event-sequence has the same
counts for five signals and `489, 497, 495, 486, 497, 501` for
`xca_joint_burst_cluster_continuity`. Quote-execution has
`456, 458, 470, 452, 488, 449` for every signal.

## Interpretation and boundary

The result removes a limited mechanical risk: these modules are not merely
2026-shaped successes. It does not demonstrate factor efficacy, stability, or
low correlation. The next permissible step, only when separately scheduled,
is a fresh full-window research build followed by a new all-complete-roots
merge and the fixed T-1 `o_0005` IC, chronology, and correlation gates. Do not
append these signals directly to the v7 selection or any live/model/aggregate
pack from this record.
