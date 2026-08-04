# Credit-duration-term-supply healthy wrapper (research-only)

## Scope and boundary

This is a pure, fail-closed catalogue selection over the unchanged source
module `research_factor_mining_daily_credit_duration_term_supply_dynamics_v2`.
It changes no formula, field, strict-T-1 rule, parameter, cache key,
registration, source data, aggregate catalogue, `defs.__init__`, configuration,
FactorStore, model, mask, trading rule, live artifact, DB, or scheduler.

The new view is:

```text
cbond_on.domain.factors.defs.research_factor_mining_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1
```

It returns nine original source `CatalogEntry` objects in source order and
retains all four source families:

| Family | Retained signals |
| --- | --- |
| `prior_credit_duration_convexity_motion_robust` | `cdmr_yield_duration_delta_corr20`, `cdmr_yield_convexity_delta_corr20`, `cdmr_duration_convexity_delta_corr20` |
| `prior_term_supply_roll_rebalancing_robust` | `tsrr2_supply_term_level_corr20`, `tsrr2_duration_term_ratio_delta5` |
| `prior_credit_flow_absorption_dynamics_robust` | `cfar_yield_amount_capacity_delta_corr20`, `cfar_yield_volume_capacity_delta_corr20` |
| `prior_convexity_anchor_transition_robust` | `catr_convexity_duration_delta_corr20`, `catr_convexity_anchor_delta_corr20` |

The exact excluded source entries are:

- `tsrr2_supply_term_standardized_oos20`;
- `cfar_yield_price_supply_standardized_oos20`;
- `catr_convexity_state_standardized_oos20`.

## Source-smoke evidence and disposition

The immutable source-v2 two-score-day smoke at
`D:/cbond_on/research_scratch/factor_mining_20260803_daily_credit_duration_term_supply_dynamics_v2_smoke_20260428_20260429`
had `research_only: true`, matching source SHA-256
`013b558c1f286ce1a69e7a97438fc0c6e4ebfd313ba6f74d73f19dd5c8b4013c`, 654
unique `(dt, code)` rows, source-order schema, only `14:30:00` timestamps, and
zero `Inf`/`-Inf`. All nine retained signals were nonempty and
cross-sectionally nonconstant on both days. The source record contains the
full audit: `factor_mining_daily_credit_duration_term_supply_dynamics_v2_preflight_20260803.md`.

The same smoke observed absolute maxima of 11179.8078, 16.6724, and 46.8104
for the three excluded standardized-OOS entries on 2026-04-28. Those
current-observation scale excursions make the source-v2 twelve-signal
catalogue unsuitable for aggregate admission. This wrapper does not cure or
hide them; it omits the exact three entries and preserves the source artifacts
for audit.

No direct wrapper smoke was run: it only filters original source entries and
keeps their kernel. The source smoke remains the signal-level quality evidence.
This is not generic runner-wiring validation, an IC/backtest result, or a
selection result.

## Fail-closed provenance checks

The wrapper rejects execution if any source contract drifts:

- source catalogue version;
- exact twelve-entry order or duplicate signal names;
- source family mapping;
- source kernel name; or
- `FactorRegistry` identity for
  `FactorMiningDailyCreditDurationTermSupplyDynamicsV2`.

It is deliberately absent from `defs.__init__` and every aggregate. A future
research aggregate must substitute this nine-entry view for v2, not append
both catalogues, because retained names are identical.

## Static collision and verification evidence

The nine retained signal names and four family names have zero overlap with
the current research-only source v1 (12 signals/four families), aggregate v3
(97/35), aggregate v4 (193/67), and immutable v7 family catalogue (535/77).
The v7 file is a direct family-to-signal map; all 535 unique signal names and
77 family names were compared read-only.

Focused validation:

```powershell
py -3.11 -m pytest -q \
  tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_v2.py \
  tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1.py \
  tests/test_run_factor_mining_expansion.py
```

Result: `11 passed`.

```powershell
py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1.py \
  tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1.py
```

Result: `All checks passed!`

The generic no-write preflight accepted the view as research-only: 9 signals /
4 families, source kernel unchanged, fixed `2025-01-01..2026-07-30` window,
T1430/14:30 factor time, 14:42 label time, Python engine, and all reports
disabled:

```powershell
py -3.11 harness/tools/run_factor_mining_expansion.py \
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1 \
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1_full \
  --start 2025-01-01 --end 2026-07-30
```

No `--execute` was passed; the previously absent scratch root remained absent.
No full build has been started, and no live implication exists.
