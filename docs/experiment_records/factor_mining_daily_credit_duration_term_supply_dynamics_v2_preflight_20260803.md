# Daily credit-duration-term-supply dynamics v2 (research-only preflight and smoke)

## Motivation and boundary

This is an independent research-only v2, not a modification of the audited
v1 module.  The v1 two-day smoke was structurally valid but exposed raw
convexity beta/extreme residual values consistent with small-denominator and
ill-conditioned regressions.  V1 and its smoke root remain preserved as audit
evidence and must not be added to a later aggregate.

V2 changes the estimator rather than clipping or filling the v1 output:

- every former single-predictor raw slope is now a Pearson correlation subject
  to an explicit minimum effective standard deviation of `1e-6` for both
  series;
- every multivariate innovation is now an out-of-sample residual in the prior
  target's own standard-deviation units;
- every such standardized regression fails closed unless each predictor has
  standard deviation above `1e-6`, the design is full rank, and its condition
  number is at most `1e4`.

No `clip`, winsorization, `fillna(0)`, fallback source, direct I/O, or use of
labels/PnL/scores is present.  Invalid, low-scale, or ill-conditioned history
stays `NaN`.

Added research-only files:

- `cbond_on/domain/factors/defs/research_factor_mining_daily_credit_duration_term_supply_dynamics_v2.py`;
- `tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_v2.py`.

V1 was not changed.  No aggregate, `defs.__init__`, configuration, production
FactorStore, model, live artifact, DB, scheduler, trading rule, mask, label,
or source data was changed.  The subsequent v2 two-day smoke wrote only its
dedicated research scratch root; no full build was started.

## Catalogue and strict-T-1 contract

Kernel:

```text
factor_mining_daily_credit_duration_term_supply_dynamics_v2
```

The catalogue has 12 signals in four distinct families:

| Family | Signals |
| --- | --- |
| `prior_credit_duration_convexity_motion_robust` | `cdmr_yield_duration_delta_corr20`, `cdmr_yield_convexity_delta_corr20`, `cdmr_duration_convexity_delta_corr20` |
| `prior_term_supply_roll_rebalancing_robust` | `tsrr2_supply_term_level_corr20`, `tsrr2_duration_term_ratio_delta5`, `tsrr2_supply_term_standardized_oos20` |
| `prior_credit_flow_absorption_dynamics_robust` | `cfar_yield_amount_capacity_delta_corr20`, `cfar_yield_volume_capacity_delta_corr20`, `cfar_yield_price_supply_standardized_oos20` |
| `prior_convexity_anchor_transition_robust` | `catr_convexity_duration_delta_corr20`, `catr_convexity_anchor_delta_corr20`, `catr_convexity_state_standardized_oos20` |

Only declared fields are used:

- `market_cbond.daily_price`: `close_price`, `volume`, `amount`;
- `market_cbond.daily_base`: `current_yield`, `duration`, `convexity`,
  `year_to_mat`, `remain_size`, `bond_prem_ratio`, `conv_value`, and
  `pure_redemption_value`.

The known broken debt-floor fields are not requested.  `daily_price` provides
the independently established last strict-prior anchor A; both daily sources
are filtered to `trade_date < T` before date-key joining.  An instrument needs
an exact base row at A and consecutive daily-price sessions for every history
window.  Supply-dependent signals intentionally retain their v1 coverage risk:
flat supply histories fail the effective-scale gate and return `NaN` rather
than a synthetic zero.

## Static collision checks and verification

Direct catalogue comparison found zero signal and family collisions with:

- v1;
- strict-PIT aggregate v3;
- immutable v4 candidate catalogue (`475` signals / `64` families);
- current v7 catalogue (`535` signals / `77` families).

V2 is not imported by `cbond_on/domain/factors/defs/__init__.py`.

Focused tests plus generic expansion-runner tests:

```powershell
py -3.11 -m pytest -q \
  tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_v2.py \
  tests/test_run_factor_mining_expansion.py
```

Result: `9 passed`.

The focused tests cover finite synthetic strict-prior outputs, bounded
correlations, score-date/future exclusion, stale/gapped history failure,
missing-field and duplicate-key rejection, invalid anchor behavior, explicit
low-scale rejection, and explicit high-condition-number rejection.

Ruff:

```powershell
py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_credit_duration_term_supply_dynamics_v2.py \
  tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_v2.py
```

Result: `All checks passed!`

The generic no-write preflight accepted the module as research-only with 12
signals / four families for the fixed `2025-01-01..2026-07-30` window,
T1430/14:30 factor time, 14:42 label time, Python engine, and all report
stages disabled:

```powershell
py -3.11 harness/tools/run_factor_mining_expansion.py \
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_daily_credit_duration_term_supply_dynamics_v2 \
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_daily_credit_duration_term_supply_dynamics_v2_full \
  --start 2025-01-01 --end 2026-07-30
```

No `--execute` was passed and the previously absent scratch root remained
absent afterward.

## Completed two-score-day smoke and numerical disposition

The isolated source-v2 smoke completed on `2026-04-28..2026-04-29` under:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_credit_duration_term_supply_dynamics_v2_smoke_20260428_20260429
```

It used the generic research-only launcher with `--execute`, while all
backtest, screening, walk-forward, and bad-factor report stages remained
disabled. The manifest is `research_only: true`, declares the 12 source
entries/four families above, and records catalogue SHA-256
`013b558c1f286ce1a69e7a97438fc0c6e4ebfd313ba6f74d73f19dd5c8b4013c`, which
matches the current v2 source file.

The structural audit found exactly the expected 12 columns in source order,
MultiIndex `(dt, code)`, unique combined rows, only `14:30:00` timestamps,
and zero `Inf` or `-Inf` values. There were 330 rows on 2026-04-28 and 324
on 2026-04-29. No source signal was all-empty or cross-sectionally constant
on either day. The eight Pearson-correlation outputs remained bounded, with a
combined observed absolute maximum of `0.9987866201`.

Coverage was high for the three `cdmr_*` signals, both flow correlations, and
both convexity correlations (`327/330` then `322/324` finite); the
duration-to-term ratio was complete on both days. The deliberately fail-closed
supply-level correlation was finite for `126/330` and `124/324` names. These
are smoke quality observations, not IC, return, or selection results.

Three standardized OOS innovations remain numerically unsuitable for any
aggregate despite the regression-history gates:

| Source signal | 2026-04-28 absolute maximum | 2026-04-29 absolute maximum | Disposition |
| --- | ---: | ---: | --- |
| `tsrr2_supply_term_standardized_oos20` | 11179.8078 | 54.1991 | exclude |
| `cfar_yield_price_supply_standardized_oos20` | 16.6724 | 0.8164 | exclude |
| `catr_convexity_state_standardized_oos20` | 46.8104 | 15.0958 | exclude |

The underlying issue is an exceptional current observation relative to a
low-variance prior target distribution; it is not repaired by silently
clipping, filling, or rescaling the source formula. Consequently v2 itself
must not enter an aggregate or full-window build. A separate healthy wrapper
may select the nine remaining original entries, but that pure entry filter is
not a new source build or an IC result. No live implication exists.
