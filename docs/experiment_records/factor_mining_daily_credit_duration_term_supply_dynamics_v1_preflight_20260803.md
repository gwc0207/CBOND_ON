# Daily credit-duration-term-supply dynamics v1 (research-only preflight)

## Scope and boundary

This record covers a new import-only, strict-T-1 daily factor catalogue.  It
contains the local implementation, focused-test, lint, generic-runner
no-write-preflight, and subsequently authorized two-score-day structural-smoke
evidence.  It is not a full-window screen, IC result, correlation result,
backtest, aggregate admission, model change, or live promotion.

Added research-only files:

- `cbond_on/domain/factors/defs/research_factor_mining_daily_credit_duration_term_supply_dynamics_v1.py`;
- `tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_v1.py`.

No aggregate catalogue, `defs.__init__`, configuration, production FactorStore
root, model, live artifact, DB, scheduler, trading rule, mask, label, or
source data was changed.  The only execution was the isolated two-day scratch
smoke documented below; no full build was started.

## Catalogue and time contract

Kernel:

```text
factor_mining_daily_credit_duration_term_supply_dynamics_v1
```

The module has 12 signals in four distinct families, with three continuous
signals per family:

| Family | Signals |
| --- | --- |
| `prior_credit_duration_convexity_motion` | `cdcm_yield_duration_delta_beta20`, `cdcm_yield_convexity_delta_beta20`, `cdcm_duration_convexity_delta_corr20` |
| `prior_term_supply_roll_rebalancing` | `tsrr_supply_term_level_beta20`, `tsrr_duration_term_ratio_delta5`, `tsrr_supply_term_oos_residual20` |
| `prior_credit_flow_absorption_dynamics` | `cfad_yield_amount_capacity_delta_beta20`, `cfad_yield_volume_capacity_delta_beta20`, `cfad_yield_price_supply_oos_residual20` |
| `prior_convexity_anchor_transition` | `cat_convexity_duration_delta_beta20`, `cat_convexity_anchor_delta_beta20`, `cat_convexity_state_oos_residual20` |

The only declared daily inputs are:

- `market_cbond.daily_price`: `close_price`, `volume`, `amount`;
- `market_cbond.daily_base`: `current_yield`, `duration`, `convexity`,
  `year_to_mat`, `remain_size`, `bond_prem_ratio`, `conv_value`, and
  `pure_redemption_value`.

The module does not request `debt_puredebt_ratio`, `puredebt_prem_ratio`, or
any other debt-floor field.  It uses `daily_price.close_price` to establish
the independent latest completed anchor A, filters *each* source to
`trade_date < T` before joining, and accepts an instrument only when its base
history has a row on that same A.  All window calculations require consecutive
daily-price sessions.  Invalid/nonpositive inputs and stale/gapped histories
remain `NaN`; there is no `fillna(0)` or fallback source.

The resulting implementation has no direct file, database, network, label,
score, PnL, mask, or trading-list I/O.  It consumes only the supplied factor
context and returns `(dt, code)`-aligned `pd.Series` values.

## Static duplicate check

The 12 candidate signal names and four family names were compared directly
against these current source-of-truth catalogues:

- strict-PIT aggregate v3 (`97` signals / `35` families);
- complete immutable v4 candidate catalogue at
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v4_all_complete_catalog_v1/family_catalog.json`
  (`475` / `64`);
- current v7 catalogue at
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v7_with_joint_catalog_v1/family_catalog.json`
  (`535` / `77`).

Result: zero signal-name collisions and zero family-name collisions in all
three comparisons.  The module is not imported by `defs.__init__`.

## Verification

The factor-backtest preflight was run first:

```powershell
py -3.11 harness/tools/agent_preflight.py --mode factor-backtest
```

Focused synthetic strict-PIT tests and the generic expansion-runner tests:

```powershell
py -3.11 -m pytest -q \
  tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_v1.py \
  tests/test_run_factor_mining_expansion.py
```

Result: `10 passed`.

The focused tests prove all 12 signals build finite values from complete
strict-prior histories, contain no `Inf`, ignore injected score-day/future
rows exactly, fail closed for stale anchors or a nonconsecutive base history,
reject missing fields and duplicate strict-prior rows, and preserve `NaN` for
invalid conversion-anchor inputs.

Ruff:

```powershell
py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_credit_duration_term_supply_dynamics_v1.py \
  tests/test_research_factor_mining_daily_credit_duration_term_supply_dynamics_v1.py
```

Result: `All checks passed!`

Generic no-write preflight (no `--execute`):

```powershell
py -3.11 harness/tools/run_factor_mining_expansion.py \
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_daily_credit_duration_term_supply_dynamics_v1 \
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_daily_credit_duration_term_supply_dynamics_v1_full \
  --start 2025-01-01 --end 2026-07-30
```

It accepted the module as research-only with 12 signals / four families,
T1430/14:30 factors and 14:42 labels, Python engine, all report stages
disabled, and a previously absent scratch root.  The root remained absent
after the command; no data build or derived output was created.

## Completed two-score-day T1430 smoke

After the independently running v4 aggregate smoke naturally completed and
parent coordination released resources, this module alone was built through
the same generic entrypoint:

```powershell
py -3.11 harness/tools/run_factor_mining_expansion.py \
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_daily_credit_duration_term_supply_dynamics_v1 \
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_daily_credit_duration_term_supply_dynamics_v1_smoke_20260428_20260429 \
  --start 2026-04-28 --end 2026-04-29 --execute
```

The resulting root is strictly isolated below research scratch.  Its one
research-only manifest and one family catalogue match the current source
module's SHA-256, catalogue version, family/signals/order, and all 12 expected
columns exactly.  The two FactorStore files are:

- `factor_data/factors/T1430/2026-04/20260428.parquet` (`330` rows);
- `factor_data/factors/T1430/2026-04/20260429.parquet` (`324` rows).

Both files have the same 12-column schema, MultiIndex `(dt, code)`, unique
combined index, only `14:30:00` timestamps, and zero `Inf` / `-Inf` cells.
All 12 signals are nonempty and cross-sectionally nonconstant on both dates.

| Signal group | 2026-04-28 finite rows | 2026-04-29 finite rows | Structural observation |
| --- | ---: | ---: | --- |
| `cdcm_*` (3), `cfad_*_capacity_*` (2), and `cat_*` (3) | 327/330 each | 322/324 each | broad coverage; no daily constant |
| `tsrr_duration_term_ratio_delta5` | 330/330 | 324/324 | broad coverage; 8.8% / 7.7% exact formula zeros, but not constant |
| `tsrr_supply_term_level_beta20`, `tsrr_supply_term_oos_residual20` | 134/330 each | 133/324 each | about 41% finite; strict variation/conditioning requirement rejects the rest |
| `cfad_yield_price_supply_oos_residual20` | 134/330 | 136/324 | about 41--42% finite; shares the strict supply-variation dependence |

This is a successful structural build but **not** a clean numerical-health
result.  The three convexity signals below have materially unstable raw scale:

- `cat_convexity_duration_delta_beta20`: `[-12,836.47, 4,455.37]` over the
  smoke;
- `cat_convexity_anchor_delta_beta20`: `[-0.0848, 8,543.28]`;
- `cat_convexity_state_oos_residual20`: `[-37.49, 553.05]`.

The values are finite and nonconstant, but are consistent with small predictor
variation or ill-conditioned prior regressions.  The current code rejects
exactly constant/rank-deficient inputs only; it has no condition-number gate or
standardized-beta transformation.  No formula was changed after this audit.
Treat the module as structurally smoke-validated but numerically yellow until
the owner/parent decides whether a more robust design is warranted.  The
observed 41% supply-dependent coverage also needs full-window evidence before
any screen or aggregate consideration.

## Next step and non-result

This catalogue is a future research-only candidate, not an admitted factor
set.  Before a full-root build, its numerical conditioning and supply-coverage
warnings must be resolved or explicitly accepted.  It then still needs an
isolated full-root structural audit, fixed `2025-01-01` IC/validity screen on
the T-1 `o_0005` pool, and the global redundancy rule before any signal can be
considered for a selected research list.  The
required thresholds remain `abs(mean daily Pearson IC) > 0.02`, at least 250
valid days, and mean daily max-absolute Pearson/Spearman redundancy `<0.80`
within family / `<0.70` across families with at least 200 common valid days.
No live implication exists.
