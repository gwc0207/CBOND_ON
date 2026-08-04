# Intraday × T-1 State Interaction Factor Mining v1 (r5 smoke)

Date: 2026-08-03

Status: research-only candidate catalogue; smoke-validated only.  It has not
been run across the requested IC window, screened, admitted to a model, or
promoted to any live configuration.

## Question and fixed contract

Can new information be added beyond the existing intraday, daily, hybrid, and
cross-asset catalogues by interacting a physically visible same-day T1430
state with a strictly completed T-1 state *change, event, or memory*?

The eventual fixed screen remains:

- requested IC start: `2025-01-01` (first usable trading session is
  `2025-01-02`);
- fixed previous-trading-day `o_0005` universe;
- T1430 panel, visible intraday data only through physical `14:29:00`;
- `14:42` label contract;
- absolute mean daily Pearson IC strictly above `0.02`;
- within-family absolute correlation strictly below `0.80` and cross-family
  absolute correlation strictly below `0.70` on the later unified screen.

This module does not calculate IC itself and does not loosen any later gate.

## Implementation

New research-only files:

- `cbond_on/domain/factors/defs/research_factor_mining_intraday_state_interaction_v1.py`
- `tests/test_research_factor_mining_intraday_state_interaction_v1.py`

The module is deliberately excluded from `defs.__init__`, factor contracts,
live configuration, model feature lists, scheduler configuration, production
FactorStore, and database writes.

It defines 36 signals in six families:

1. `isi_defensive_path_state`
   - `isi_def_tail_eff_x_floor_change5`
   - `isi_def_tail_jump_x_redemption_prem_z20`
   - `isi_def_gapabsorb_x_convexity_duration`
   - `isi_def_terminal_loc_x_maturity_duration_gap`
   - `isi_def_noise_x_premium_yield_innovation`
   - `isi_def_reclaim_x_floor_vol20`
2. `isi_supply_event_response`
   - `isi_supply_flowaccel_x_shrink1`
   - `isi_supply_tail_impact_x_recency60`
   - `isi_supply_depthrec_x_eventrate20`
   - `isi_supply_gapabsorb_x_cumchange20`
   - `isi_supply_midlast_x_shrink_liq_impulse`
   - `isi_supply_lullrelease_x_change1`
3. `isi_call_contract_book_gate`
   - `isi_call_imbalance_x_active_age`
   - `isi_call_spread_shift_x_progress_velocity`
   - `isi_call_depthrec_x_required_days`
   - `isi_call_midlead_x_barrier_asym`
   - `isi_call_gapabsorb_x_contract_recency`
   - `isi_call_midlead_x_active_age`
4. `isi_adjustment_discontinuity_repricing`
   - `isi_adj_open_gap_x_net_shift`
   - `isi_adj_gapabsorb_x_cross_gap`
   - `isi_adj_midlast_x_absimbalance`
   - `isi_adj_tailflow_x_magnitude`
   - `isi_adj_rotation_x_recency`
   - `isi_adj_quotephase_x_eventrate`
5. `isi_liquidity_execution_memory`
   - `isi_liq_tailvwap_x_lagflowbeta`
   - `isi_liq_flowlead_x_highflow_response`
   - `isi_liq_depthimpact_x_impactmemory`
   - `isi_liq_gapresid_x_amount_persistence`
   - `isi_liq_eventimpact_x_amihud_z`
   - `isi_liq_lull_x_twap_transition`
6. `isi_cross_asset_daily_regime_gate`
   - `isi_cross_phase_rotation_x_beta_shift`
   - `isi_cross_tailcojump_x_residvol`
   - `isi_cross_bookcoherence_x_volforecast`
   - `isi_cross_stockshockbook_x_trackingresid`
   - `isi_cross_tailrange_x_liqsharediv`
   - `isi_cross_quotechannel_x_barrierdist`

The first five families use only the bond T1430 path plus strict prior daily
state.  The last family requires `stock_panel`, but its mapping is only the
exact prior-session `daily_base.stock_code`; it never consumes the score-day
mapping or `ctx.bond_stock_map`.

## PIT and source boundary

- Intraday rows require indexed score date **and** a physical `trade_time` on
  that score date, in a continuous market session, no later than `14:29:00`.
- All daily data comes exclusively from declared context requirements:
  `market_cbond.daily_price`, `market_cbond.daily_base`, and
  `market_cbond.daily_twap`.
- Every daily source is explicitly filtered to `trade_date < score_date`.
  `daily_price` supplies the common latest completed session anchor; a code
  without an exact `daily_base` row on that anchor emits `NaN`, rather than
  carrying stale state forward.
- The actual local `daily_base` stock-side field convention is `stk_*`:
  `stk_amount`, `stk_deal`, and `stk_prev_close_price`.  The module uses these
  names and does not guess alternate fields.
- Counter resets fail closed except for the pre-existing repository contract
  that retains a bounded signed correction in `amount` (absolute <= 100 and
  relative <= `1e-5`); no value is clipped or filled with zero.

The design avoids the existing exact hybrids: current flow × premium,
moneyness, floor, or duration; current return × trigger state; quote pressure
× stock-volatility; liquidity recovery × remaining size; current flow ×
overnight response; range × TWAP curve; and impact × turnover.

## Verification

Focused tests and generic runner contract:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
py -3.11 -m pytest -q `
  tests/test_research_factor_mining_intraday_state_interaction_v1.py `
  tests/test_run_factor_mining_expansion.py -p no:cacheprovider
py -3.11 -m ruff check `
  cbond_on/domain/factors/defs/research_factor_mining_intraday_state_interaction_v1.py `
  tests/test_research_factor_mining_intraday_state_interaction_v1.py
```

Result: `10 passed`; Ruff and `git diff --check` passed.

The tests cover catalogue cardinality, context declarations, score/future daily
contamination, nonphysical/stale panel rows, missing daily sources, duplicate
strict-prior keys, stale daily-base anchors, mapping isolation, NaN, and Inf.

The no-write launcher preflight also passed for the eventual fresh full root:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_intraday_state_interaction_v1_full
```

No full build was started.

## Real DataHub single-day smoke

The accepted immutable smoke root is:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_intraday_state_interaction_v1_r5_smoke_20260428
```

Command:

```powershell
py -3.11 harness/tools/run_factor_mining_expansion.py `
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_intraday_state_interaction_v1 `
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_intraday_state_interaction_v1_r5_smoke_20260428 `
  --start 2026-04-28 --end 2026-04-28 --execute
```

Output FactorStore:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_intraday_state_interaction_v1_r5_smoke_20260428/factor_data/factors/T1430/2026-04/20260428.parquet
```

Observed output:

- 330 unique `(dt, code)` rows;
- 36 expected candidate columns;
- 10,506 finite cells and 1,374 `NaN` cells;
- zero `Inf` / `-Inf` cells;
- every column has at least one finite value (range 235--306 finite rows);
- zero constant columns on this real one-day sample.

The first smoke root failed fast before factor output because the provisional
module requested non-existent `daily_base.stock_*` aliases.  It is retained as
audit-only and was not reused.  r2--r4 were likewise retained as audit-only
scratch outputs while a single degenerate call candidate was revised.  The r5
root above is the final smoke evidence for this version.

## Next action and caveat

Wait for the current active research builds to finish naturally and for the
resource gate to permit a *new, empty* full root.  Then build the fixed
`2025-01-01..2026-07-30` range, merge only immutable complete roots, and let
the existing unified fixed-universe screen decide IC and both correlation
thresholds.  Do not treat this smoke as evidence that any of the 36 factors
has `|IC| > 0.02` or should enter a model or live chain.
