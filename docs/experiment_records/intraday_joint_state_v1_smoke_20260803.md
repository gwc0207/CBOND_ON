# Intraday joint-state v1 — strict-PIT smoke record (2026-08-03)

## Status and question

This is a research-only catalogue smoke record.  The question is whether a
new set of stock/bond joint price-discovery, tail-condition, and state
interaction factors can be constructed from the fixed T1430 contract without
changing any current trading rule, mask, model, score, live configuration,
database, scheduler, or production FactorStore.

This record is **not** an IC result, a correlation-screen result, a backtest,
or a promotion decision.  No full-window build was started.

## Candidate catalogue

Implementation:

```text
cbond_on/domain/factors/defs/research_factor_mining_intraday_joint_state_v1.py
kernel: factor_mining_intraday_joint_state_v1
catalogue version: 20260803_intraday_joint_state_v1
```

The catalogue contains 42 candidate signals in seven distinct families.  All
exact formula strings are preserved in the module's `FORMULAS` registry.

| Family | Candidates | Joint hypothesis |
| --- | --- | --- |
| `joint_opening_gap_transmission` | `joint_gap_signed_alignment`, `joint_stock_gap_bond_early_response`, `joint_bond_gap_stock_early_response`, `joint_gap_relative_absorption`, `joint_gap_phase_pass_through_asymmetry`, `joint_gap_tail_resolution_alignment` | Opening discontinuities and their subsequent resolution may identify cross-asset transmission that a full-session return gap misses. |
| `joint_phase_vector_coherence` | `joint_phase_return_cosine`, `joint_phase_centered_return_corr`, `joint_phase_direction_agreement_share`, `joint_phase_transition_agreement`, `joint_phase_mismatch_energy`, `joint_phase_tail_rotation_gap` | Four fixed clock phases describe joint agreement and rotation rather than a single contemporaneous beta or extreme timestamp. |
| `joint_quote_trade_channel_transmission` | `joint_stock_mid_bond_last_tail_response`, `joint_stock_last_bond_mid_tail_response`, `joint_bond_mid_stock_last_tail_response`, `joint_bond_last_stock_mid_tail_response`, `joint_cross_quote_trade_dynamic_gap`, `joint_cross_quote_trade_phase_shift_gap` | A mapped asset can initiate price discovery in midpoint quotes versus traded last prices, with different downstream transmission. |
| `joint_stock_shock_bond_book_gate` | `joint_stock_early_x_bond_imbalance`, `joint_stock_early_x_bond_spread`, `joint_stock_late_x_bond_imbalance_shift`, `joint_stock_late_x_bond_spread_change`, `joint_stock_full_x_bond_depth_recovery`, `joint_stock_tail_x_bond_microprice_bias` | The same stock shock can have different bond implications under the bond's contemporaneous book state. |
| `joint_cross_book_price_elasticity` | `joint_book_delta_imbalance_return_coupling_gap`, `joint_book_delta_spread_return_coupling_gap`, `joint_book_delta_depth_return_coupling_gap`, `joint_book_delta_microprice_return_coupling_gap`, `joint_book_tail_imbalance_coupling_gap`, `joint_book_phase_coupling_shift_gap` | Compare each asset's own same-tick book-state-change/return coupling, not static cross-asset book-level differences. |
| `joint_cross_book_state_synchrony` | `joint_book_imbalance_transition_alignment`, `joint_book_spread_transition_alignment`, `joint_book_depth_transition_alignment`, `joint_book_microprice_transition_alignment`, `joint_book_state_transition_coherence`, `joint_book_transition_x_tail_return_gap` | Early-to-late book relocation can be shared or disconnected across the mapped pair. |
| `joint_tail_cojump_containment` | `joint_tail_signed_cojump`, `joint_tail_range_coexpansion`, `joint_tail_efficiency_alignment`, `joint_tail_jump_intensity_alignment`, `joint_tail_terminal_location_coshock`, `joint_tail_drawdown_containment` | Tail co-jumps, path containment, and range co-expansion are distinct from a tail-beta residual. |

The design was audited against existing strict-PIT v3 cross-asset families
(`stock_intraday_lead_lag`, `stock_bond_beta_residual`,
`stock_bond_path_asynchrony`, `stock_bond_microstructure_divergence`,
`stock_bond_liquidity_transmission`, `stock_bond_limit_stress`), the 13
intraday-combined families, the three single-asset microstructure-v2 families,
and the six single-bond cross-sectional-residual families.  The new catalogue
does not reuse an existing signal name or implement a window-only variant of
those formulas.

## Inputs and time contract

Current-day inputs are restricted to the supplied context:

```text
ctx.panel / ctx.stock_panel:
  trade_time, pre_close, open, last,
  ask_price1, bid_price1, ask_volume1, bid_volume1

ctx.daily_data:
  market_cbond.daily_price: exchange_code, close_price
  market_cbond.daily_base:  exchange_code, stock_code
```

- The registered factor declares `requires_stock_panel = True`.
- It deliberately declares `requires_bond_stock_map = False` and never reads
  `ctx.bond_stock_map`.  The generic map loader can resolve a score-day map;
  that is not treated as a sufficient T-1 mapping certificate here.
- The mapping instead uses only `daily_base.stock_code` on the latest common
  completed `daily_price` session strictly before score day `T`.  A bond must
  have both daily-base and daily-price rows on that exact prior session; a
  stale base mapping returns `NaN`.
- Both panels are filtered independently to the physical score date, continuous
  trading sessions, and `trade_time <= T 14:29:00`.  Rows at `14:29:30`,
  `14:30:00`, after the cutoff, before open, during lunch, or relabelled from
  another physical day are excluded.
- No file reads, DB/network calls, implicit fill, label, pool/mask, score, or
  strategy output exists in factor code.  Invalid/missing mapping, prices, or
  L1 quotes remain explicit `NaN`.

The overall research comparison contract remains the factor-mining fixed
calendar `2025-01-01..2026-07-30` (first actual trading date `2025-01-02`),
T1430 / 14:30 factor timestamp, same-day 14:42 label, and T-1 `o_0005`
universe.  The smoke below exercises only one date and does not evaluate IC.

## Focused verification

```powershell
py -3.11 -m pytest -q tests/test_research_factor_mining_intraday_joint_state_v1.py -p no:cacheprovider
# 7 passed

py -3.11 -m ruff check `
  cbond_on/domain/factors/defs/research_factor_mining_intraday_joint_state_v1.py `
  tests/test_research_factor_mining_intraday_joint_state_v1.py
# All checks passed
```

The focused tests verify catalogue/registry shape, no `+/-inf`, strict prior
daily mapping despite a deliberately conflicting context map, failure on a
stale base mapping, physical-day filtering including `14:29:30` and `14:30`,
missing L1 field failure, and unknown-signal failure.

## Real single-day scratch smoke

Command:

```powershell
py -3.11 harness/tools/run_factor_mining_expansion.py `
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_intraday_joint_state_v1 `
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_intraday_joint_state_v1_smoke_20260428 `
  --start 2026-04-28 --end 2026-04-28 --execute
```

All derived outputs stayed under:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_intraday_joint_state_v1_smoke_20260428
```

The completed FactorStore is:

```text
.../factor_data/factors/T1430/2026-04/20260428.parquet
```

Its audit results were:

| Check | Result |
| --- | ---: |
| `(dt, code)` rows | 330, unique |
| Factor timestamp | only `2026-04-28 14:30:00` |
| Catalogue columns | 42 / 42 |
| Finite cells | 12,647 |
| Explicit NaN cells | 1,213 |
| `+/-inf` cells | 0 |
| Per-column finite coverage | 291--302 rows |
| Empty or constant candidate columns | 0 |
| Run manifest | `research_only=true`, 42 signals, 7 families |

The NaNs are retained rather than imputed; they arise from unavailable strict
T-1 mappings or valid-path/book requirements.  This one-day smoke is only a
structural/data-availability result and cannot establish predictive IC.

## Boundary and next authorized step

No live config, model config, live factor list, FactorStore, production result
root, DB, scheduler, score, backtest, mask, or strategy rule was changed.  No
full-window process is running for this catalogue.

After owner review and capacity approval, the proposed previously-absent full
scratch root is:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_intraday_joint_state_v1_full
```

Before any full build, re-run the expansion-launcher no-write preflight against
that exact root.  After a complete 381-day store passes schema/index/Inf and
coverage audits, it must enter the unified fixed-pool IC and redundancy screen;
only that later screen can determine whether any candidate meets the requested
`|mean daily Pearson IC| > 0.02` and correlation gates.
