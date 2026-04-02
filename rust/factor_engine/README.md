# CBOND_ON Rust Factor Engine

This crate provides Python extension module `cbond_on_rust` for factor computation.

## Build (local venv)

```bash
cd rust/factor_engine
python -m pip install maturin
python -m maturin develop --release
```

## Current behavior

- Python factor pipeline can route to Rust by setting `compute.engine = "rust"`.
- Runtime is **fail-fast**: if Rust is selected, no Python fallback is used.
- Rust kernels are implemented for the `lgbm_factor_MSE` dependency set:
  - `aacb`, `volen`, `ret_window`, `ret_open_to_time`, `mom_slope`, `volatility`,
    `range_ratio`, `price_position`, `volume_sum`, `amount_sum`, `vwap`,
    `volume_imbalance`, `spread`, `depth_imbalance`, `midprice_move`,
    `turnover_rate`, `amihud_illiq`, `microprice_bias`, `depth_slope`,
    `return_skew`, `vwap_gap`.
- Additional live-factor kernels implemented:
  - `order_flow_imbalance_v1`, `depth_weighted_imbalance_v1`, `intraday_momentum_v1`,
    `bid_ask_spread_v1`, `price_level_position_v1`, `volume_price_trend_v1`,
    `trade_intensity_v1`, `volatility_scaled_return_v1`,
    `alpha001_signed_power_v1` ~ `alpha010_close_change_rank_v1`.
- Current implemented total: 41 factors (see `factor_manifest.json`).
- Other factors are currently fail-fast with explicit `rust factor kernel not implemented: ...`.
- Suggested rollout:
  - `factor_config.compute.engine = "rust"` and `live_factors_config.compute.engine = "rust"` are both supported.
  - factor packs beyond the covered set remain fail-fast until kernels are added.

## Unified planning (windows / levels / time-ranges)

Rust runtime now builds a per-run factor plan from `specs` before computing:

- unique `window_minutes` set
- unique book `levels` set
- unique `ret_open_to_time` ranges (`start_time`, `end_time`)

Limits are configurable via compute config (both factor and live):

- `plan_max_windows` (default `8`)
- `plan_max_levels` (default `8`)
- `plan_max_time_ranges` (default `8`)
- `plan_log_summary` (default `true`)

If a run exceeds a configured limit, Rust fails fast with explicit error.

Coverage tracker:

- `factor_manifest.json` lists all registered factors (`total_factors=97`) and Rust implementation status.
