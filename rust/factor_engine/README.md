# CBOND_ON Rust Factor Engine

This crate provides Python extension module `cbond_on_rust` for factor computation.

## Build (local venv)

```bash
cd rust/factor_engine
python -m pip install maturin
python -m maturin develop --release
```

## Windows rebuild (without venv, recommended for this repo)

When running from repo root (`cbond_on/`), local module path `cbond_on_rust/` shadows site-packages.
After Rust changes, rebuild **and** replace local `.pyd` to avoid old ABI/signature mismatch.

```powershell
cd C:\Users\BaiYang\CBOND_ON\cbond_on\rust\factor_engine
$py = "C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\python.exe"
$env:PYO3_PYTHON = $py

# 1) build wheel
& $py -m pip install -U maturin
& $py -m maturin build --release -i $py

# 2) reinstall wheel into site-packages (optional but recommended)
$whl = Get-ChildItem .\target\wheels\cbond_on_rust-*.whl |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
& $py -m pip install --force-reinstall $whl.FullName

# 3) replace local repo module binary (important)
$tmp = Join-Path $env:TEMP "cbond_on_rust_wheel_unpack"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive -Path $whl.FullName -DestinationPath $tmp -Force
$pyd = Get-ChildItem "$tmp\cbond_on_rust\cbond_on_rust*.pyd" | Select-Object -First 1
Copy-Item $pyd.FullName "C:\Users\BaiYang\CBOND_ON\cbond_on\cbond_on_rust\" -Force

# 4) quick check
& $py -c "import cbond_on_rust,inspect;print(cbond_on_rust.__file__);print(inspect.signature(cbond_on_rust.compute_factor_frame))"
```

Expected signature after daily context upgrade:
`(panel_df, specs_payload, stock_df=None, map_df=None, daily_data=None, _compute_params=None)`

## Current behavior

- Python factor pipeline can route to Rust by setting `compute.engine = "rust"`.
- A `rust_first` run uses the ordinary public `compute_factor_frame` API for
  every requested factor instance; it has no Python or hybrid compute branch.
- Runtime is **fail-fast**: if Rust is selected, no Python fallback is used.
- The runtime authority is the **loaded**
  `cbond_on_rust.factor_capabilities()` payload, specifically its generic
  `factor_contracts` entries. Source files, this README, and
  `factor_manifest.json` are static documentation and cannot prove that an
  installed `.pyd` is current.
- The current profile-neutral capability ABI revision is
  `rust_factor_contracts_20260806_r1`; profile names such as `live50` do not
  belong to the core Rust capability surface.
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
    `alpha001_signed_power_v1` ~ `alpha010_close_change_rank_v1`, plus the
    frozen live50 alpha instances `alpha019`, `alpha024`, `alpha025`, `alpha030`,
    `alpha041`, `alpha050`, and `alpha078`.
- `factor_manifest.json` records 50 fully implemented catalog factor keys and
  13 research factor families implemented only for the exact frozen contracts
  listed there. A `rust_status` of `implemented_frozen_contracts_only` never
  authorizes an arbitrary `params.signal` variant.
- Other factors remain fail-fast with explicit `rust factor kernel not implemented: ...`.
- Suggested rollout:
  - `factor_config.compute.engine = "rust"` and
    `compute.execution_policy = "rust_first"` must be paired.
  - A new factor instance must be rejected unless the loaded extension declares
    an exact contract ID, output column, factor key, signal, and parameter hash.

## Frozen live50 contract

`live50_rust50_20260806` is one ordered, 50-instance Rust contract. The former
27/23 split is not a runtime category: all 50 instances enter one
`compute_factor_frame` call and one FactorStore/model feature path.

Its static sources are:

- pack: `cbond_on/config/factor/packs/live_screened_no_winsor_50_20260805.json5`;
- profile: `cbond_on/factor_contracts/profiles/live50_rust50_20260806.json5`;
- manifest: `factor_manifest.json` → `frozen_contracts`.

The manifest stores the ordered 50 capability-shaped records
`{id, output_col, factor, signal, params_sha256}` and the profile's complete
`specs_sha256`. `tests/test_rust_factor_manifest_live50_contract.py` compares
all of those records against the pack and profile, so an ID, order, parameter,
or output-column change cannot silently drift. It does not load a production
binary; deployment validation must separately call `factor_capabilities()` on
the freshly loaded wheel.

## New experimental factors: Rust-first policy

Every new experiment follows the same runtime policy as live:

1. Implement and parity-test the Rust kernel first.
2. Give every runnable parameterized instance a unique `rust_contract_id`.
3. Add the exact `{id, output_col, factor, signal, params_sha256}` record to
   the Rust capability table in the built extension, and record its catalog or
   frozen-contract scope in `factor_manifest.json`.
4. Run with `compute.engine = "rust"` and
   `compute.execution_policy = "rust_first"`; the loaded capability handshake
   must pass before panel/FactorStore work begins.

Python may remain only as a parity/reference implementation. It is not a
normal research, batch, or live fallback. Reproducing an archived Python result
requires the explicit root-config exception
`execution_policy = "legacy_reference_only"`, `legacy_reference_only: true`,
and a non-empty `legacy_reference_reason`.

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

- `factor_manifest.json` schema v2 contains a 112-key static catalog plus the
  immutable live50 instance contracts. Its `total_factors` counts catalog keys,
  not parameterized instance contracts; only loaded `factor_capabilities()`
  decides runtime executability.
