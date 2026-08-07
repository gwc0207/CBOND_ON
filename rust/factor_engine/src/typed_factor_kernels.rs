//! Typed Rust implementations for exact factor contracts.
//!
//! `compute_factor_frame` calls these kernels internally for the relevant
//! factor functions. The public runtime has one Rust entrypoint and one output
//! assembly path for every requested spec.

use crate::typed_factor_daily::{
    compute_p1_daily_signal, TypedFactorDailyBaseRow, TypedFactorDailyContext,
    TypedFactorDailyPriceRow, TypedFactorDailyTwapRow,
};
use crate::typed_factor_daily_cross_asset::{
    TypedFactorCrossAssetBaseRow, TypedFactorCrossAssetPriceRow, TypedFactorDailyCrossAssetContext,
};
use crate::typed_factor_daily_information::{
    compute_daily_information_signal, rjst_amount_joint_transition_nonzero_counts,
};
use crate::typed_factor_daily_paths::{
    dret_drawup_drawdown_asym, TypedFactorDailyPathContext, TypedFactorDailyPathPriceRow,
    TypedFactorDailyPriceAnchorRow, TypedFactorDailyTrackingBaseRow,
    TypedFactorDailyTrackingContext,
};
use crate::typed_factor_daily_rank_state::{
    TypedFactorRankStateBaseRow, TypedFactorRankStateContext, TypedFactorRankStatePriceRow,
};
use crate::typed_factor_intraday::{
    lrd_cross_side_reprice_symmetry, quote_execution_typed_factor, session_label, BookRow,
    QedTypedFactorMetrics, QuoteRow, QED_AT_QUOTE_TOL,
};
use chrono::NaiveDate;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BTreeSet, HashSet};

const SOURCE_BASE: &str = "market_cbond.daily_base";
const SOURCE_PRICE: &str = "market_cbond.daily_price";
const SOURCE_TWAP: &str = "market_cbond.daily_twap";
const QED_FACTOR: &str = "factor_mining_quote_execution_dynamics_v1";
const LRD_FACTOR: &str = "factor_mining_orderbook_repricing_v1";
const QED_SIGNALS: &[&str] = &[
    "qed_prior_quote_location_dispersion",
    "qed_prior_quote_tail_penetration",
    "qed_prior_quote_lag2_agreement",
];
const LRD_SIGNAL: &str = "lrd_cross_side_reprice_symmetry";
const QED_REQUIRED_PANEL_COLUMNS: &[&str] = &[
    "trade_time",
    "last",
    "ask_price1",
    "bid_price1",
    "num_trades",
];

/// Whether a factor uses the typed Rust kernel rather than the standard
/// panel-kernel implementation.  This is a kernel implementation detail, not
/// a live-factor cohort or a second public execution route.
pub(crate) fn is_typed_factor_family(factor: &str) -> bool {
    matches!(
        factor,
        "factor_mining_daily_catalog_v1"
            | "factor_mining_daily_bond_stock_return_flow_information_v1"
            | "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1"
            | "factor_mining_daily_contract_stock_v1"
            | "factor_mining_daily_expansion_v1"
            | "factor_mining_daily_ohlc_wick_path_asymmetry_v1"
            | "factor_mining_daily_incremental_v1"
            | "factor_mining_daily_liquidity_channel_composition_v1"
            | "factor_mining_orderbook_repricing_v1"
            | "factor_mining_daily_capacity_rank_coupling_v1"
            | "factor_mining_quote_execution_dynamics_v1"
            | "factor_mining_daily_return_liquidity_topology_v1"
            | "factor_mining_daily_asymmetric_state_transitions_v1"
    )
}

// `factor_mining_daily_catalog_v1` declares this complete source contract for
// every catalogue signal, including the one base-gap output admitted to
// live50.  The kernel bridge validates it even though the formula itself only
// consumes two fields, so missing-column behaviour remains fail-closed.
const CATALOG_BASE_COLUMNS: &[&str] = &[
    "year_to_mat",
    "remain_size",
    "current_yield",
    "cb_conv_price",
    "turnover_rate",
    "stock_code",
    "stock_close_price",
    "bond_prem_ratio",
    "debt_puredebt_ratio",
    "puredebt_prem_ratio",
    "conv_value",
    "ytm",
    "duration",
    "modify_duration",
    "convexity",
    "base_rate",
    "stock_volatility",
    "pure_redemption_value",
    "redemption_prem_ratio",
    "cb_prev_close_price",
    "cb_close_price",
    "cb_volume",
    "cb_amount",
    "cb_deal",
    "stk_volume",
    "stk_amount",
    "stk_deal",
    "cb_call_price",
    "trigger_is_price",
    "trigger_cum_days",
    "trigger_reach_days",
    "in_trigger_process",
    "trigger_price_revise",
    "trigger_cum_days_revise",
    "trigger_reach_days_revise",
];
const CATALOG_PRICE_COLUMNS: &[&str] = &[
    "prev_close_price",
    "act_prev_close_price",
    "close_price",
    "open_price",
    "high_price",
    "low_price",
    "volume",
    "amount",
    "deal",
];
const CATALOG_TWAP_COLUMNS: &[&str] = &[
    "twap_0930_0935",
    "twap_0935_1000",
    "twap_1000_1030",
    "twap_1100_1130",
    "twap_1300_1330",
    "twap_1330_1400",
    "twap_1400_1430",
    "twap_1430_1442",
    "twap_1430_1500",
    "twap_1442_1457",
];
const EXPANSION_TWAP_COLUMNS: &[&str] = &[
    "twap_0930_0935",
    "twap_0935_1000",
    "twap_1100_1130",
    "twap_1300_1330",
    "twap_1400_1430",
    "twap_1430_1442",
    "twap_1442_1457",
];
const INFORMATION_LCC_PRICE_COLUMNS: &[&str] = &["volume", "amount", "deal"];
const INFORMATION_TOPOLOGY_PRICE_COLUMNS: &[&str] =
    &["prev_close_price", "close_price", "amount", "deal"];
const RANK_STATE_PRCN_PRICE_COLUMNS: &[&str] = &["prev_close_price", "close_price", "amount"];
const RANK_STATE_PRCN_BASE_COLUMNS: &[&str] = &["remain_size"];
// The Python family validates its full source contract even though the selected
// fall-beta output reads only adjusted previous close, close, and current yield.
const RANK_STATE_YDPT_PRICE_COLUMNS: &[&str] = &[
    "act_prev_close_price",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
];
const RANK_STATE_YDPT_BASE_COLUMNS: &[&str] =
    &["current_yield", "duration", "convexity", "stock_volatility"];
const CROSS_BSFST_PRICE_COLUMNS: &[&str] = &["prev_close_price", "close_price", "amount"];
const CROSS_BSFST_BASE_COLUMNS: &[&str] = &["stk_prev_close_price", "stk_close_price"];
const CROSS_BSSRC_PRICE_COLUMNS: &[&str] = &["prev_close_price", "close_price"];
const CROSS_BSSRC_BASE_COLUMNS: &[&str] =
    &["stock_code", "stk_prev_close_price", "stk_close_price"];

#[derive(Clone, Debug)]
struct KernelSpec {
    factor: String,
    signal: String,
    output_col: String,
}

/// Parsed physical score-day intraday rows.  The map keys intentionally keep
/// the raw labelled `(dt, code)` values: Python's QED output is reindexed to
/// those labelled keys, while LRD emits only its physical-session groups.
#[derive(Default)]
struct TypedFactorIntradayContext {
    qed_rows: BTreeMap<(String, String), Vec<QuoteRow>>,
    lrd_rows: BTreeMap<(String, String), Vec<BookRow>>,
    lrd_keys: BTreeSet<(String, String)>,
}

#[derive(Default)]
struct SourceNeed {
    columns: BTreeSet<&'static str>,
    // Expansion/incremental kernels raise when a selected source is absent;
    // the catalog kernel treats an absent/empty source as empty history.
    missing_fails: bool,
    // This is a source-schema requirement, not a code-canonicalisation mode.
    // Expansion/incremental loaders require exchange_code explicitly even
    // though their code canonicalisers remain optional-exchange.
    exchange_required: bool,
}

#[derive(Default)]
struct ContractNeeds {
    base: SourceNeed,
    price: SourceNeed,
    twap: SourceNeed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CanonicalContract {
    OptionalExchange,
    StrictExchange,
}

/// Optional-exchange and strict-exchange families cannot share a parsed
/// context: their Python code canonicalisers intentionally differ when the
/// raw source has a dotted alias or no valid exchange. Retain both contexts
/// when a kernel call mixes their specs.
#[derive(Default)]
struct KernelNeeds {
    optional: ContractNeeds,
    strict: ContractNeeds,
    // These families require typed contexts which preserve fields absent from
    // the P1/information context.
    rank_state: ContractNeeds,
    cross_asset: ContractNeeds,
}

fn add_columns(
    need: &mut SourceNeed,
    columns: &[&'static str],
    missing_fails: bool,
    exchange_required: bool,
) {
    need.columns.extend(columns.iter().copied());
    need.missing_fails |= missing_fails;
    need.exchange_required |= exchange_required;
}

fn parse_specs(specs_payload: &Bound<'_, PyAny>) -> PyResult<Vec<KernelSpec>> {
    let specs = specs_payload.downcast::<PyList>()?;
    let mut output_cols = HashSet::new();
    let mut out = Vec::with_capacity(specs.len());
    for raw in specs.iter() {
        let dict = raw.downcast::<PyDict>()?;
        let factor = dict
            .get_item("factor")?
            .ok_or_else(|| PyErr::new::<PyKeyError, _>("typed_factor kernel spec missing factor"))?
            .extract::<String>()?;
        let output_col = match dict.get_item("output_col")? {
            Some(value) if !value.is_none() => value.extract::<String>()?,
            _ => dict
                .get_item("name")?
                .ok_or_else(|| {
                    PyErr::new::<PyKeyError, _>("typed_factor kernel spec missing name")
                })?
                .extract::<String>()?,
        };
        let params_value = dict.get_item("params")?.ok_or_else(|| {
            PyErr::new::<PyKeyError, _>("typed_factor kernel spec missing params")
        })?;
        let params = params_value.downcast::<PyDict>()?;
        let signal = params
            .get_item("signal")?
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>("typed_factor kernel spec requires params.signal")
            })?
            .extract::<String>()?
            .trim()
            .to_string();
        if signal.is_empty() {
            return Err(PyErr::new::<PyValueError, _>(
                "typed_factor kernel spec requires non-empty params.signal",
            ));
        }
        if !output_cols.insert(output_col.clone()) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor kernel has duplicate output_col: {output_col}"
            )));
        }
        out.push(KernelSpec {
            factor,
            signal,
            output_col,
        });
    }
    Ok(out)
}

/// Return whether a concrete `(factor, signal)` pair is implemented by the
/// typed Rust kernel set.
///
/// This deliberately operates at the instance level.  Several research
/// factor classes multiplex many `params.signal` variants, so a class-level
/// `factor` allowlist would incorrectly advertise unported variants as Rust
/// capable.
pub fn is_supported_typed_spec(factor: &str, signal: Option<&str>) -> bool {
    let Some(signal) = signal.map(str::trim).filter(|value| !value.is_empty()) else {
        return false;
    };
    supported_typed_factor_pairs()
        .iter()
        .any(|(known_factor, known_signal)| *known_factor == factor && *known_signal == signal)
}

/// Typed factor/signal pairs advertised by the generic capability surface.
pub fn supported_typed_factor_pairs() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "factor_mining_daily_catalog_v1",
            "base_debt_premium_floor_gap",
        ),
        (
            "factor_mining_daily_bond_stock_return_flow_information_v1",
            "bsfst_stock_return_bond_flow_mutual_information60",
        ),
        (
            "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1",
            "bssrc_bond_stock_rank_correlation60",
        ),
        (
            "factor_mining_daily_contract_stock_v1",
            "bstk_tail_cocrash_residual20",
        ),
        (
            "factor_mining_daily_expansion_v1",
            "dliq_volume_return_corr20",
        ),
        (
            "factor_mining_daily_ohlc_wick_path_asymmetry_v1",
            "dohw_intraday_sign_range_asymmetry60",
        ),
        (
            "factor_mining_daily_ohlc_wick_path_asymmetry_v1",
            "dohw_mean_wick_asymmetry60",
        ),
        (
            "factor_mining_daily_expansion_v1",
            "dredemption_bondpremium_interaction",
        ),
        (
            "factor_mining_daily_expansion_v1",
            "dredemption_premium_z20",
        ),
        (
            "factor_mining_daily_catalog_v1",
            "dret_drawup_drawdown_asym",
        ),
        ("factor_mining_daily_expansion_v1", "dret_volatility_20"),
        (
            "factor_mining_daily_incremental_v1",
            "drt_rebound_from_low20",
        ),
        ("factor_mining_daily_expansion_v1", "dtwap_morning_slope20"),
        (
            "factor_mining_daily_liquidity_channel_composition_v1",
            "lcc_amount_trade_size_information60",
        ),
        (
            "factor_mining_daily_liquidity_channel_composition_v1",
            "lcc_volume_deal_information60",
        ),
        (
            "factor_mining_orderbook_repricing_v1",
            "lrd_cross_side_reprice_symmetry",
        ),
        (
            "factor_mining_daily_capacity_rank_coupling_v1",
            "prcn_return_capacity_rank_corr60",
        ),
        (
            "factor_mining_quote_execution_dynamics_v1",
            "qed_prior_quote_lag2_agreement",
        ),
        (
            "factor_mining_quote_execution_dynamics_v1",
            "qed_prior_quote_location_dispersion",
        ),
        (
            "factor_mining_quote_execution_dynamics_v1",
            "qed_prior_quote_tail_penetration",
        ),
        (
            "factor_mining_daily_return_liquidity_topology_v1",
            "rjst_amount_joint_transition_entropy60",
        ),
        (
            "factor_mining_daily_return_liquidity_topology_v1",
            "rlmi_return_deal_sign_mutual_information60",
        ),
        (
            "factor_mining_daily_asymmetric_state_transitions_v1",
            "ydpt_yield_fall_return_beta60",
        ),
    ]
}

fn is_qed_spec(spec: &KernelSpec) -> bool {
    spec.factor == QED_FACTOR && QED_SIGNALS.contains(&spec.signal.as_str())
}

fn is_lrd_spec(spec: &KernelSpec) -> bool {
    spec.factor == LRD_FACTOR && spec.signal == LRD_SIGNAL
}

fn is_intraday_spec(spec: &KernelSpec) -> bool {
    is_qed_spec(spec) || is_lrd_spec(spec)
}

fn source_needs(specs: &[KernelSpec]) -> PyResult<KernelNeeds> {
    let mut needs = KernelNeeds::default();
    for spec in specs {
        match (spec.factor.as_str(), spec.signal.as_str()) {
            (QED_FACTOR, signal) if QED_SIGNALS.contains(&signal) => {
                // QED consumes only the physical score-day panel.  Missing
                // fields are deliberately handled as an all-NaN output, not
                // as a daily-source or parser exception.
            }
            (LRD_FACTOR, LRD_SIGNAL) => {
                // LRD likewise has no daily source.  Its strict panel schema
                // is validated by the typed intraday parser below.
            }
            ("factor_mining_daily_catalog_v1", "base_debt_premium_floor_gap")
            | ("factor_mining_daily_catalog_v1", "dret_drawup_drawdown_asym") => {
                add_columns(&mut needs.optional.base, CATALOG_BASE_COLUMNS, false, false);
                add_columns(
                    &mut needs.optional.price,
                    CATALOG_PRICE_COLUMNS,
                    false,
                    false,
                );
                add_columns(&mut needs.optional.twap, CATALOG_TWAP_COLUMNS, false, false);
            }
            ("factor_mining_daily_expansion_v1", "dredemption_bondpremium_interaction")
            | ("factor_mining_daily_expansion_v1", "dredemption_premium_z20") => {
                add_columns(
                    &mut needs.optional.base,
                    &[
                        "bond_prem_ratio",
                        "ytm",
                        "redemption_prem_ratio",
                        "pure_redemption_value",
                    ],
                    true,
                    true,
                );
            }
            ("factor_mining_daily_expansion_v1", "dret_volatility_20") => {
                add_columns(
                    &mut needs.optional.price,
                    &[
                        "prev_close_price",
                        "open_price",
                        "high_price",
                        "low_price",
                        "close_price",
                    ],
                    true,
                    true,
                );
            }
            ("factor_mining_daily_expansion_v1", "dliq_volume_return_corr20") => {
                add_columns(
                    &mut needs.optional.price,
                    &[
                        "prev_close_price",
                        "close_price",
                        "volume",
                        "amount",
                        "deal",
                    ],
                    true,
                    true,
                );
            }
            ("factor_mining_daily_expansion_v1", "dtwap_morning_slope20") => {
                add_columns(&mut needs.optional.twap, EXPANSION_TWAP_COLUMNS, true, true);
            }
            ("factor_mining_daily_incremental_v1", "drt_rebound_from_low20") => {
                add_columns(
                    &mut needs.optional.price,
                    &["close_price", "high_price", "low_price", "amount"],
                    true,
                    true,
                );
            }
            ("factor_mining_daily_ohlc_wick_path_asymmetry_v1", "dohw_mean_wick_asymmetry60")
            | (
                "factor_mining_daily_ohlc_wick_path_asymmetry_v1",
                "dohw_intraday_sign_range_asymmetry60",
            ) => {
                add_columns(
                    &mut needs.strict.price,
                    &[
                        "prev_close_price",
                        "open_price",
                        "high_price",
                        "low_price",
                        "close_price",
                    ],
                    true,
                    true,
                );
            }
            ("factor_mining_daily_contract_stock_v1", "bstk_tail_cocrash_residual20") => {
                add_columns(&mut needs.strict.price, &["close_price"], true, true);
                add_columns(
                    &mut needs.strict.base,
                    &[
                        "cb_prev_close_price",
                        "cb_close_price",
                        "stk_prev_close_price",
                        "stk_close_price",
                    ],
                    true,
                    true,
                );
            }
            (
                "factor_mining_daily_liquidity_channel_composition_v1",
                "lcc_amount_trade_size_information60",
            )
            | (
                "factor_mining_daily_liquidity_channel_composition_v1",
                "lcc_volume_deal_information60",
            ) => {
                add_columns(
                    &mut needs.strict.price,
                    INFORMATION_LCC_PRICE_COLUMNS,
                    true,
                    true,
                );
            }
            (
                "factor_mining_daily_return_liquidity_topology_v1",
                "rlmi_return_deal_sign_mutual_information60",
            )
            | (
                "factor_mining_daily_return_liquidity_topology_v1",
                "rjst_amount_joint_transition_entropy60",
            ) => {
                add_columns(
                    &mut needs.strict.price,
                    INFORMATION_TOPOLOGY_PRICE_COLUMNS,
                    true,
                    true,
                );
            }
            (
                "factor_mining_daily_capacity_rank_coupling_v1",
                "prcn_return_capacity_rank_corr60",
            ) => {
                add_columns(
                    &mut needs.rank_state.price,
                    RANK_STATE_PRCN_PRICE_COLUMNS,
                    true,
                    true,
                );
                add_columns(
                    &mut needs.rank_state.base,
                    RANK_STATE_PRCN_BASE_COLUMNS,
                    true,
                    true,
                );
            }
            (
                "factor_mining_daily_asymmetric_state_transitions_v1",
                "ydpt_yield_fall_return_beta60",
            ) => {
                add_columns(
                    &mut needs.rank_state.price,
                    RANK_STATE_YDPT_PRICE_COLUMNS,
                    true,
                    true,
                );
                add_columns(
                    &mut needs.rank_state.base,
                    RANK_STATE_YDPT_BASE_COLUMNS,
                    true,
                    true,
                );
            }
            (
                "factor_mining_daily_bond_stock_return_flow_information_v1",
                "bsfst_stock_return_bond_flow_mutual_information60",
            ) => {
                add_columns(
                    &mut needs.cross_asset.price,
                    CROSS_BSFST_PRICE_COLUMNS,
                    true,
                    true,
                );
                add_columns(
                    &mut needs.cross_asset.base,
                    CROSS_BSFST_BASE_COLUMNS,
                    true,
                    true,
                );
            }
            (
                "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1",
                "bssrc_bond_stock_rank_correlation60",
            ) => {
                add_columns(
                    &mut needs.cross_asset.price,
                    CROSS_BSSRC_PRICE_COLUMNS,
                    true,
                    true,
                );
                add_columns(
                    &mut needs.cross_asset.base,
                    CROSS_BSSRC_BASE_COLUMNS,
                    true,
                    true,
                );
            }
            _ => {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "typed_factor daily kernel kernel is not implemented for factor={} signal={}",
                    spec.factor, spec.signal
                )));
            }
        }
    }
    Ok(needs)
}

fn canonical_contract_for_factor(factor: &str) -> CanonicalContract {
    // P1 catalog, expansion, and incremental each use the same optional
    // exchange attachment rule in Python. The information subset uses the
    // strict parser shared by the topology and LCC reference kernels.
    match factor {
        "factor_mining_daily_catalog_v1"
        | "factor_mining_daily_expansion_v1"
        | "factor_mining_daily_incremental_v1" => CanonicalContract::OptionalExchange,
        "factor_mining_daily_liquidity_channel_composition_v1"
        | "factor_mining_daily_return_liquidity_topology_v1" => CanonicalContract::StrictExchange,
        _ => CanonicalContract::StrictExchange,
    }
}

fn normalized_date_strings(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    column: &str,
) -> PyResult<Vec<String>> {
    let source = df.call_method1("__getitem__", (column,))?;
    let pandas = py.import_bound("pandas")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("errors", "coerce")?;
    let parsed = pandas.call_method("to_datetime", (source,), Some(&kwargs))?;
    let normalized = parsed.getattr("dt")?.call_method0("normalize")?;
    let text = normalized.call_method1("astype", ("str",))?;
    text.call_method0("tolist")?.extract::<Vec<String>>()
}

fn parse_date(value: &str) -> Option<NaiveDate> {
    let trimmed = value.trim();
    if trimmed.len() < 10 {
        return None;
    }
    NaiveDate::parse_from_str(&trimmed[..10], "%Y-%m-%d").ok()
}

fn canonical_exchange(value: &str) -> String {
    match value.trim().to_ascii_uppercase().as_str() {
        "XSHG" | "SHSE" => "SH".to_string(),
        "XSHE" | "SZSE" => "SZ".to_string(),
        "BSE" | "BJSE" => "BJ".to_string(),
        other => other.to_string(),
    }
}

/// Canonicalise a daily source code under the exact Python family contract.
///
/// `None` preserves a pandas missing value. This matters because pandas 3's
/// `Series.astype(str).tolist()` returns a float `nan` for a missing `str`
/// value; passing that through `extract::<String>()` both raises in PyO3 and
/// loses the Python loader's missing-key behaviour.
fn canonical_market_code(
    value: Option<&str>,
    exchange: Option<&str>,
    contract: CanonicalContract,
) -> Option<String> {
    let mut code = value?.trim().to_ascii_uppercase();
    if code.ends_with(".0") {
        code.truncate(code.len().saturating_sub(2));
    }
    if contract == CanonicalContract::StrictExchange {
        // Strict Python kernels first resolve a dotted suffix. Hence `.XSHG`
        // and `.SHSE` are valid self-contained Shanghai codes even if the
        // daily exchange field is missing or invalid.
        if code.is_empty() || code == "NAN" {
            return Some(String::new());
        }
        if let Some((bare, suffix)) = code.rsplit_once('.') {
            let suffix = canonical_exchange(suffix);
            if !bare.is_empty() && matches!(suffix.as_str(), "SH" | "SZ" | "BJ") {
                return Some(format!("{bare}.{suffix}"));
            }
        }
        let suffix = canonical_exchange(exchange.unwrap_or(""));
        return if matches!(suffix.as_str(), "SH" | "SZ" | "BJ") {
            Some(format!("{code}.{suffix}"))
        } else {
            Some(String::new())
        };
    }

    // Catalog, expansion, and incremental reproduce
    // `values.astype(str).str.strip().str.upper().str.replace(r"\\.0$", "")`
    // plus optional exchange attachment. A dotted alias is intentionally not
    // normalised: Python only recognises `.SH/.SZ/.BJ` before deciding whether
    // to attach the separate exchange.
    let has_suffix = code.ends_with(".SH") || code.ends_with(".SZ") || code.ends_with(".BJ");
    let exchange = canonical_exchange(exchange.unwrap_or(""));
    if !has_suffix && matches!(exchange.as_str(), "SH" | "SZ" | "BJ") {
        code.push('.');
        code.push_str(&exchange);
    }
    Some(code)
}

fn require_columns(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    source: &str,
    need: &SourceNeed,
) -> PyResult<()> {
    let mut missing = Vec::new();
    for column in ["trade_date", "code"] {
        if !super::has_col(py, df, column)? {
            missing.push(column.to_string());
        }
    }
    if need.exchange_required && !super::has_col(py, df, "exchange_code")? {
        missing.push("exchange_code".to_string());
    }
    for column in &need.columns {
        if !super::has_col(py, df, column)? {
            missing.push((*column).to_string());
        }
    }
    if !missing.is_empty() {
        return Err(PyErr::new::<PyKeyError, _>(format!(
            "typed_factor kernel {source} missing required columns: {missing:?}"
        )));
    }
    Ok(())
}

fn source_frame<'py>(
    daily_data: Option<&Bound<'py, PyAny>>,
    source: &str,
    missing_fails: bool,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(raw_daily) = daily_data else {
        return if missing_fails {
            Err(PyErr::new::<PyKeyError, _>(format!(
                "typed_factor kernel missing daily source: {source}"
            )))
        } else {
            Ok(None)
        };
    };
    let dict = raw_daily.downcast::<PyDict>()?;
    let Some(frame) = dict.get_item(source)? else {
        return if missing_fails {
            Err(PyErr::new::<PyKeyError, _>(format!(
                "typed_factor kernel missing daily source: {source}"
            )))
        } else {
            Ok(None)
        };
    };
    if frame.is_none() {
        return if missing_fails {
            Err(PyErr::new::<PyKeyError, _>(format!(
                "typed_factor kernel missing daily source: {source}"
            )))
        } else {
            Ok(None)
        };
    }
    Ok(Some(frame))
}

fn is_empty(df: &Bound<'_, PyAny>) -> PyResult<bool> {
    df.getattr("empty")?.extract::<bool>()
}

fn numeric_or_nan(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    column: &str,
    length: usize,
) -> PyResult<Vec<f64>> {
    if super::has_col(py, df, column)? {
        super::col_to_f64_vec(py, df, column)
    } else {
        Ok(vec![f64::NAN; length])
    }
}

fn nullable_string_values_or_missing(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    column: &str,
    length: usize,
) -> PyResult<Vec<Option<String>>> {
    if !super::has_col(py, df, column)? {
        return Ok(vec![None; length]);
    }
    let source = df.call_method1("__getitem__", (column,))?;
    let text = source.call_method1("astype", ("str",))?;
    let raw_values = text.call_method0("tolist")?;
    let values = raw_values.downcast::<PyList>()?;
    let missing = text.call_method0("isna")?.call_method0("tolist")?;
    let missing = missing.extract::<Vec<bool>>()?;
    if values.len() != missing.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "typed_factor kernel {column} missing-mask length mismatch"
        )));
    }
    values
        .iter()
        .zip(missing)
        .map(|(value, is_missing)| {
            if is_missing {
                Ok(None)
            } else {
                value.extract::<String>().map(Some)
            }
        })
        .collect()
}

fn sort_daily_context(ctx: &mut TypedFactorDailyContext) {
    for rows in ctx.base_by_code.values_mut() {
        rows.sort_by_key(|row| row.trade_date);
    }
    for rows in ctx.price_by_code.values_mut() {
        rows.sort_by_key(|row| row.trade_date);
    }
    for rows in ctx.twap_by_code.values_mut() {
        rows.sort_by_key(|row| row.trade_date);
    }
}

fn populate_daily_context(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    needs: &ContractNeeds,
    canonical_contract: CanonicalContract,
    score_date: NaiveDate,
    ctx: &mut TypedFactorDailyContext,
) -> PyResult<()> {
    parse_base_source(
        py,
        daily_data,
        &needs.base,
        canonical_contract,
        score_date,
        ctx,
    )?;
    parse_price_source(
        py,
        daily_data,
        &needs.price,
        canonical_contract,
        score_date,
        ctx,
    )?;
    parse_twap_source(
        py,
        daily_data,
        &needs.twap,
        canonical_contract,
        score_date,
        ctx,
    )?;
    sort_daily_context(ctx);
    Ok(())
}

/// Build a separate typed history for the path kernels.  It deliberately does
/// not reuse `TypedFactorDailyContext`: the legacy P1/info rows omit the adjusted
/// prior close and open fields required by the four path contracts.
fn populate_daily_path_context(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    need: &SourceNeed,
    canonical_contract: CanonicalContract,
    score_date: NaiveDate,
    ctx: &mut TypedFactorDailyPathContext,
) -> PyResult<()> {
    if need.columns.is_empty() {
        return Ok(());
    }
    let Some(df) = source_frame(daily_data, SOURCE_PRICE, need.missing_fails)? else {
        return Ok(());
    };
    if is_empty(&df)? && !need.missing_fails {
        return Ok(());
    }
    require_columns(py, &df, SOURCE_PRICE, need)?;
    if is_empty(&df)? {
        return Ok(());
    }

    let dates = normalized_date_strings(py, &df, "trade_date")?;
    let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
    let n = dates.len();
    let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
    let previous = numeric_or_nan(py, &df, "prev_close_price", n)?;
    let adjusted_previous = numeric_or_nan(py, &df, "act_prev_close_price", n)?;
    let opened = numeric_or_nan(py, &df, "open_price", n)?;
    let high = numeric_or_nan(py, &df, "high_price", n)?;
    let low = numeric_or_nan(py, &df, "low_price", n)?;
    let close = numeric_or_nan(py, &df, "close_price", n)?;
    if [
        codes.len(),
        exchanges.len(),
        previous.len(),
        adjusted_previous.len(),
        opened.len(),
        high.len(),
        low.len(),
        close.len(),
    ]
    .iter()
    .any(|length| *length != n)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor kernel daily-path price length mismatch",
        ));
    }
    let mut seen = HashSet::new();
    for i in 0..n {
        let Some(trade_date) = parse_date(&dates[i]) else {
            continue;
        };
        if trade_date >= score_date {
            continue;
        }
        let Some(code) = canonical_market_code(
            codes[i].as_deref(),
            exchanges[i].as_deref(),
            canonical_contract,
        ) else {
            continue;
        };
        if code.is_empty() {
            continue;
        }
        if !seen.insert((trade_date, code.clone())) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor kernel {SOURCE_PRICE} has duplicate strict-prior rows for {code} on {trade_date}"
            )));
        }
        ctx.price_by_code
            .entry(code.clone())
            .or_default()
            .push(TypedFactorDailyPathPriceRow {
                trade_date,
                code,
                prev_close_price: previous[i],
                act_prev_close_price: adjusted_previous[i],
                open_price: opened[i],
                high_price: high[i],
                low_price: low[i],
                close_price: close[i],
            });
    }
    for rows in ctx.price_by_code.values_mut() {
        rows.sort_by_key(|row| row.trade_date);
    }
    Ok(())
}

/// Keep the tracking factor's independently sourced market-price calendar
/// separate from its historical bond/stock base data.  The Python kernel uses
/// the former only to establish session adjacency and its strict anchor.
fn populate_daily_tracking_context(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    needs: &ContractNeeds,
    score_date: NaiveDate,
    ctx: &mut TypedFactorDailyTrackingContext,
) -> PyResult<()> {
    if !needs.price.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_PRICE, needs.price.missing_fails)? else {
            return Ok(());
        };
        if !(is_empty(&df)? && !needs.price.missing_fails) {
            require_columns(py, &df, SOURCE_PRICE, &needs.price)?;
            if !is_empty(&df)? {
                let dates = normalized_date_strings(py, &df, "trade_date")?;
                let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
                let n = dates.len();
                let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
                if codes.len() != n || exchanges.len() != n {
                    return Err(PyErr::new::<PyValueError, _>(
                        "typed_factor kernel daily tracking price length mismatch",
                    ));
                }
                let mut seen = HashSet::new();
                for i in 0..n {
                    let Some(trade_date) = parse_date(&dates[i]) else {
                        continue;
                    };
                    if trade_date >= score_date {
                        continue;
                    }
                    let Some(code) = canonical_market_code(
                        codes[i].as_deref(),
                        exchanges[i].as_deref(),
                        CanonicalContract::StrictExchange,
                    ) else {
                        continue;
                    };
                    if code.is_empty() {
                        continue;
                    }
                    if !seen.insert((trade_date, code.clone())) {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "typed_factor kernel {SOURCE_PRICE} has duplicate strict-prior rows for {code} on {trade_date}"
                        )));
                    }
                    ctx.price_anchor_by_code
                        .entry(code.clone())
                        .or_default()
                        .push(TypedFactorDailyPriceAnchorRow { trade_date, code });
                }
            }
        }
    }

    if !needs.base.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_BASE, needs.base.missing_fails)? else {
            return Ok(());
        };
        if !(is_empty(&df)? && !needs.base.missing_fails) {
            require_columns(py, &df, SOURCE_BASE, &needs.base)?;
            if !is_empty(&df)? {
                let dates = normalized_date_strings(py, &df, "trade_date")?;
                let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
                let n = dates.len();
                let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
                let cb_previous = numeric_or_nan(py, &df, "cb_prev_close_price", n)?;
                let cb_close = numeric_or_nan(py, &df, "cb_close_price", n)?;
                let stock_previous = numeric_or_nan(py, &df, "stk_prev_close_price", n)?;
                let stock_close = numeric_or_nan(py, &df, "stk_close_price", n)?;
                if [
                    codes.len(),
                    exchanges.len(),
                    cb_previous.len(),
                    cb_close.len(),
                    stock_previous.len(),
                    stock_close.len(),
                ]
                .iter()
                .any(|length| *length != n)
                {
                    return Err(PyErr::new::<PyValueError, _>(
                        "typed_factor kernel daily tracking base length mismatch",
                    ));
                }
                let mut seen = HashSet::new();
                for i in 0..n {
                    let Some(trade_date) = parse_date(&dates[i]) else {
                        continue;
                    };
                    if trade_date >= score_date {
                        continue;
                    }
                    let Some(code) = canonical_market_code(
                        codes[i].as_deref(),
                        exchanges[i].as_deref(),
                        CanonicalContract::StrictExchange,
                    ) else {
                        continue;
                    };
                    if code.is_empty() {
                        continue;
                    }
                    if !seen.insert((trade_date, code.clone())) {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "typed_factor kernel {SOURCE_BASE} has duplicate strict-prior rows for {code} on {trade_date}"
                        )));
                    }
                    ctx.base_by_code.entry(code.clone()).or_default().push(
                        TypedFactorDailyTrackingBaseRow {
                            trade_date,
                            code,
                            cb_prev_close_price: cb_previous[i],
                            cb_close_price: cb_close[i],
                            stk_prev_close_price: stock_previous[i],
                            stk_close_price: stock_close[i],
                        },
                    );
                }
            }
        }
    }
    for rows in ctx.price_anchor_by_code.values_mut() {
        rows.sort_by_key(|row| row.trade_date);
    }
    for rows in ctx.base_by_code.values_mut() {
        rows.sort_by_key(|row| row.trade_date);
    }
    Ok(())
}

/// Build the strict typed rows for PRCN and YDPT without reusing the legacy
/// P1 context.  Source canonicalisation and duplicate detection happen before
/// any output-code selection, exactly as in the two Python families.
fn populate_rank_state_context(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    needs: &ContractNeeds,
    score_date: NaiveDate,
    ctx: &mut TypedFactorRankStateContext,
) -> PyResult<()> {
    if !needs.price.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_PRICE, needs.price.missing_fails)? else {
            return Ok(());
        };
        if !(is_empty(&df)? && !needs.price.missing_fails) {
            require_columns(py, &df, SOURCE_PRICE, &needs.price)?;
            if !is_empty(&df)? {
                let dates = normalized_date_strings(py, &df, "trade_date")?;
                let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
                let n = dates.len();
                let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
                let previous = numeric_or_nan(py, &df, "prev_close_price", n)?;
                let adjusted_previous = numeric_or_nan(py, &df, "act_prev_close_price", n)?;
                let close = numeric_or_nan(py, &df, "close_price", n)?;
                let amount = numeric_or_nan(py, &df, "amount", n)?;
                if [
                    codes.len(),
                    exchanges.len(),
                    previous.len(),
                    adjusted_previous.len(),
                    close.len(),
                    amount.len(),
                ]
                .iter()
                .any(|length| *length != n)
                {
                    return Err(PyErr::new::<PyValueError, _>(
                        "typed_factor kernel rank/state daily_price length mismatch",
                    ));
                }
                let mut seen = HashSet::new();
                for i in 0..n {
                    let Some(trade_date) = parse_date(&dates[i]) else {
                        continue;
                    };
                    if trade_date >= score_date {
                        continue;
                    }
                    let code = canonical_market_code(
                        codes[i].as_deref(),
                        exchanges[i].as_deref(),
                        CanonicalContract::StrictExchange,
                    )
                    .unwrap_or_default();
                    if code.is_empty() {
                        continue;
                    }
                    if !seen.insert((trade_date, code.clone())) {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "typed_factor kernel {SOURCE_PRICE} has duplicate strict-prior rows for {code} on {trade_date}"
                        )));
                    }
                    ctx.price_rows.push(TypedFactorRankStatePriceRow {
                        trade_date,
                        code,
                        prev_close_price: previous[i],
                        act_prev_close_price: adjusted_previous[i],
                        close_price: close[i],
                        amount: amount[i],
                    });
                }
            }
        }
    }

    if !needs.base.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_BASE, needs.base.missing_fails)? else {
            return Ok(());
        };
        if !(is_empty(&df)? && !needs.base.missing_fails) {
            require_columns(py, &df, SOURCE_BASE, &needs.base)?;
            if !is_empty(&df)? {
                let dates = normalized_date_strings(py, &df, "trade_date")?;
                let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
                let n = dates.len();
                let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
                let remain_size = numeric_or_nan(py, &df, "remain_size", n)?;
                let current_yield = numeric_or_nan(py, &df, "current_yield", n)?;
                if [
                    codes.len(),
                    exchanges.len(),
                    remain_size.len(),
                    current_yield.len(),
                ]
                .iter()
                .any(|length| *length != n)
                {
                    return Err(PyErr::new::<PyValueError, _>(
                        "typed_factor kernel rank/state daily_base length mismatch",
                    ));
                }
                let mut seen = HashSet::new();
                for i in 0..n {
                    let Some(trade_date) = parse_date(&dates[i]) else {
                        continue;
                    };
                    if trade_date >= score_date {
                        continue;
                    }
                    let code = canonical_market_code(
                        codes[i].as_deref(),
                        exchanges[i].as_deref(),
                        CanonicalContract::StrictExchange,
                    )
                    .unwrap_or_default();
                    if code.is_empty() {
                        continue;
                    }
                    if !seen.insert((trade_date, code.clone())) {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "typed_factor kernel {SOURCE_BASE} has duplicate strict-prior rows for {code} on {trade_date}"
                        )));
                    }
                    ctx.base_rows.push(TypedFactorRankStateBaseRow {
                        trade_date,
                        code,
                        remain_size: remain_size[i],
                        current_yield: current_yield[i],
                    });
                }
            }
        }
    }
    Ok(())
}

/// Build the independent raw daily context for the two cross-asset families.
/// The typed module receives already strict-canonical source codes; `stock_code`
/// remains empty for bsFST-only calls, preserving that factor's smaller schema.
fn populate_cross_asset_context(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    needs: &ContractNeeds,
    score_date: NaiveDate,
    ctx: &mut TypedFactorDailyCrossAssetContext,
) -> PyResult<()> {
    if !needs.price.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_PRICE, needs.price.missing_fails)? else {
            return Ok(());
        };
        if !(is_empty(&df)? && !needs.price.missing_fails) {
            require_columns(py, &df, SOURCE_PRICE, &needs.price)?;
            if !is_empty(&df)? {
                let dates = normalized_date_strings(py, &df, "trade_date")?;
                let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
                let n = dates.len();
                let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
                let previous = numeric_or_nan(py, &df, "prev_close_price", n)?;
                let close = numeric_or_nan(py, &df, "close_price", n)?;
                let amount = numeric_or_nan(py, &df, "amount", n)?;
                if [
                    codes.len(),
                    exchanges.len(),
                    previous.len(),
                    close.len(),
                    amount.len(),
                ]
                .iter()
                .any(|length| *length != n)
                {
                    return Err(PyErr::new::<PyValueError, _>(
                        "typed_factor kernel cross-asset daily_price length mismatch",
                    ));
                }
                let mut seen = HashSet::new();
                for i in 0..n {
                    let Some(trade_date) = parse_date(&dates[i]) else {
                        continue;
                    };
                    if trade_date >= score_date {
                        continue;
                    }
                    let code = canonical_market_code(
                        codes[i].as_deref(),
                        exchanges[i].as_deref(),
                        CanonicalContract::StrictExchange,
                    )
                    .unwrap_or_default();
                    if code.is_empty() {
                        continue;
                    }
                    if !seen.insert((trade_date, code.clone())) {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "typed_factor kernel {SOURCE_PRICE} has duplicate strict-prior rows for {code} on {trade_date}"
                        )));
                    }
                    ctx.price_rows.push(TypedFactorCrossAssetPriceRow {
                        trade_date,
                        code,
                        exchange_code: String::new(),
                        prev_close_price: previous[i],
                        close_price: close[i],
                        amount: amount[i],
                    });
                }
            }
        }
    }

    if !needs.base.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_BASE, needs.base.missing_fails)? else {
            return Ok(());
        };
        if !(is_empty(&df)? && !needs.base.missing_fails) {
            require_columns(py, &df, SOURCE_BASE, &needs.base)?;
            if !is_empty(&df)? {
                let dates = normalized_date_strings(py, &df, "trade_date")?;
                let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
                let n = dates.len();
                let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
                let stock_codes = nullable_string_values_or_missing(py, &df, "stock_code", n)?;
                let stock_previous = numeric_or_nan(py, &df, "stk_prev_close_price", n)?;
                let stock_close = numeric_or_nan(py, &df, "stk_close_price", n)?;
                if [
                    codes.len(),
                    exchanges.len(),
                    stock_codes.len(),
                    stock_previous.len(),
                    stock_close.len(),
                ]
                .iter()
                .any(|length| *length != n)
                {
                    return Err(PyErr::new::<PyValueError, _>(
                        "typed_factor kernel cross-asset daily_base length mismatch",
                    ));
                }
                let mut seen = HashSet::new();
                for i in 0..n {
                    let Some(trade_date) = parse_date(&dates[i]) else {
                        continue;
                    };
                    if trade_date >= score_date {
                        continue;
                    }
                    let code = canonical_market_code(
                        codes[i].as_deref(),
                        exchanges[i].as_deref(),
                        CanonicalContract::StrictExchange,
                    )
                    .unwrap_or_default();
                    if code.is_empty() {
                        continue;
                    }
                    if !seen.insert((trade_date, code.clone())) {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "typed_factor kernel {SOURCE_BASE} has duplicate strict-prior rows for {code} on {trade_date}"
                        )));
                    }
                    let stock_code = canonical_market_code(
                        stock_codes[i].as_deref(),
                        exchanges[i].as_deref(),
                        CanonicalContract::StrictExchange,
                    )
                    .unwrap_or_default();
                    ctx.base_rows.push(TypedFactorCrossAssetBaseRow {
                        trade_date,
                        code,
                        exchange_code: String::new(),
                        stock_code,
                        stk_prev_close_price: stock_previous[i],
                        stk_close_price: stock_close[i],
                    });
                }
            }
        }
    }
    Ok(())
}

fn parse_base_source(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    need: &SourceNeed,
    canonical_contract: CanonicalContract,
    score_date: NaiveDate,
    ctx: &mut TypedFactorDailyContext,
) -> PyResult<()> {
    if need.columns.is_empty() {
        return Ok(());
    }
    let Some(df) = source_frame(daily_data, SOURCE_BASE, need.missing_fails)? else {
        return Ok(());
    };
    if is_empty(&df)? && !need.missing_fails {
        return Ok(());
    }
    require_columns(py, &df, SOURCE_BASE, need)?;
    if is_empty(&df)? {
        return Ok(());
    }
    let dates = normalized_date_strings(py, &df, "trade_date")?;
    let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
    let n = dates.len();
    let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
    let debt = numeric_or_nan(py, &df, "debt_puredebt_ratio", n)?;
    let pure = numeric_or_nan(py, &df, "puredebt_prem_ratio", n)?;
    let bond = numeric_or_nan(py, &df, "bond_prem_ratio", n)?;
    let redemption = numeric_or_nan(py, &df, "redemption_prem_ratio", n)?;
    if [
        codes.len(),
        exchanges.len(),
        debt.len(),
        pure.len(),
        bond.len(),
        redemption.len(),
    ]
    .iter()
    .any(|length| *length != n)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor kernel daily_base length mismatch",
        ));
    }
    let mut seen = HashSet::new();
    for i in 0..n {
        let Some(trade_date) = parse_date(&dates[i]) else {
            continue;
        };
        if trade_date >= score_date {
            continue;
        }
        let Some(code) = canonical_market_code(
            codes[i].as_deref(),
            exchanges[i].as_deref(),
            canonical_contract,
        ) else {
            continue;
        };
        if code.is_empty() {
            continue;
        }
        if !seen.insert((trade_date, code.clone())) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor kernel {SOURCE_BASE} has duplicate strict-prior rows for {code} on {trade_date}"
            )));
        }
        ctx.base_by_code
            .entry(code.clone())
            .or_default()
            .push(TypedFactorDailyBaseRow {
                trade_date,
                code,
                debt_puredebt_ratio: debt[i],
                puredebt_prem_ratio: pure[i],
                bond_prem_ratio: bond[i],
                redemption_prem_ratio: redemption[i],
            });
    }
    Ok(())
}

fn parse_price_source(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    need: &SourceNeed,
    canonical_contract: CanonicalContract,
    score_date: NaiveDate,
    ctx: &mut TypedFactorDailyContext,
) -> PyResult<()> {
    if need.columns.is_empty() {
        return Ok(());
    }
    let Some(df) = source_frame(daily_data, SOURCE_PRICE, need.missing_fails)? else {
        return Ok(());
    };
    if is_empty(&df)? && !need.missing_fails {
        return Ok(());
    }
    require_columns(py, &df, SOURCE_PRICE, need)?;
    if is_empty(&df)? {
        return Ok(());
    }
    let dates = normalized_date_strings(py, &df, "trade_date")?;
    let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
    let n = dates.len();
    let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
    let prev = numeric_or_nan(py, &df, "prev_close_price", n)?;
    let close = numeric_or_nan(py, &df, "close_price", n)?;
    let high = numeric_or_nan(py, &df, "high_price", n)?;
    let low = numeric_or_nan(py, &df, "low_price", n)?;
    let volume = numeric_or_nan(py, &df, "volume", n)?;
    let amount = numeric_or_nan(py, &df, "amount", n)?;
    let deal = numeric_or_nan(py, &df, "deal", n)?;
    if [
        codes.len(),
        exchanges.len(),
        prev.len(),
        close.len(),
        high.len(),
        low.len(),
        volume.len(),
        amount.len(),
        deal.len(),
    ]
    .iter()
    .any(|length| *length != n)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor kernel daily_price length mismatch",
        ));
    }
    let mut seen = HashSet::new();
    for i in 0..n {
        let Some(trade_date) = parse_date(&dates[i]) else {
            continue;
        };
        if trade_date >= score_date {
            continue;
        }
        let Some(code) = canonical_market_code(
            codes[i].as_deref(),
            exchanges[i].as_deref(),
            canonical_contract,
        ) else {
            continue;
        };
        if code.is_empty() {
            continue;
        }
        if !seen.insert((trade_date, code.clone())) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor kernel {SOURCE_PRICE} has duplicate strict-prior rows for {code} on {trade_date}"
            )));
        }
        ctx.price_by_code
            .entry(code.clone())
            .or_default()
            .push(TypedFactorDailyPriceRow {
                trade_date,
                code,
                prev_close_price: prev[i],
                close_price: close[i],
                high_price: high[i],
                low_price: low[i],
                volume: volume[i],
                amount: amount[i],
                deal: deal[i],
            });
    }
    Ok(())
}

fn parse_twap_source(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    need: &SourceNeed,
    canonical_contract: CanonicalContract,
    score_date: NaiveDate,
    ctx: &mut TypedFactorDailyContext,
) -> PyResult<()> {
    if need.columns.is_empty() {
        return Ok(());
    }
    let Some(df) = source_frame(daily_data, SOURCE_TWAP, need.missing_fails)? else {
        return Ok(());
    };
    if is_empty(&df)? && !need.missing_fails {
        return Ok(());
    }
    require_columns(py, &df, SOURCE_TWAP, need)?;
    if is_empty(&df)? {
        return Ok(());
    }
    let dates = normalized_date_strings(py, &df, "trade_date")?;
    let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
    let n = dates.len();
    let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
    let morning_open = numeric_or_nan(py, &df, "twap_0930_0935", n)?;
    let morning_end = numeric_or_nan(py, &df, "twap_0935_1000", n)?;
    if [
        codes.len(),
        exchanges.len(),
        morning_open.len(),
        morning_end.len(),
    ]
    .iter()
    .any(|length| *length != n)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor kernel daily_twap length mismatch",
        ));
    }
    let mut seen = HashSet::new();
    for i in 0..n {
        let Some(trade_date) = parse_date(&dates[i]) else {
            continue;
        };
        if trade_date >= score_date {
            continue;
        }
        let Some(code) = canonical_market_code(
            codes[i].as_deref(),
            exchanges[i].as_deref(),
            canonical_contract,
        ) else {
            continue;
        };
        if code.is_empty() {
            continue;
        }
        if !seen.insert((trade_date, code.clone())) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor kernel {SOURCE_TWAP} has duplicate strict-prior rows for {code} on {trade_date}"
            )));
        }
        ctx.twap_by_code
            .entry(code.clone())
            .or_default()
            .push(TypedFactorDailyTwapRow {
                trade_date,
                code,
                twap_0930_0935: morning_open[i],
                twap_0935_1000: morning_end[i],
            });
    }
    Ok(())
}

fn explicit_score_date(compute_params: Option<&Bound<'_, PyAny>>) -> PyResult<Option<NaiveDate>> {
    let Some(raw) = compute_params else {
        return Ok(None);
    };
    if raw.is_none() {
        return Ok(None);
    }
    let dict = raw.downcast::<PyDict>()?;
    let Some(value) = dict.get_item("__factor_score_date")? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let text = value.extract::<String>()?;
    parse_date(&text).map(Some).ok_or_else(|| {
        PyErr::new::<PyValueError, _>("typed factor kernel has invalid __factor_score_date")
    })
}

fn score_date_from_panel(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    compute_params: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<NaiveDate>> {
    if let Some(score_date) = explicit_score_date(compute_params)? {
        return Ok(Some(score_date));
    }
    if is_empty(panel_df)? {
        return Ok(None);
    }
    if !super::has_col(py, panel_df, "dt")?
        || !super::has_col(py, panel_df, "code")?
        || !super::has_col(py, panel_df, "seq")?
    {
        return Err(PyErr::new::<PyKeyError, _>(
            "typed_factor kernel panel requires dt, code, and seq",
        ));
    }
    let dates = normalized_date_strings(py, panel_df, "dt")?;
    let unique: BTreeSet<NaiveDate> = dates.iter().filter_map(|value| parse_date(value)).collect();
    if unique.len() != 1 {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor kernel requires one valid score date in panel_df",
        ));
    }
    Ok(unique.into_iter().next())
}

/// The legacy daily catalog is the only paths4 family whose output universe
/// is defined by physical score-day snapshots rather than the complete
/// labelled score-day panel.  Retain the original label timestamp in the key
/// because pandas' MultiIndex output does not normalise it to midnight.
fn catalog_output_keys(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    score_date: NaiveDate,
) -> PyResult<BTreeSet<(String, String)>> {
    if !super::has_col(py, panel_df, "trade_time")? {
        return Err(PyErr::new::<PyKeyError, _>(
            "typed_factor catalog kernel panel missing required trade_time",
        ));
    }
    let labelled_dates = normalized_date_strings(py, panel_df, "dt")?;
    let raw_dates = super::col_to_str_vec(py, panel_df, "dt")?;
    let physical_dates = normalized_date_strings(py, panel_df, "trade_time")?;
    let codes = super::col_to_str_vec(py, panel_df, "code")?;
    if labelled_dates.len() != raw_dates.len()
        || labelled_dates.len() != physical_dates.len()
        || labelled_dates.len() != codes.len()
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor catalog kernel panel length mismatch",
        ));
    }
    let mut keys_on_day = BTreeSet::new();
    for (((labelled, raw_date), physical), code) in labelled_dates
        .iter()
        .zip(raw_dates.iter())
        .zip(physical_dates.iter())
        .zip(codes.iter())
    {
        if parse_date(labelled) == Some(score_date) && parse_date(physical) == Some(score_date) {
            keys_on_day.insert((raw_date.clone(), code.clone()));
        }
    }
    Ok(keys_on_day)
}

fn output_keys(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    score_date: NaiveDate,
) -> PyResult<Vec<(String, String)>> {
    let dates = normalized_date_strings(py, panel_df, "dt")?;
    // `_output_index` in the Python kernels filters on a normalised score day
    // but retains the original `dt` index value (normally 14:30 for T1430).
    // Keep that label verbatim rather than replacing it with midnight.
    let raw_dates = super::col_to_str_vec(py, panel_df, "dt")?;
    let codes = super::col_to_str_vec(py, panel_df, "code")?;
    if dates.len() != raw_dates.len() || dates.len() != codes.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor kernel panel length mismatch",
        ));
    }
    let mut keys = BTreeSet::new();
    for ((date, raw_date), code) in dates.iter().zip(raw_dates.iter()).zip(codes.iter()) {
        if parse_date(date) == Some(score_date) {
            keys.insert((raw_date.clone(), code.clone()));
        }
    }
    Ok(keys.into_iter().collect())
}

/// QED's labelled output index follows pandas' direct Timestamp comparison,
/// not merely a calendar-date comparison.  The adapter materialises that
/// result when a tz-aware label/build-day pair would otherwise lose type
/// information in the Rust date parser.
fn intraday_labelled_output_keys(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    score_date: NaiveDate,
) -> PyResult<Vec<(String, String)>> {
    if !super::has_col(py, panel_df, "__label_matches_score_day__")? {
        return output_keys(py, panel_df, score_date);
    }
    let raw_dates = super::col_to_str_vec(py, panel_df, "dt")?;
    let codes = super::col_to_str_vec(py, panel_df, "code")?;
    let matches = super::col_to_i64_vec(py, panel_df, "__label_matches_score_day__")?;
    if raw_dates.len() != codes.len() || raw_dates.len() != matches.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor intraday kernel labelled key length mismatch",
        ));
    }
    let mut keys = BTreeSet::new();
    for index in 0..raw_dates.len() {
        if matches[index] != 0 {
            keys.insert((raw_dates[index].clone(), codes[index].clone()));
        }
    }
    Ok(keys.into_iter().collect())
}

#[derive(Clone, Copy)]
struct PhysicalIntradayRow {
    index: usize,
    clock_ns: i64,
}

fn require_panel_columns(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    factor: &str,
    columns: impl IntoIterator<Item = String>,
) -> PyResult<()> {
    let missing = columns
        .into_iter()
        .filter_map(|column| match super::has_col(py, panel_df, &column) {
            Ok(true) => None,
            Ok(false) => Some(Ok(column)),
            Err(error) => Some(Err(error)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    if missing.is_empty() {
        return Ok(());
    }
    Err(PyErr::new::<PyKeyError, _>(format!(
        "{factor} missing required panel column(s): {}",
        missing.join(", ")
    )))
}

fn lrd_required_panel_columns() -> Vec<String> {
    let mut columns = Vec::with_capacity(21);
    columns.push("trade_time".to_string());
    for level in 1..=5 {
        columns.push(format!("ask_price{level}"));
        columns.push(format!("bid_price{level}"));
        columns.push(format!("ask_volume{level}"));
        columns.push(format!("bid_volume{level}"));
    }
    columns
}

/// The QED and LRD Python references derive their score day from
/// `panel.attrs["__build_day__"]` before considering the labelled index.  Do
/// not reuse the daily kernel's backend override here: that override is a
/// daily compatibility convention and would hide an intraday provenance
/// conflict.
fn intraday_score_date_from_panel(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
) -> PyResult<Option<NaiveDate>> {
    let attrs_object = panel_df.getattr("attrs")?;
    let attrs = attrs_object.downcast::<PyDict>()?;
    if let Some(raw_day) = attrs.get_item("__build_day__")? {
        if !raw_day.is_none() {
            let pandas = py.import_bound("pandas")?;
            let timestamp = pandas.getattr("Timestamp")?.call1((raw_day,))?;
            if pandas
                .call_method1("isna", (timestamp.clone(),))?
                .extract::<bool>()?
            {
                return Err(PyErr::new::<PyValueError, _>(
                    "typed_factor intraday kernel has invalid panel __build_day__",
                ));
            }
            let text = timestamp
                .call_method0("normalize")?
                .call_method1("strftime", ("%Y-%m-%d",))?
                .extract::<String>()?;
            return parse_date(&text).map(Some).ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "typed_factor intraday kernel has invalid panel __build_day__",
                )
            });
        }
    }
    if is_empty(panel_df)? {
        return Ok(None);
    }
    if !super::has_col(py, panel_df, "dt")?
        || !super::has_col(py, panel_df, "code")?
        || !super::has_col(py, panel_df, "seq")?
    {
        return Err(PyErr::new::<PyKeyError, _>(
            "typed_factor intraday kernel panel requires dt, code, and seq",
        ));
    }
    let dates = normalized_date_strings(py, panel_df, "dt")?;
    let unique: BTreeSet<NaiveDate> = dates.iter().filter_map(|value| parse_date(value)).collect();
    if unique.len() != 1 {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor intraday kernel requires panel __build_day__ for a multi-date panel",
        ));
    }
    Ok(unique.into_iter().next())
}

/// Recover the Python-visible local timestamp date and full nanosecond clock.
/// `_prepare_panel` produces the two helper columns before the extension sees
/// the frame, retaining wall-clock semantics for tz-aware inputs.  The fallback
/// keeps direct scratch calls usable for ordinary timezone-naive panels.
fn intraday_time_parts(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    length: usize,
) -> PyResult<(Vec<String>, Vec<i64>, Option<Vec<bool>>)> {
    if !super::has_col(py, panel_df, "trade_time")? {
        return Err(PyErr::new::<PyKeyError, _>(
            "typed_factor intraday kernel panel missing required trade_time",
        ));
    }
    let dates = if super::has_col(py, panel_df, "__trade_time_date__")? {
        super::col_to_str_vec(py, panel_df, "__trade_time_date__")?
    } else {
        normalized_date_strings(py, panel_df, "trade_time")?
    };
    let clocks = if super::has_col(py, panel_df, "__trade_time_clock_ns__")? {
        super::col_to_i64_vec(py, panel_df, "__trade_time_clock_ns__")?
    } else {
        let raw = if super::has_col(py, panel_df, "__trade_time_ns__")? {
            super::col_to_i64_vec(py, panel_df, "__trade_time_ns__")?
        } else {
            super::col_to_i64_vec(py, panel_df, "trade_time")?
        };
        const NS_PER_DAY: i64 = 86_400_000_000_000;
        raw.into_iter()
            .map(|value| value.rem_euclid(NS_PER_DAY))
            .collect()
    };
    let score_day_match = if super::has_col(py, panel_df, "__trade_time_matches_score_day__")? {
        Some(
            super::col_to_i64_vec(py, panel_df, "__trade_time_matches_score_day__")?
                .into_iter()
                .map(|value| value != 0)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    if dates.len() != length
        || clocks.len() != length
        || score_day_match
            .as_ref()
            .is_some_and(|matches| matches.len() != length)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor intraday kernel trade_time length mismatch",
        ));
    }
    Ok((dates, clocks, score_day_match))
}

fn physical_intraday_rows(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    score_date: NaiveDate,
    require_indexed_score_day: bool,
) -> PyResult<Vec<PhysicalIntradayRow>> {
    let labelled_dates = normalized_date_strings(py, panel_df, "dt")?;
    let length = labelled_dates.len();
    let (physical_dates, clocks, score_day_match) = intraday_time_parts(py, panel_df, length)?;
    let label_score_day_match = if super::has_col(py, panel_df, "__label_matches_score_day__")? {
        Some(
            super::col_to_i64_vec(py, panel_df, "__label_matches_score_day__")?
                .into_iter()
                .map(|value| value != 0)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    if label_score_day_match
        .as_ref()
        .is_some_and(|matches| matches.len() != length)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor intraday kernel labelled score-day length mismatch",
        ));
    }
    if require_indexed_score_day
        && length > 0
        && !(match &label_score_day_match {
            Some(matches) => matches.iter().any(|matched| *matched),
            None => labelled_dates
                .iter()
                .any(|value| parse_date(value) == Some(score_date)),
        })
    {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "{LRD_FACTOR} has no indexed rows for signal day {score_date}"
        )));
    }
    let mut out = Vec::new();
    for index in 0..length {
        let labelled_score_match = label_score_day_match
            .as_ref()
            .map(|matches| matches[index])
            .unwrap_or_else(|| parse_date(&labelled_dates[index]) == Some(score_date));
        let physical_score_match = score_day_match
            .as_ref()
            .map(|matches| matches[index])
            .unwrap_or_else(|| parse_date(&physical_dates[index]) == Some(score_date));
        if !labelled_score_match || !physical_score_match {
            continue;
        }
        if session_label(clocks[index]).is_some() {
            out.push(PhysicalIntradayRow {
                index,
                clock_ns: clocks[index],
            });
        }
    }
    Ok(out)
}

fn intraday_panel_keys(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
) -> PyResult<(Vec<String>, Vec<String>, Vec<i64>)> {
    let raw_dt = super::col_to_str_vec(py, panel_df, "dt")?;
    let codes = super::col_to_str_vec(py, panel_df, "code")?;
    let seq = super::col_to_i64_vec(py, panel_df, "seq")?;
    if raw_dt.len() != codes.len() || raw_dt.len() != seq.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor intraday kernel panel key length mismatch",
        ));
    }
    Ok((raw_dt, codes, seq))
}

fn parse_qed_intraday_rows(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    physical: &[PhysicalIntradayRow],
    ctx: &mut TypedFactorIntradayContext,
) -> PyResult<()> {
    // `_strict_physical_frame` returns None when any one QED field is absent.
    // It is not a schema error: each requested QED output remains present over
    // the labelled score-day index with NaN values.
    for column in QED_REQUIRED_PANEL_COLUMNS {
        if !super::has_col(py, panel_df, column)? {
            return Ok(());
        }
    }
    let (raw_dt, codes, seq) = intraday_panel_keys(py, panel_df)?;
    let length = raw_dt.len();
    let last = super::col_to_f64_vec(py, panel_df, "last")?;
    let ask = super::col_to_f64_vec(py, panel_df, "ask_price1")?;
    let bid = super::col_to_f64_vec(py, panel_df, "bid_price1")?;
    let count = super::col_to_f64_vec(py, panel_df, "num_trades")?;
    if [last.len(), ask.len(), bid.len(), count.len()]
        .iter()
        .any(|candidate| *candidate != length)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor QED kernel panel value length mismatch",
        ));
    }
    for row in physical {
        let index = row.index;
        ctx.qed_rows
            .entry((raw_dt[index].clone(), codes[index].clone()))
            .or_default()
            .push(QuoteRow {
                time_ns: row.clock_ns,
                seq: seq[index],
                last: last[index],
                ask_price1: ask[index],
                bid_price1: bid[index],
                num_trades: count[index],
            });
    }
    Ok(())
}

fn parse_lrd_intraday_rows(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    physical: &[PhysicalIntradayRow],
    ctx: &mut TypedFactorIntradayContext,
) -> PyResult<()> {
    let (raw_dt, codes, seq) = intraday_panel_keys(py, panel_df)?;
    let length = raw_dt.len();
    let mut ask_price = Vec::with_capacity(5);
    let mut bid_price = Vec::with_capacity(5);
    let mut ask_volume = Vec::with_capacity(5);
    let mut bid_volume = Vec::with_capacity(5);
    for level in 1..=5 {
        ask_price.push(super::col_to_f64_vec(
            py,
            panel_df,
            &format!("ask_price{level}"),
        )?);
        bid_price.push(super::col_to_f64_vec(
            py,
            panel_df,
            &format!("bid_price{level}"),
        )?);
        ask_volume.push(super::col_to_f64_vec(
            py,
            panel_df,
            &format!("ask_volume{level}"),
        )?);
        bid_volume.push(super::col_to_f64_vec(
            py,
            panel_df,
            &format!("bid_volume{level}"),
        )?);
    }
    if ask_price
        .iter()
        .chain(bid_price.iter())
        .chain(ask_volume.iter())
        .chain(bid_volume.iter())
        .any(|values| values.len() != length)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor LRD kernel panel value length mismatch",
        ));
    }
    for row in physical {
        let index = row.index;
        let mut ask_prices = [0.0; 5];
        let mut bid_prices = [0.0; 5];
        let mut ask_volumes = [0.0; 5];
        let mut bid_volumes = [0.0; 5];
        for level in 0..5 {
            ask_prices[level] = ask_price[level][index];
            bid_prices[level] = bid_price[level][index];
            ask_volumes[level] = ask_volume[level][index];
            bid_volumes[level] = bid_volume[level][index];
        }
        let key = (raw_dt[index].clone(), codes[index].clone());
        ctx.lrd_keys.insert(key.clone());
        ctx.lrd_rows.entry(key).or_default().push(BookRow {
            time_ns: row.clock_ns,
            seq: seq[index],
            ask_price: ask_prices,
            bid_price: bid_prices,
            ask_volume: ask_volumes,
            bid_volume: bid_volumes,
        });
    }
    Ok(())
}

fn build_intraday_context(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    score_date: Option<NaiveDate>,
    has_qed: bool,
    has_lrd: bool,
) -> PyResult<TypedFactorIntradayContext> {
    let mut ctx = TypedFactorIntradayContext::default();
    if has_lrd {
        require_panel_columns(py, panel_df, LRD_FACTOR, lrd_required_panel_columns())?;
        if let Some(score_date) = score_date {
            let physical = physical_intraday_rows(py, panel_df, score_date, true)?;
            parse_lrd_intraday_rows(py, panel_df, &physical, &mut ctx)?;
        }
    }
    if has_qed {
        let mut all_qed_columns_exist = true;
        for column in QED_REQUIRED_PANEL_COLUMNS {
            if !super::has_col(py, panel_df, column)? {
                all_qed_columns_exist = false;
                break;
            }
        }
        if all_qed_columns_exist {
            if let Some(score_date) = score_date {
                let physical = physical_intraday_rows(py, panel_df, score_date, false)?;
                parse_qed_intraday_rows(py, panel_df, &physical, &mut ctx)?;
            }
        }
    }
    Ok(ctx)
}

fn build_ordered_output(
    py: Python<'_>,
    keys: &[(String, String)],
    values: &[(String, Vec<f64>)],
) -> PyResult<PyObject> {
    let pandas = py.import_bound("pandas")?;
    let data = PyDict::new_bound(py);
    data.set_item(
        "dt",
        keys.iter().map(|(dt, _)| dt.clone()).collect::<Vec<_>>(),
    )?;
    data.set_item(
        "code",
        keys.iter()
            .map(|(_, code)| code.clone())
            .collect::<Vec<_>>(),
    )?;
    for (column, column_values) in values {
        if column_values.len() != keys.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor kernel output length mismatch for {column}"
            )));
        }
        data.set_item(column, column_values)?;
    }
    Ok(pandas.call_method1("DataFrame", (data,))?.into_py(py))
}

// The promotion contract is literal float equality, not mathematical
// closeness.  NumPy's reductions use a different accumulation order from a
// naive Rust iterator on some inputs.  For the handful of cheap daily scalar
// reductions below, keep the Rust data contract and routing but invoke the
// same NumPy primitive as the golden Python kernels.  This is intentional:
// it is a correctness bridge, not a relaxed tolerance or a silent fallback to
// the Python factor implementation.
fn numpy_array<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyAny>> {
    py.import_bound("numpy")?
        .call_method1("array", (values.to_vec(),))
}

fn numpy_mean(py: Python<'_>, values: &[f64]) -> PyResult<f64> {
    let numpy = py.import_bound("numpy")?;
    let array = numpy_array(py, values)?;
    numpy.call_method1("mean", (array,))?.extract::<f64>()
}

fn numpy_sample_std(py: Python<'_>, values: &[f64]) -> PyResult<f64> {
    let numpy = py.import_bound("numpy")?;
    let array = numpy_array(py, values)?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("ddof", 1)?;
    numpy
        .call_method("std", (array,), Some(&kwargs))?
        .extract::<f64>()
}

fn numpy_log(py: Python<'_>, values: &[f64]) -> PyResult<Vec<f64>> {
    let numpy = py.import_bound("numpy")?;
    let array = numpy_array(py, values)?;
    numpy
        .call_method1("log", (array,))?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()
}

fn intraday_numpy_binary(
    py: Python<'_>,
    name: &str,
    left: &[f64],
    right: &[f64],
) -> PyResult<Vec<f64>> {
    if left.len() != right.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "typed_factor intraday NumPy {name} length mismatch"
        )));
    }
    let numpy = py.import_bound("numpy")?;
    let left = numpy_array(py, left)?;
    let right = numpy_array(py, right)?;
    numpy
        .call_method1(name, (left, right))?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()
}

fn intraday_numpy_unary(py: Python<'_>, name: &str, values: &[f64]) -> PyResult<Vec<f64>> {
    let numpy = py.import_bound("numpy")?;
    let values = numpy_array(py, values)?;
    numpy
        .call_method1(name, (values,))?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()
}

fn intraday_numpy_sum(py: Python<'_>, values: &[f64]) -> PyResult<f64> {
    let numpy = py.import_bound("numpy")?;
    let values = numpy_array(py, values)?;
    numpy.call_method1("sum", (values,))?.extract::<f64>()
}

fn intraday_numpy_dot(py: Python<'_>, left: &[f64], right: &[f64]) -> PyResult<f64> {
    if left.len() != right.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor intraday NumPy dot length mismatch",
        ));
    }
    let numpy = py.import_bound("numpy")?;
    let left = numpy_array(py, left)?;
    let right = numpy_array(py, right)?;
    numpy.call_method1("dot", (left, right))?.extract::<f64>()
}

fn intraday_numpy_divide_scalar(py: Python<'_>, left: f64, right: f64) -> PyResult<f64> {
    py.import_bound("numpy")?
        .call_method1("divide", (left, right))?
        .extract::<f64>()
}

fn qed_nan_metrics() -> QedTypedFactorMetrics {
    QedTypedFactorMetrics {
        prior_quote_location_dispersion: f64::NAN,
        prior_quote_tail_penetration: f64::NAN,
        prior_quote_lag2_agreement: f64::NAN,
    }
}

fn ordered_qed_rows(rows: &[QuoteRow]) -> Vec<QuoteRow> {
    let mut ordered = rows
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, row)| session_label(row.time_ns).is_some())
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.1
            .time_ns
            .cmp(&right.1.time_ns)
            .then_with(|| left.1.seq.cmp(&right.1.seq))
            .then_with(|| left.0.cmp(&right.0))
    });
    ordered.into_iter().map(|(_, row)| row).collect()
}

fn valid_qed_row(row: &QuoteRow) -> bool {
    row.last.is_finite()
        && row.ask_price1.is_finite()
        && row.bid_price1.is_finite()
        && row.num_trades.is_finite()
        && row.last > 0.0
        && row.ask_price1 > 0.0
        && row.bid_price1 > 0.0
        && row.num_trades >= 0.0
        && row.ask_price1 >= row.bid_price1
}

/// NumPy-finalized QED metrics.  Rust owns the typed physical-panel parsing,
/// stable sort, session gate, and fail-closed checks; the sensitive array
/// arithmetic deliberately uses the same NumPy primitives as the Python
/// reference so acceptance remains IEEE-bit exact rather than merely close.
fn exact_qed_metrics(py: Python<'_>, rows: &[QuoteRow]) -> PyResult<QedTypedFactorMetrics> {
    // Keep the independently unit-tested pure kernel on the same typed input.
    // Its reductions are intentionally not returned here because NumPy's
    // accumulator/order is the promotion contract.
    let _ = quote_execution_typed_factor(rows);
    let rows = ordered_qed_rows(rows);
    if rows.len() < 12
        || rows
            .windows(2)
            .any(|pair| pair[0].time_ns == pair[1].time_ns)
        || rows.iter().any(|row| !valid_qed_row(row))
    {
        return Ok(qed_nan_metrics());
    }

    let last = rows.iter().map(|row| row.last).collect::<Vec<_>>();
    let ask = rows.iter().map(|row| row.ask_price1).collect::<Vec<_>>();
    let bid = rows.iter().map(|row| row.bid_price1).collect::<Vec<_>>();
    let count = rows.iter().map(|row| row.num_trades).collect::<Vec<_>>();
    let numpy = py.import_bound("numpy")?;
    let increments = numpy
        .call_method1("diff", (numpy_array(py, &count)?,))?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()?;
    let prior_ask = &ask[..ask.len() - 1];
    let prior_bid = &bid[..bid.len() - 1];
    let spread = intraday_numpy_binary(py, "subtract", prior_ask, prior_bid)?;
    if increments
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Ok(qed_nan_metrics());
    }
    let midpoint_sum = intraday_numpy_binary(py, "add", prior_ask, prior_bid)?;
    let two = vec![2.0; midpoint_sum.len()];
    let midpoint = intraday_numpy_binary(py, "divide", &midpoint_sum, &two)?;
    let numerator = intraday_numpy_binary(py, "subtract", &last[1..], &midpoint)?;
    let denominator = intraday_numpy_binary(py, "divide", &spread, &two)?;
    let locations_all = intraday_numpy_binary(py, "divide", &numerator, &denominator)?;
    let mut locations = Vec::new();
    let mut weights = Vec::new();
    for index in 0..increments.len() {
        if increments[index] > 0.0 && spread[index] > 1e-12 {
            locations.push(locations_all[index]);
            weights.push(increments[index]);
        }
    }
    if locations.len() < 4
        || locations.iter().any(|value| !value.is_finite())
        || weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 1e-12)
    {
        return Ok(qed_nan_metrics());
    }

    let total = intraday_numpy_sum(py, &weights)?;
    if !total.is_finite() || total <= 1e-12 {
        return Ok(qed_nan_metrics());
    }
    let center =
        intraday_numpy_divide_scalar(py, intraday_numpy_dot(py, &weights, &locations)?, total)?;
    let centered =
        intraday_numpy_binary(py, "subtract", &locations, &vec![center; locations.len()])?;
    // `(location - center) ** 2` invokes NumPy's `power` ufunc in the Python
    // reference. Do not substitute multiplication: the promotion contract is
    // bit-level identity, including edge floating-point inputs.
    let squared = intraday_numpy_binary(py, "power", &centered, &vec![2.0; centered.len()])?;
    let variance =
        intraday_numpy_divide_scalar(py, intraday_numpy_dot(py, &weights, &squared)?, total)?;
    let (dispersion, tail_penetration) = if !variance.is_finite() || variance < 0.0 {
        (f64::NAN, f64::NAN)
    } else {
        let root = intraday_numpy_unary(py, "sqrt", &[variance.max(0.0)])?[0];
        let absolute = intraday_numpy_unary(py, "absolute", &locations)?;
        let tail_base =
            intraday_numpy_binary(py, "subtract", &absolute, &vec![1.0; absolute.len()])?;
        let tail_clipped =
            intraday_numpy_binary(py, "maximum", &tail_base, &vec![0.0; tail_base.len()])?;
        let penetration = intraday_numpy_unary(py, "log1p", &tail_clipped)?;
        let tail = intraday_numpy_divide_scalar(
            py,
            intraday_numpy_dot(py, &weights, &penetration)?,
            total,
        )?;
        (root, tail)
    };
    let directions = locations
        .iter()
        .filter_map(|location| {
            if *location >= 1.0 - QED_AT_QUOTE_TOL {
                Some(1.0)
            } else if *location <= -1.0 + QED_AT_QUOTE_TOL {
                Some(-1.0)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let lag2_agreement = if directions.len() < 4 {
        f64::NAN
    } else {
        let product = intraday_numpy_binary(
            py,
            "multiply",
            &directions[2..],
            &directions[..directions.len() - 2],
        )?;
        numpy
            .call_method1("mean", (numpy_array(py, &product)?,))?
            .extract::<f64>()?
    };
    Ok(QedTypedFactorMetrics {
        prior_quote_location_dispersion: if dispersion.is_finite() {
            dispersion
        } else {
            f64::NAN
        },
        prior_quote_tail_penetration: if tail_penetration.is_finite() {
            tail_penetration
        } else {
            f64::NAN
        },
        prior_quote_lag2_agreement: if lag2_agreement.is_finite() {
            lag2_agreement
        } else {
            f64::NAN
        },
    })
}

fn ordered_lrd_rows(rows: &[BookRow]) -> Vec<(BookRow, u8)> {
    let mut ordered = rows
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, row)| session_label(row.time_ns).map(|session| (index, row, session)))
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.1
            .time_ns
            .cmp(&right.1.time_ns)
            .then_with(|| left.1.seq.cmp(&right.1.seq))
            .then_with(|| left.0.cmp(&right.0))
    });
    ordered
        .into_iter()
        .map(|(_, row, session)| (row, session))
        .collect()
}

fn valid_lrd_row(row: &BookRow) -> bool {
    row.ask_price
        .iter()
        .chain(row.bid_price.iter())
        .all(|value| value.is_finite() && *value > 0.0)
        && row
            .ask_volume
            .iter()
            .chain(row.bid_volume.iter())
            .all(|value| value.is_finite() && *value >= 0.0)
        && row.ask_price[0] >= row.bid_price[0]
        && row.ask_price.windows(2).all(|pair| pair[1] >= pair[0])
        && row.bid_price.windows(2).all(|pair| pair[1] <= pair[0])
}

fn lrd_numpy_directions(
    py: Python<'_>,
    rows: &[(BookRow, u8)],
    bid_side: bool,
) -> PyResult<Option<Vec<f64>>> {
    let intervals = rows.len().saturating_sub(1);
    let mut moves_by_level = Vec::with_capacity(5);
    for level in 0..5 {
        let prior = rows
            .iter()
            .take(intervals)
            .map(|(row, _)| {
                if bid_side {
                    row.bid_price[level]
                } else {
                    row.ask_price[level]
                }
            })
            .collect::<Vec<_>>();
        let current = rows
            .iter()
            .skip(1)
            .map(|(row, _)| {
                if bid_side {
                    row.bid_price[level]
                } else {
                    row.ask_price[level]
                }
            })
            .collect::<Vec<_>>();
        let ratio = intraday_numpy_binary(py, "divide", &current, &prior)?;
        let log_move = intraday_numpy_unary(py, "log", &ratio)?;
        if log_move.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        moves_by_level.push(log_move);
    }
    let mut direction = Vec::with_capacity(intervals);
    for index in 0..intervals {
        if rows[index].1 != rows[index + 1].1 {
            direction.push(0.0);
            continue;
        }
        let positive = moves_by_level
            .iter()
            .filter(|moves| moves[index] > 1e-12)
            .count();
        let negative = moves_by_level
            .iter()
            .filter(|moves| moves[index] < -1e-12)
            .count();
        direction.push(if positive >= 3 && positive > negative {
            1.0
        } else if negative >= 3 && negative > positive {
            -1.0
        } else {
            0.0
        });
    }
    Ok(Some(direction))
}

/// NumPy-finalized LRD symmetry.  As with QED, the pure Rust kernel remains
/// the typed implementation target, while NumPy computes the log moves and
/// final mean to preserve the Python threshold and reduction bit path.
fn exact_lrd_value(py: Python<'_>, rows: &[BookRow]) -> PyResult<f64> {
    let _ = lrd_cross_side_reprice_symmetry(rows);
    let rows = ordered_lrd_rows(rows);
    if rows.len() < 12 || rows.iter().any(|(row, _)| !valid_lrd_row(row)) {
        return Ok(f64::NAN);
    }
    let Some(bid_direction) = lrd_numpy_directions(py, &rows, true)? else {
        return Ok(f64::NAN);
    };
    let Some(ask_direction) = lrd_numpy_directions(py, &rows, false)? else {
        return Ok(f64::NAN);
    };
    let products = bid_direction
        .iter()
        .zip(ask_direction.iter())
        .filter_map(|(bid, ask)| (*bid != 0.0 && *ask != 0.0).then_some(bid * ask))
        .collect::<Vec<_>>();
    if products.len() < 3 {
        return Ok(f64::NAN);
    }
    let value = numpy_mean(py, &products)?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

/// Reconstruct the strict-family global-calendar tail used by the OHLC path
/// factors.  The parser has already applied `< score_date`, strict canonical
/// codes, and global duplicate rejection, so this function only carries the
/// Python feature-frame's source-calendar anchor/window semantics.
fn strict_path_recent_rows<'a>(
    ctx: &'a TypedFactorDailyPathContext,
    code: &str,
) -> Vec<&'a TypedFactorDailyPathPriceRow> {
    let calendar: BTreeSet<NaiveDate> = ctx
        .price_by_code
        .values()
        .flat_map(|rows| rows.iter().map(|row| row.trade_date))
        .collect();
    let Some(anchor) = calendar.last().copied() else {
        return Vec::new();
    };
    let Some(rows) = ctx.price_by_code.get(code) else {
        return Vec::new();
    };
    if rows.last().map(|row| row.trade_date) != Some(anchor) {
        return Vec::new();
    }
    let start = calendar.len().saturating_sub(60);
    let Some(first_date) = calendar.iter().nth(start).copied() else {
        return Vec::new();
    };
    rows.iter()
        .filter(|row| row.trade_date >= first_date)
        .collect()
}

fn path_wick_value(row: &TypedFactorDailyPathPriceRow) -> f64 {
    let width = row.high_price - row.low_price;
    let valid = row.open_price > 1e-12
        && row.high_price > 1e-12
        && row.low_price > 1e-12
        && row.close_price > 1e-12
        && row.prev_close_price > 1e-12
        && width > 1e-12
        && row.high_price >= row.open_price
        && row.high_price >= row.close_price
        && row.low_price <= row.open_price
        && row.low_price <= row.close_price;
    if !valid {
        return f64::NAN;
    }
    let top = row.open_price.max(row.close_price);
    let bottom = row.open_price.min(row.close_price);
    (row.high_price - top - (bottom - row.low_price)) / width
}

fn path_log_ratios(rows: &[&TypedFactorDailyPathPriceRow]) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let mut intraday = Vec::with_capacity(rows.len());
    let mut ranges = Vec::with_capacity(rows.len());
    for row in rows {
        let width = row.high_price - row.low_price;
        let valid = row.open_price > 1e-12
            && row.high_price > 1e-12
            && row.low_price > 1e-12
            && row.close_price > 1e-12
            && row.prev_close_price > 1e-12
            && width > 1e-12
            && row.high_price >= row.open_price
            && row.high_price >= row.close_price
            && row.low_price <= row.open_price
            && row.low_price <= row.close_price;
        if valid {
            intraday.push(row.close_price / row.open_price);
            ranges.push(row.high_price / row.low_price);
        } else {
            intraday.push(f64::NAN);
            ranges.push(f64::NAN);
        }
    }
    Ok((intraday, ranges))
}

/// Same structural path as `typed_factor_daily_paths`, with only NumPy's own
/// elementwise `log` and final `mean` reductions used to preserve literal
/// Python IEEE results.  This is not a Python-factor fallback: source parsing,
/// canonicalisation, global-calendar construction, and state masks stay Rust.
fn exact_dohw_mean_wick(
    py: Python<'_>,
    ctx: &TypedFactorDailyPathContext,
    code: &str,
) -> PyResult<f64> {
    let recent = strict_path_recent_rows(ctx, code);
    let Some(terminal) = recent.last() else {
        return Ok(f64::NAN);
    };
    if !path_wick_value(terminal).is_finite() {
        return Ok(f64::NAN);
    }
    let values: Vec<f64> = recent
        .iter()
        .map(|row| path_wick_value(row))
        .filter(|value| value.is_finite())
        .collect();
    if values.len() < 45 {
        return Ok(f64::NAN);
    }
    let value = numpy_mean(py, &values)?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

fn exact_dohw_intraday_sign_range(
    py: Python<'_>,
    ctx: &TypedFactorDailyPathContext,
    code: &str,
) -> PyResult<f64> {
    let recent = strict_path_recent_rows(ctx, code);
    if recent.is_empty() {
        return Ok(f64::NAN);
    }
    let (return_ratios, range_ratios) = path_log_ratios(&recent)?;
    let intraday = numpy_log(py, &return_ratios)?;
    let ranges = numpy_log(py, &range_ratios)?;
    let terminal_return = intraday.last().copied().unwrap_or(f64::NAN);
    let terminal_range = ranges.last().copied().unwrap_or(f64::NAN);
    if !terminal_return.is_finite() || !terminal_range.is_finite() {
        return Ok(f64::NAN);
    }
    let paired: Vec<(f64, f64)> = intraday
        .into_iter()
        .zip(ranges)
        .filter(|(ret, range)| ret.is_finite() && range.is_finite())
        .collect();
    if paired.len() < 45 {
        return Ok(f64::NAN);
    }
    let positive: Vec<f64> = paired
        .iter()
        .filter(|(ret, _)| *ret > 0.0)
        .map(|(_, range)| *range)
        .collect();
    let negative: Vec<f64> = paired
        .iter()
        .filter(|(ret, _)| *ret < 0.0)
        .map(|(_, range)| *range)
        .collect();
    if positive.len() < 8 || negative.len() < 8 {
        return Ok(f64::NAN);
    }
    let value = numpy_mean(py, &positive)? - numpy_mean(py, &negative)?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

/// Return the exact 41-row strict tail that the Python bstk family reaches
/// after its date-key inner join to the independent daily-price calendar.
/// The terminal code must be present in the current price anchor and the tail
/// must occupy the final 41 adjacent global price sessions.
fn strict_bstk_tail_rows<'a>(
    ctx: &'a TypedFactorDailyTrackingContext,
    code: &str,
) -> Option<Vec<&'a TypedFactorDailyTrackingBaseRow>> {
    let calendar: BTreeSet<NaiveDate> = ctx
        .price_anchor_by_code
        .values()
        .flat_map(|rows| rows.iter().map(|row| row.trade_date))
        .collect();
    let anchor = calendar.last().copied()?;
    let price_rows = ctx.price_anchor_by_code.get(code)?;
    if price_rows.last().map(|row| row.trade_date) != Some(anchor) {
        return None;
    }
    let price_dates: BTreeSet<NaiveDate> = price_rows.iter().map(|row| row.trade_date).collect();
    let base_rows = ctx.base_by_code.get(code)?;
    let joined: Vec<&TypedFactorDailyTrackingBaseRow> = base_rows
        .iter()
        .filter(|row| price_dates.contains(&row.trade_date))
        .collect();
    if joined.last().map(|row| row.trade_date) != Some(anchor) || joined.len() < 41 {
        return None;
    }
    let tail = joined[joined.len() - 41..].to_vec();
    let expected: Vec<NaiveDate> = calendar.iter().rev().take(41).copied().rev().collect();
    if expected.len() != 41
        || tail
            .iter()
            .map(|row| row.trade_date)
            .ne(expected.into_iter())
    {
        return None;
    }
    Some(tail)
}

fn numpy_returns(py: Python<'_>, previous: &[f64], current: &[f64]) -> PyResult<Vec<f64>> {
    let previous = numpy_array(py, previous)?;
    let current = numpy_array(py, current)?;
    current
        .call_method1("__truediv__", (&previous,))?
        .call_method1("__sub__", (1.0,))?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()
}

fn numpy_safe_beta(py: Python<'_>, stock: &[f64], bond: &[f64]) -> PyResult<f64> {
    if stock.len() < 5
        || stock.len() != bond.len()
        || stock.iter().any(|value| !value.is_finite())
        || bond.iter().any(|value| !value.is_finite())
    {
        return Ok(f64::NAN);
    }
    let numpy = py.import_bound("numpy")?;
    let stock = numpy_array(py, stock)?;
    let bond = numpy_array(py, bond)?;
    let stock_mean = numpy.call_method1("mean", (&stock,))?.extract::<f64>()?;
    let centered_stock = stock.call_method1("__sub__", (stock_mean,))?;
    let denominator = numpy
        .call_method1("dot", (&centered_stock, &centered_stock))?
        .extract::<f64>()?;
    if !denominator.is_finite() || denominator <= 1e-12 {
        return Ok(f64::NAN);
    }
    let bond_mean = numpy.call_method1("mean", (&bond,))?.extract::<f64>()?;
    let centered_bond = bond.call_method1("__sub__", (bond_mean,))?;
    let numerator = numpy
        .call_method1("dot", (&centered_stock, &centered_bond))?
        .extract::<f64>()?;
    let value = numerator / denominator;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

/// NumPy bridge for the bstk signal's final return/beta/quantile/mean
/// reductions.  Rust retains source contracts, code canonicalisation, anchor
/// validation, date-key joining, and complete-session checks above.
fn exact_bstk_tail_cocrash(
    py: Python<'_>,
    ctx: &TypedFactorDailyTrackingContext,
    code: &str,
) -> PyResult<f64> {
    let Some(tail) = strict_bstk_tail_rows(ctx, code) else {
        return Ok(f64::NAN);
    };
    if tail.iter().any(|row| {
        !row.cb_prev_close_price.is_finite()
            || !row.cb_close_price.is_finite()
            || !row.stk_prev_close_price.is_finite()
            || !row.stk_close_price.is_finite()
            || row.cb_prev_close_price <= 1e-12
            || row.cb_close_price <= 1e-12
            || row.stk_prev_close_price <= 1e-12
            || row.stk_close_price <= 1e-12
    }) {
        return Ok(f64::NAN);
    }
    let cb_previous: Vec<f64> = tail.iter().map(|row| row.cb_prev_close_price).collect();
    let cb_close: Vec<f64> = tail.iter().map(|row| row.cb_close_price).collect();
    let stock_previous: Vec<f64> = tail.iter().map(|row| row.stk_prev_close_price).collect();
    let stock_close: Vec<f64> = tail.iter().map(|row| row.stk_close_price).collect();
    let bond_returns = numpy_returns(py, &cb_previous, &cb_close)?;
    let stock_returns = numpy_returns(py, &stock_previous, &stock_close)?;
    let beta20 = numpy_safe_beta(py, &stock_returns[20..40], &bond_returns[20..40])?;
    if !beta20.is_finite() {
        return Ok(f64::NAN);
    }
    let numpy = py.import_bound("numpy")?;
    let residual = numpy_array(py, &bond_returns[21..])?.call_method1(
        "__sub__",
        (numpy_array(py, &stock_returns[21..])?.call_method1("__mul__", (beta20,))?,),
    )?;
    let stock_tail = numpy_array(py, &stock_returns[21..])?;
    let threshold = numpy
        .call_method1("quantile", (&stock_tail, 0.2))?
        .extract::<f64>()?;
    if !threshold.is_finite() {
        return Ok(f64::NAN);
    }
    let residual_values = residual.call_method0("tolist")?.extract::<Vec<f64>>()?;
    let selected: Vec<f64> = residual_values
        .into_iter()
        .zip(stock_returns[21..].iter().copied())
        .filter_map(|(value, stock)| (stock <= threshold).then_some(value))
        .collect();
    if selected.len() < 4 || selected.iter().any(|value| !value.is_finite()) {
        return Ok(f64::NAN);
    }
    let value = numpy_mean(py, &selected)?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

fn numpy_corrcoef(py: Python<'_>, left: &[f64], right: &[f64]) -> PyResult<f64> {
    let numpy = py.import_bound("numpy")?;
    let left = numpy_array(py, left)?;
    let right = numpy_array(py, right)?;
    let rows = numpy
        .call_method1("corrcoef", (left, right))?
        .call_method0("tolist")?
        .extract::<Vec<Vec<f64>>>()?;
    Ok(rows
        .get(0)
        .and_then(|row| row.get(1))
        .copied()
        .unwrap_or(f64::NAN))
}

fn finite_tail(values: impl IntoIterator<Item = f64>, count: usize, min_count: usize) -> Vec<f64> {
    let finite: Vec<f64> = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < min_count {
        Vec::new()
    } else {
        finite[finite.len().saturating_sub(count)..].to_vec()
    }
}

fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() <= 1e-12 {
        f64::NAN
    } else {
        let value = numerator / denominator;
        if value.is_finite() {
            value
        } else {
            f64::NAN
        }
    }
}

fn daily_return(close: f64, previous_close: f64) -> f64 {
    if close.is_finite() && previous_close.is_finite() && previous_close.abs() > 1e-12 {
        let value = close / previous_close - 1.0;
        if value.is_finite() {
            value
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    }
}

fn exact_dredemption_z(py: Python<'_>, ctx: &TypedFactorDailyContext, code: &str) -> PyResult<f64> {
    let values: Vec<f64> = ctx
        .base_by_code
        .get(code)
        .map(|rows| rows.iter().map(|row| row.redemption_prem_ratio).collect())
        .unwrap_or_default();
    let latest = values.last().copied().unwrap_or(f64::NAN);
    let tail = finite_tail(values, 20, 12);
    if !latest.is_finite() || tail.len() < 2 {
        return Ok(f64::NAN);
    }
    Ok(safe_div(
        latest - numpy_mean(py, &tail)?,
        numpy_sample_std(py, &tail)?,
    ))
}

fn exact_dret_volatility(
    py: Python<'_>,
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> PyResult<f64> {
    let returns = ctx
        .price_by_code
        .get(code)
        .map(|rows| {
            rows.iter()
                .map(|row| daily_return(row.close_price, row.prev_close_price))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let tail = finite_tail(returns, 20, 12);
    if tail.len() < 2 {
        Ok(f64::NAN)
    } else {
        numpy_sample_std(py, &tail)
    }
}

fn exact_dtwap_mean(py: Python<'_>, ctx: &TypedFactorDailyContext, code: &str) -> PyResult<f64> {
    let values = ctx
        .twap_by_code
        .get(code)
        .map(|rows| {
            rows.iter()
                .map(|row| daily_return(row.twap_0935_1000, row.twap_0930_0935))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let tail = finite_tail(values, 20, 12);
    if tail.is_empty() {
        Ok(f64::NAN)
    } else {
        numpy_mean(py, &tail)
    }
}

fn exact_dliq_corr(py: Python<'_>, ctx: &TypedFactorDailyContext, code: &str) -> PyResult<f64> {
    let Some(rows) = ctx.price_by_code.get(code) else {
        return Ok(f64::NAN);
    };
    let log_input: Vec<f64> = rows
        .iter()
        .map(|row| {
            if row.volume.is_finite() && row.volume > 0.0 {
                row.volume
            } else {
                f64::NAN
            }
        })
        .collect();
    let log_volume = numpy_log(py, &log_input)?;
    let returns: Vec<f64> = rows
        .iter()
        .map(|row| daily_return(row.close_price, row.prev_close_price))
        .collect();
    let mut pairs: Vec<(f64, f64)> = log_volume
        .into_iter()
        .zip(returns)
        // Pandas dropna keeps infinities, exactly as the reference does.
        .filter(|(left, right)| !left.is_nan() && !right.is_nan())
        .collect();
    if pairs.len() > 20 {
        pairs = pairs.split_off(pairs.len() - 20);
    }
    if pairs.len() < 12 {
        return Ok(f64::NAN);
    }
    let left: Vec<f64> = pairs.iter().map(|(value, _)| *value).collect();
    let right: Vec<f64> = pairs.iter().map(|(_, value)| *value).collect();
    let left_std = numpy_sample_std(py, &left)?;
    let right_std = numpy_sample_std(py, &right)?;
    if !left_std.is_finite() || !right_std.is_finite() || left_std <= 1e-12 || right_std <= 1e-12 {
        return Ok(f64::NAN);
    }
    numpy_corrcoef(py, &left, &right)
}

fn exact_p1_value(
    py: Python<'_>,
    ctx: &TypedFactorDailyContext,
    code: &str,
    signal: &str,
) -> PyResult<f64> {
    match signal {
        "dredemption_premium_z20" => exact_dredemption_z(py, ctx, code),
        "dret_volatility_20" => exact_dret_volatility(py, ctx, code),
        "dliq_volume_return_corr20" => exact_dliq_corr(py, ctx, code),
        "dtwap_morning_slope20" => exact_dtwap_mean(py, ctx, code),
        _ => compute_p1_daily_signal(ctx, code, signal)
            .map_err(|error| PyErr::new::<PyRuntimeError, _>(error.to_string())),
    }
}

/// Reproduce the Python RJST final reduction literally after the strict
/// Rust-side state/transition builder has supplied non-zero 9x9 counts:
/// `-np.sum(p * np.log(p)) / np.log(81.0)`. Keeping this bridge local avoids
/// an ULP drift from Rust's reduction order without invoking the Python factor.
fn numpy_rjst_entropy(py: Python<'_>, nonzero_counts: &[f64]) -> PyResult<f64> {
    if nonzero_counts.is_empty() {
        return Ok(f64::NAN);
    }
    let numpy = py.import_bound("numpy")?;
    let counts = numpy_array(py, nonzero_counts)?;
    let total = counts.call_method0("sum")?;
    let probabilities = counts.call_method1("__truediv__", (&total,))?;
    let log_probabilities = numpy.call_method1("log", (&probabilities,))?;
    let product = probabilities.call_method1("__mul__", (&log_probabilities,))?;
    let summed = numpy.call_method1("sum", (&product,))?;
    let normalizer = numpy.call_method1("log", (81.0,))?;
    let value = summed
        .call_method0("__neg__")?
        .call_method1("__truediv__", (&normalizer,))?
        .extract::<f64>()?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

/// Use the same NumPy elementwise division/log path as the strict daily
/// references.  `finite_required=false` intentionally preserves the rank
/// families' treatment of +/-Inf as rankable values.
fn numpy_log_ratio(
    py: Python<'_>,
    numerator: &[f64],
    denominator: &[f64],
    finite_required: bool,
) -> PyResult<Vec<f64>> {
    if numerator.len() != denominator.len() {
        return Ok(Vec::new());
    }
    let valid = |left: f64, right: f64| {
        left > 1e-12
            && right > 1e-12
            && (!finite_required || (left.is_finite() && right.is_finite()))
    };
    let left: Vec<f64> = numerator
        .iter()
        .copied()
        .zip(denominator.iter().copied())
        .map(|(left, right)| if valid(left, right) { left } else { f64::NAN })
        .collect();
    let right: Vec<f64> = numerator
        .iter()
        .copied()
        .zip(denominator.iter().copied())
        .map(|(left, right)| if valid(left, right) { right } else { f64::NAN })
        .collect();
    let numpy = py.import_bound("numpy")?;
    let quotient =
        numpy_array(py, &left)?.call_method1("__truediv__", (numpy_array(py, &right)?,))?;
    numpy
        .call_method1("log", (&quotient,))?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()
}

fn numpy_subtract(py: Python<'_>, left: &[f64], right: &[f64]) -> PyResult<Vec<f64>> {
    if left.len() != right.len() {
        return Ok(Vec::new());
    }
    numpy_array(py, left)?
        .call_method1("__sub__", (numpy_array(py, right)?,))?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()
}

fn numpy_adjacent_difference(py: Python<'_>, values: &[f64]) -> PyResult<Vec<f64>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let numpy = py.import_bound("numpy")?;
    let diffs = numpy
        .call_method1("diff", (numpy_array(py, values)?,))?
        .call_method0("tolist")?
        .extract::<Vec<f64>>()?;
    let mut output = Vec::with_capacity(values.len());
    output.push(f64::NAN);
    output.extend(diffs);
    Ok(output)
}

/// Literal counterpart of the references' `mean -> center -> dot -> sqrt ->
/// dot/divide` Pearson calculation.  Do not replace this with `corrcoef`: the
/// Python family source does not use that reduction order.
fn numpy_centered_correlation(py: Python<'_>, left: &[f64], right: &[f64]) -> PyResult<f64> {
    if left.len() != right.len() || left.is_empty() {
        return Ok(f64::NAN);
    }
    let numpy = py.import_bound("numpy")?;
    let left = numpy_array(py, left)?;
    let right = numpy_array(py, right)?;
    let left_mean = numpy.call_method1("mean", (&left,))?.extract::<f64>()?;
    let right_mean = numpy.call_method1("mean", (&right,))?.extract::<f64>()?;
    let centered_left = left.call_method1("__sub__", (left_mean,))?;
    let centered_right = right.call_method1("__sub__", (right_mean,))?;
    let left_square = numpy
        .call_method1("dot", (&centered_left, &centered_left))?
        .extract::<f64>()?;
    let right_square = numpy
        .call_method1("dot", (&centered_right, &centered_right))?
        .extract::<f64>()?;
    let denominator = numpy
        .call_method1(
            "sqrt",
            (numpy.call_method1("multiply", (left_square, right_square))?,),
        )?
        .extract::<f64>()?;
    if !denominator.is_finite() || denominator <= 1e-12 {
        return Ok(f64::NAN);
    }
    let numerator = numpy
        .call_method1("dot", (&centered_left, &centered_right))?
        .extract::<f64>()?;
    let value = numpy
        .call_method1("divide", (numerator, denominator))?
        .extract::<f64>()?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

/// Literal version of `_slope` in the YDPT family.  Selection and source
/// state construction stay Rust-side; NumPy owns all final arithmetic.
fn numpy_ols_slope(py: Python<'_>, left: &[f64], right: &[f64]) -> PyResult<f64> {
    if left.len() != right.len()
        || left.len() < 8
        || left.iter().any(|value| !value.is_finite())
        || right.iter().any(|value| !value.is_finite())
    {
        return Ok(f64::NAN);
    }
    let numpy = py.import_bound("numpy")?;
    let left = numpy_array(py, left)?;
    let right = numpy_array(py, right)?;
    let left_mean = numpy.call_method1("mean", (&left,))?.extract::<f64>()?;
    let right_mean = numpy.call_method1("mean", (&right,))?.extract::<f64>()?;
    let centered_left = left.call_method1("__sub__", (left_mean,))?;
    let denominator = numpy
        .call_method1("dot", (&centered_left, &centered_left))?
        .extract::<f64>()?;
    if !denominator.is_finite() || denominator <= 1e-12 {
        return Ok(f64::NAN);
    }
    let centered_right = right.call_method1("__sub__", (right_mean,))?;
    let numerator = numpy
        .call_method1("dot", (&centered_left, &centered_right))?
        .extract::<f64>()?;
    let value = numpy
        .call_method1("divide", (numerator, denominator))?
        .extract::<f64>()?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

fn numpy_ptp(py: Python<'_>, values: &[f64]) -> PyResult<f64> {
    if values.is_empty() {
        return Ok(f64::NAN);
    }
    py.import_bound("numpy")?
        .call_method1("ptp", (numpy_array(py, values)?,))?
        .extract::<f64>()
}

/// Pandas `rank(method="average", pct=True)` for a duplicate-free code set.
/// Pandas excludes NaN but ranks infinities, so this must not use `is_finite`.
fn pandas_average_pct_rank(values: &[f64]) -> Vec<f64> {
    let mut result = vec![f64::NAN; values.len()];
    let mut ordered: Vec<(f64, usize)> = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| (!value.is_nan()).then_some((value, index)))
        .collect();
    ordered.sort_by(|(left, _), (right, _)| {
        left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
    });
    if ordered.is_empty() {
        return result;
    }
    let denominator = ordered.len() as f64;
    let mut start = 0;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && ordered[end].0 == ordered[start].0 {
            end += 1;
        }
        let percentile = ((start + 1 + end) as f64) * 0.5 / denominator;
        for &(_, index) in &ordered[start..end] {
            result[index] = percentile;
        }
        start = end;
    }
    result
}

#[derive(Clone)]
struct RankStateJoinedRow {
    trade_date: NaiveDate,
    code: String,
    prev_close_price: f64,
    act_prev_close_price: f64,
    close_price: f64,
    amount: f64,
    remain_size: f64,
    current_yield: f64,
}

fn rank_state_joined_history(
    ctx: &TypedFactorRankStateContext,
) -> Option<(Vec<RankStateJoinedRow>, NaiveDate)> {
    let price: Vec<_> = ctx
        .price_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .collect();
    let base: Vec<_> = ctx
        .base_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .collect();
    let price_anchor = price.iter().map(|row| row.trade_date).max()?;
    let base_anchor = base.iter().map(|row| row.trade_date).max()?;
    if price_anchor != base_anchor {
        return None;
    }
    let base_by_key: BTreeMap<(NaiveDate, String), &TypedFactorRankStateBaseRow> = base
        .iter()
        .map(|row| ((row.trade_date, row.code.clone()), *row))
        .collect();
    let mut joined = Vec::new();
    for row in price {
        let Some(base_row) = base_by_key.get(&(row.trade_date, row.code.clone())) else {
            continue;
        };
        joined.push(RankStateJoinedRow {
            trade_date: row.trade_date,
            code: row.code.clone(),
            prev_close_price: row.prev_close_price,
            act_prev_close_price: row.act_prev_close_price,
            close_price: row.close_price,
            amount: row.amount,
            remain_size: base_row.remain_size,
            current_yield: base_row.current_yield,
        });
    }
    joined.sort_by(|left, right| {
        (left.trade_date, left.code.as_str()).cmp(&(right.trade_date, right.code.as_str()))
    });
    Some((joined, price_anchor))
}

fn exact_prcn_rank_correlation(
    py: Python<'_>,
    ctx: &TypedFactorRankStateContext,
    code: &str,
) -> PyResult<f64> {
    let Some((joined, anchor)) = rank_state_joined_history(ctx) else {
        return Ok(f64::NAN);
    };
    if joined.is_empty() {
        return Ok(f64::NAN);
    }
    let dates: BTreeSet<NaiveDate> = joined.iter().map(|row| row.trade_date).collect();
    let positions: BTreeMap<NaiveDate, usize> = dates
        .iter()
        .copied()
        .enumerate()
        .map(|(position, date)| (date, position))
        .collect();
    let Some(anchor_position) = positions.get(&anchor).copied() else {
        return Ok(f64::NAN);
    };
    let returns = numpy_log_ratio(
        py,
        &joined.iter().map(|row| row.close_price).collect::<Vec<_>>(),
        &joined
            .iter()
            .map(|row| row.prev_close_price)
            .collect::<Vec<_>>(),
        false,
    )?;
    let capacities = numpy_log_ratio(
        py,
        &joined.iter().map(|row| row.amount).collect::<Vec<_>>(),
        &joined.iter().map(|row| row.remain_size).collect::<Vec<_>>(),
        false,
    )?;
    let mut by_date: BTreeMap<NaiveDate, Vec<usize>> = BTreeMap::new();
    for (index, row) in joined.iter().enumerate() {
        by_date.entry(row.trade_date).or_default().push(index);
    }
    let mut return_rank = vec![f64::NAN; joined.len()];
    let mut capacity_rank = vec![f64::NAN; joined.len()];
    for indices in by_date.values() {
        let daily_return: Vec<f64> = indices.iter().map(|index| returns[*index]).collect();
        let daily_capacity: Vec<f64> = indices.iter().map(|index| capacities[*index]).collect();
        for (index, value) in indices.iter().zip(pandas_average_pct_rank(&daily_return)) {
            return_rank[*index] = value;
        }
        for (index, value) in indices.iter().zip(pandas_average_pct_rank(&daily_capacity)) {
            capacity_rank[*index] = value;
        }
    }
    let first_position = anchor_position.saturating_sub(59);
    let path: Vec<(f64, f64)> = joined
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            (row.code == code
                && positions
                    .get(&row.trade_date)
                    .is_some_and(|position| *position >= first_position))
            .then_some((return_rank[index], capacity_rank[index]))
        })
        .collect();
    let target_terminal = joined
        .iter()
        .filter(|row| row.code == code)
        .map(|row| row.trade_date)
        .max();
    if target_terminal != Some(anchor) {
        return Ok(f64::NAN);
    }
    let Some((last_left, last_right)) = path.last().copied() else {
        return Ok(f64::NAN);
    };
    if !last_left.is_finite() || !last_right.is_finite() {
        return Ok(f64::NAN);
    }
    let pairs: Vec<(f64, f64)> = path
        .into_iter()
        .filter(|(left, right)| left.is_finite() && right.is_finite())
        .collect();
    if pairs.len() < 45 {
        return Ok(f64::NAN);
    }
    let left: Vec<f64> = pairs.iter().map(|(left, _)| *left).collect();
    let right: Vec<f64> = pairs.iter().map(|(_, right)| *right).collect();
    numpy_centered_correlation(py, &left, &right)
}

fn exact_ydpt_fall_beta(
    py: Python<'_>,
    ctx: &TypedFactorRankStateContext,
    code: &str,
) -> PyResult<f64> {
    let Some((joined, anchor)) = rank_state_joined_history(ctx) else {
        return Ok(f64::NAN);
    };
    let path: Vec<_> = joined.into_iter().filter(|row| row.code == code).collect();
    if path.last().map(|row| row.trade_date) != Some(anchor) || path.len() < 61 {
        return Ok(f64::NAN);
    }
    let tail = &path[path.len() - 61..];
    if tail.iter().any(|row| {
        !row.act_prev_close_price.is_finite()
            || !row.close_price.is_finite()
            || !row.current_yield.is_finite()
            || row.act_prev_close_price <= 1e-12
            || row.close_price <= 1e-12
    }) {
        return Ok(f64::NAN);
    }
    let returns = numpy_subtract(
        py,
        &numpy_log(
            py,
            &tail.iter().map(|row| row.close_price).collect::<Vec<_>>(),
        )?,
        &numpy_log(
            py,
            &tail
                .iter()
                .map(|row| row.act_prev_close_price)
                .collect::<Vec<_>>(),
        )?,
    )?;
    if returns.iter().any(|value| !value.is_finite()) {
        return Ok(f64::NAN);
    }
    let yield_change = numpy_adjacent_difference(
        py,
        &tail.iter().map(|row| row.current_yield).collect::<Vec<_>>(),
    )?;
    let mut state = Vec::new();
    let mut response = Vec::new();
    for (change, value) in yield_change
        .into_iter()
        .skip(1)
        .zip(returns.into_iter().skip(1))
    {
        if change < -1e-12 {
            state.push(change);
            response.push(value);
        }
    }
    numpy_ols_slope(py, &state, &response)
}

fn strict_cross_sources<'a>(
    ctx: &'a TypedFactorDailyCrossAssetContext,
) -> Option<(
    Vec<&'a TypedFactorCrossAssetPriceRow>,
    Vec<&'a TypedFactorCrossAssetBaseRow>,
    Vec<NaiveDate>,
    NaiveDate,
)> {
    let mut price: Vec<_> = ctx
        .price_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .collect();
    let mut base: Vec<_> = ctx
        .base_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .collect();
    let price_anchor = price.iter().map(|row| row.trade_date).max()?;
    let base_anchor = base.iter().map(|row| row.trade_date).max()?;
    if price_anchor != base_anchor {
        return None;
    }
    price.sort_by(|left, right| {
        (left.trade_date, left.code.as_str()).cmp(&(right.trade_date, right.code.as_str()))
    });
    base.sort_by(|left, right| {
        (left.trade_date, left.code.as_str()).cmp(&(right.trade_date, right.code.as_str()))
    });
    let sessions: Vec<NaiveDate> = price
        .iter()
        .map(|row| row.trade_date)
        .chain(base.iter().map(|row| row.trade_date))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Some((price, base, sessions, price_anchor))
}

fn sign3_cross(value: f64) -> Option<usize> {
    if !value.is_finite() {
        None
    } else if value < -1e-12 {
        Some(0)
    } else if value.abs() <= 1e-12 {
        Some(1)
    } else {
        Some(2)
    }
}

/// Final bsFST MI reduction using the exact NumPy primitive sequence from the
/// Python family.  State construction remains Rust-side and is discrete.
fn numpy_bsfst_mutual_information(
    py: Python<'_>,
    stock_return: &[f64],
    amount_change: &[f64],
) -> PyResult<f64> {
    if stock_return.len() != amount_change.len() || stock_return.is_empty() {
        return Ok(f64::NAN);
    }
    if sign3_cross(*stock_return.last().unwrap_or(&f64::NAN)).is_none()
        || sign3_cross(*amount_change.last().unwrap_or(&f64::NAN)).is_none()
    {
        return Ok(f64::NAN);
    }
    let mut counts = [[0.5_f64; 3]; 3];
    let mut observed = 0usize;
    for (stock, amount) in stock_return
        .iter()
        .copied()
        .zip(amount_change.iter().copied())
    {
        if let (Some(stock_state), Some(amount_state)) = (sign3_cross(stock), sign3_cross(amount)) {
            counts[stock_state][amount_state] += 1.0;
            observed += 1;
        }
    }
    if observed < 45 {
        return Ok(f64::NAN);
    }
    let numpy = py.import_bound("numpy")?;
    let counts = numpy.call_method1(
        "array",
        (vec![
            counts[0].to_vec(),
            counts[1].to_vec(),
            counts[2].to_vec(),
        ],),
    )?;
    let total = counts.call_method0("sum")?;
    let probability = counts.call_method1("__truediv__", (&total,))?;
    let row_kwargs = PyDict::new_bound(py);
    row_kwargs.set_item("axis", 1)?;
    row_kwargs.set_item("keepdims", true)?;
    let row_probability = probability.call_method("sum", (), Some(&row_kwargs))?;
    let column_kwargs = PyDict::new_bound(py);
    column_kwargs.set_item("axis", 0)?;
    column_kwargs.set_item("keepdims", true)?;
    let column_probability = probability.call_method("sum", (), Some(&column_kwargs))?;
    let denominator = row_probability.call_method1("__mul__", (&column_probability,))?;
    let ratio = probability.call_method1("__truediv__", (&denominator,))?;
    let logarithm = numpy.call_method1("log", (&ratio,))?;
    let product = probability.call_method1("__mul__", (&logarithm,))?;
    let information = numpy.call_method1("sum", (&product,))?;
    let normalizer = py.import_bound("math")?.call_method1("log", (3.0,))?;
    let value = information
        .call_method1("__truediv__", (&normalizer,))?
        .extract::<f64>()?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

fn exact_bsfst_mutual_information(
    py: Python<'_>,
    ctx: &TypedFactorDailyCrossAssetContext,
    code: &str,
) -> PyResult<f64> {
    if code.is_empty() {
        return Ok(f64::NAN);
    }
    let Some((price, base, sessions, anchor)) = strict_cross_sources(ctx) else {
        return Ok(f64::NAN);
    };
    let base_by_key: BTreeMap<(NaiveDate, String), &TypedFactorCrossAssetBaseRow> = base
        .iter()
        .map(|row| ((row.trade_date, row.code.clone()), *row))
        .collect();
    let mut target_by_date: BTreeMap<
        NaiveDate,
        (
            &TypedFactorCrossAssetPriceRow,
            &TypedFactorCrossAssetBaseRow,
        ),
    > = BTreeMap::new();
    for row in price.into_iter().filter(|row| row.code == code) {
        if let Some(base_row) = base_by_key.get(&(row.trade_date, row.code.clone())) {
            target_by_date.insert(row.trade_date, (row, *base_row));
        }
    }
    if target_by_date.keys().next_back().copied() != Some(anchor) {
        return Ok(f64::NAN);
    }
    let first = sessions.len().saturating_sub(60);
    let mut stock_close = Vec::with_capacity(sessions.len() - first);
    let mut stock_previous = Vec::with_capacity(sessions.len() - first);
    let mut amount = Vec::with_capacity(sessions.len() - first);
    for day in sessions.iter().skip(first) {
        if let Some((price_row, base_row)) = target_by_date.get(day) {
            stock_close.push(base_row.stk_close_price);
            stock_previous.push(base_row.stk_prev_close_price);
            amount.push(price_row.amount);
        } else {
            stock_close.push(f64::NAN);
            stock_previous.push(f64::NAN);
            amount.push(f64::NAN);
        }
    }
    let stock_return = numpy_log_ratio(py, &stock_close, &stock_previous, true)?;
    let log_amount = numpy_log(
        py,
        &amount
            .into_iter()
            .map(|value| {
                if value.is_finite() && value > 1e-12 {
                    value
                } else {
                    f64::NAN
                }
            })
            .collect::<Vec<_>>(),
    )?;
    let amount_change = numpy_adjacent_difference(py, &log_amount)?;
    numpy_bsfst_mutual_information(py, &stock_return, &amount_change)
}

fn exact_bssrc_rank_correlation(
    py: Python<'_>,
    ctx: &TypedFactorDailyCrossAssetContext,
    code: &str,
) -> PyResult<f64> {
    let Some((price, base, _sessions, anchor)) = strict_cross_sources(ctx) else {
        return Ok(f64::NAN);
    };

    let bond_return = numpy_log_ratio(
        py,
        &price.iter().map(|row| row.close_price).collect::<Vec<_>>(),
        &price
            .iter()
            .map(|row| row.prev_close_price)
            .collect::<Vec<_>>(),
        false,
    )?;
    let mut price_by_date: BTreeMap<NaiveDate, Vec<usize>> = BTreeMap::new();
    for (index, row) in price.iter().enumerate() {
        price_by_date.entry(row.trade_date).or_default().push(index);
    }
    let mut bond_rank = BTreeMap::new();
    for indices in price_by_date.values() {
        let values: Vec<f64> = indices.iter().map(|index| bond_return[*index]).collect();
        for (index, rank) in indices.iter().zip(pandas_average_pct_rank(&values)) {
            bond_rank.insert((price[*index].trade_date, price[*index].code.clone()), rank);
        }
    }

    let stock_rows: Vec<&TypedFactorCrossAssetBaseRow> = base
        .iter()
        .copied()
        .filter(|row| !row.stock_code.is_empty())
        .collect();
    let stock_return = numpy_log_ratio(
        py,
        &stock_rows
            .iter()
            .map(|row| row.stk_close_price)
            .collect::<Vec<_>>(),
        &stock_rows
            .iter()
            .map(|row| row.stk_prev_close_price)
            .collect::<Vec<_>>(),
        false,
    )?;
    let mut stock_by_key: BTreeMap<(NaiveDate, String), Vec<usize>> = BTreeMap::new();
    for (index, row) in stock_rows.iter().enumerate() {
        stock_by_key
            .entry((row.trade_date, row.stock_code.clone()))
            .or_default()
            .push(index);
    }
    let mut stock_values: BTreeMap<NaiveDate, Vec<(String, f64)>> = BTreeMap::new();
    for ((trade_date, stock_code), indices) in stock_by_key {
        let values: Vec<f64> = indices.iter().map(|index| stock_return[*index]).collect();
        let value = if values.iter().any(|value| !value.is_finite()) {
            f64::NAN
        } else if numpy_ptp(py, &values)? <= 1e-12 {
            values[0]
        } else {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor kernel inconsistent strict-prior underlying return for {stock_code} on {trade_date}"
            )));
        };
        stock_values
            .entry(trade_date)
            .or_default()
            .push((stock_code, value));
    }
    let mut stock_rank_by_underlying = BTreeMap::new();
    for (trade_date, values) in stock_values {
        let ranks =
            pandas_average_pct_rank(&values.iter().map(|(_, value)| *value).collect::<Vec<_>>());
        for ((stock_code, _), rank) in values.into_iter().zip(ranks) {
            stock_rank_by_underlying.insert((trade_date, stock_code), rank);
        }
    }
    let mut stock_rank_by_bond = BTreeMap::new();
    for row in base
        .iter()
        .copied()
        .filter(|row| !row.stock_code.is_empty())
    {
        if let Some(rank) = stock_rank_by_underlying.get(&(row.trade_date, row.stock_code.clone()))
        {
            stock_rank_by_bond.insert((row.trade_date, row.code.clone()), *rank);
        }
    }

    // Python's final bond/stock rank merge is inner.  An empty/unmapped
    // underlying therefore removes that bond/day before the terminal tail is
    // selected; it must not survive as a NaN physical row.
    if code.is_empty() {
        return Ok(f64::NAN);
    }
    let mut path = Vec::new();
    for row in price.iter().copied().filter(|row| row.code == code) {
        let key = (row.trade_date, row.code.clone());
        if let Some(stock_rank) = stock_rank_by_bond.get(&key) {
            path.push((
                row.trade_date,
                bond_rank.get(&key).copied().unwrap_or(f64::NAN),
                *stock_rank,
            ));
        }
    }
    if path.last().map(|(trade_date, _, _)| *trade_date) != Some(anchor) {
        return Ok(f64::NAN);
    }
    let start = path.len().saturating_sub(60);
    let recent = &path[start..];
    let Some((_, terminal_bond, terminal_stock)) = recent.last() else {
        return Ok(f64::NAN);
    };
    if !terminal_bond.is_finite() || !terminal_stock.is_finite() {
        return Ok(f64::NAN);
    }
    let pairs: Vec<(f64, f64)> = recent
        .iter()
        .filter_map(|(_, bond, stock)| {
            (bond.is_finite() && stock.is_finite()).then_some((*bond, *stock))
        })
        .collect();
    if pairs.len() < 45 {
        return Ok(f64::NAN);
    }
    let upper = pairs.iter().filter(|(_, stock)| *stock >= 0.75).count();
    let lower = pairs.iter().filter(|(_, stock)| *stock <= 0.25).count();
    if upper < 8 || lower < 8 {
        return Ok(f64::NAN);
    }
    let bond: Vec<f64> = pairs.iter().map(|(bond, _)| *bond).collect();
    let stock: Vec<f64> = pairs.iter().map(|(_, stock)| *stock).collect();
    numpy_centered_correlation(py, &bond, &stock)
}

fn exact_daily_kernel_value(
    py: Python<'_>,
    ctx: &TypedFactorDailyContext,
    optional_path_ctx: &TypedFactorDailyPathContext,
    strict_path_ctx: &TypedFactorDailyPathContext,
    tracking_ctx: &TypedFactorDailyTrackingContext,
    rank_state_ctx: &TypedFactorRankStateContext,
    cross_asset_ctx: &TypedFactorDailyCrossAssetContext,
    code: &str,
    spec: &KernelSpec,
) -> PyResult<f64> {
    match spec.factor.as_str() {
        "factor_mining_daily_capacity_rank_coupling_v1"
            if spec.signal == "prcn_return_capacity_rank_corr60" =>
        {
            exact_prcn_rank_correlation(py, rank_state_ctx, code)
        }
        "factor_mining_daily_asymmetric_state_transitions_v1"
            if spec.signal == "ydpt_yield_fall_return_beta60" =>
        {
            exact_ydpt_fall_beta(py, rank_state_ctx, code)
        }
        "factor_mining_daily_bond_stock_return_flow_information_v1"
            if spec.signal == "bsfst_stock_return_bond_flow_mutual_information60" =>
        {
            exact_bsfst_mutual_information(py, cross_asset_ctx, code)
        }
        "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1"
            if spec.signal == "bssrc_bond_stock_rank_correlation60" =>
        {
            exact_bssrc_rank_correlation(py, cross_asset_ctx, code)
        }
        "factor_mining_daily_catalog_v1" if spec.signal == "dret_drawup_drawdown_asym" => {
            dret_drawup_drawdown_asym(optional_path_ctx, code)
                .map_err(|error| PyErr::new::<PyRuntimeError, _>(error.to_string()))
        }
        "factor_mining_daily_ohlc_wick_path_asymmetry_v1" => match spec.signal.as_str() {
            "dohw_mean_wick_asymmetry60" => exact_dohw_mean_wick(py, strict_path_ctx, code),
            "dohw_intraday_sign_range_asymmetry60" => {
                exact_dohw_intraday_sign_range(py, strict_path_ctx, code)
            }
            _ => Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor daily path kernel has unknown OHLC signal={}",
                spec.signal
            ))),
        },
        "factor_mining_daily_contract_stock_v1"
            if spec.signal == "bstk_tail_cocrash_residual20" =>
        {
            exact_bstk_tail_cocrash(py, tracking_ctx, code)
        }
        "factor_mining_daily_return_liquidity_topology_v1"
            if spec.signal == "rjst_amount_joint_transition_entropy60" =>
        {
            let counts = rjst_amount_joint_transition_nonzero_counts(ctx, code)
                .map_err(|error| PyErr::new::<PyRuntimeError, _>(error.to_string()))?;
            match counts {
                Some(counts) => numpy_rjst_entropy(py, &counts),
                None => Ok(f64::NAN),
            }
        }
        "factor_mining_daily_liquidity_channel_composition_v1"
        | "factor_mining_daily_return_liquidity_topology_v1" => {
            compute_daily_information_signal(ctx, code, &spec.signal)
                .map_err(|error| PyErr::new::<PyRuntimeError, _>(error.to_string()))
        }
        _ => exact_p1_value(py, ctx, code, &spec.signal),
    }
}

/// Typed implementation shared by the ordinary Rust dispatcher and the
/// explicit parity-test entrypoint.  It remains an internal Rust function so
/// `compute_factor_frame` does not re-enter a second public Python API or
/// create a pandas merge path for any subset of outputs.
pub(crate) fn compute_typed_factor_frame_impl(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    specs_payload: &Bound<'_, PyAny>,
    stock_df: Option<&Bound<'_, PyAny>>,
    map_df: Option<&Bound<'_, PyAny>>,
    daily_data: Option<&Bound<'_, PyAny>>,
    _compute_params: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    // None of these kernels consume stock/map inputs. Keep the explicit
    // rejection so this internal contract cannot silently grow.
    if stock_df.is_some() || map_df.is_some() {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor kernel does not accept stock_df or map_df",
        ));
    }
    let specs = parse_specs(specs_payload)?;
    let has_qed = specs.iter().any(is_qed_spec);
    let has_lrd = specs.iter().any(is_lrd_spec);
    let has_intraday = has_qed || has_lrd;
    let has_daily = specs.iter().any(|spec| !is_intraday_spec(spec));

    // Python LRD calls `ensure_trade_time` before it resolves __build_day__.
    // Preserve that error precedence without pre-validating the rest of the
    // LRD book schema (those fields are checked after score-day resolution by
    // the Python reference).
    if has_lrd {
        require_panel_columns(py, panel_df, LRD_FACTOR, ["trade_time".to_string()])?;
    }

    // Daily kernel calls preserve the established backend score-date override.
    // QED/LRD deliberately do not: their Python references privilege the
    // panel's own __build_day__ attribute, including its invalid/conflicting
    // error behaviour.
    let daily_score_date = if has_daily {
        score_date_from_panel(py, panel_df, _compute_params)?
    } else {
        None
    };
    let intraday_score_date = if has_intraday {
        intraday_score_date_from_panel(py, panel_df)?
    } else {
        None
    };
    let intraday_ctx = build_intraday_context(py, panel_df, intraday_score_date, has_qed, has_lrd)?;

    let labelled_keys: BTreeSet<(String, String)> = daily_score_date
        .map(|score_date| output_keys(py, panel_df, score_date))
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .collect();
    let has_catalog_spec = specs
        .iter()
        .any(|spec| spec.factor == "factor_mining_daily_catalog_v1");
    let catalog_keys = if has_catalog_spec {
        daily_score_date
            .map(|score_date| catalog_output_keys(py, panel_df, score_date))
            .transpose()?
            .unwrap_or_default()
    } else {
        BTreeSet::new()
    };
    let qed_keys: BTreeSet<(String, String)> = if has_qed {
        intraday_score_date
            .map(|score_date| intraday_labelled_output_keys(py, panel_df, score_date))
            .transpose()?
            .unwrap_or_default()
            .into_iter()
            .collect()
    } else {
        BTreeSet::new()
    };
    let per_spec_keys: Vec<BTreeSet<(String, String)>> = specs
        .iter()
        .map(|spec| {
            if is_qed_spec(spec) {
                qed_keys.clone()
            } else if is_lrd_spec(spec) {
                intraday_ctx.lrd_keys.clone()
            } else if spec.factor == "factor_mining_daily_catalog_v1" {
                catalog_keys.clone()
            } else {
                labelled_keys.clone()
            }
        })
        .collect();
    let keys: Vec<(String, String)> = per_spec_keys
        .iter()
        .flat_map(|keys| keys.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    // LRD's strict schema has already been checked above.  Every other family
    // exits before daily-source validation when its output universe is empty.
    if keys.is_empty() {
        let columns = specs
            .iter()
            .map(|spec| (spec.output_col.clone(), Vec::new()))
            .collect::<Vec<_>>();
        return build_ordered_output(py, &keys, &columns);
    }

    let needs = source_needs(&specs)?;
    let context_score_date = daily_score_date
        .or(intraday_score_date)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).expect("constant date"));
    let daily_has_output_keys = specs
        .iter()
        .zip(per_spec_keys.iter())
        .any(|(spec, spec_keys)| !is_intraday_spec(spec) && !spec_keys.is_empty());
    let populate_daily = daily_score_date.is_some() && daily_has_output_keys;
    let mut optional_ctx = TypedFactorDailyContext::new(context_score_date);
    let mut strict_ctx = TypedFactorDailyContext::new(context_score_date);
    let mut optional_path_ctx = TypedFactorDailyPathContext::new(context_score_date);
    let mut strict_path_ctx = TypedFactorDailyPathContext::new(context_score_date);
    let mut tracking_ctx = TypedFactorDailyTrackingContext::new(context_score_date);
    let mut rank_state_ctx = TypedFactorRankStateContext::new(context_score_date);
    let mut cross_asset_ctx = TypedFactorDailyCrossAssetContext::new(context_score_date);
    if populate_daily {
        populate_daily_context(
            py,
            daily_data,
            &needs.optional,
            CanonicalContract::OptionalExchange,
            context_score_date,
            &mut optional_ctx,
        )?;
        populate_daily_context(
            py,
            daily_data,
            &needs.strict,
            CanonicalContract::StrictExchange,
            context_score_date,
            &mut strict_ctx,
        )?;
        // The path data shape is intentionally distinct from the legacy
        // P1/info context: it preserves adjusted prior closes/open prices and
        // the independent tracking calendar.
        populate_daily_path_context(
            py,
            daily_data,
            &needs.optional.price,
            CanonicalContract::OptionalExchange,
            context_score_date,
            &mut optional_path_ctx,
        )?;
        populate_daily_path_context(
            py,
            daily_data,
            &needs.strict.price,
            CanonicalContract::StrictExchange,
            context_score_date,
            &mut strict_path_ctx,
        )?;
        populate_daily_tracking_context(
            py,
            daily_data,
            &needs.strict,
            context_score_date,
            &mut tracking_ctx,
        )?;
        populate_rank_state_context(
            py,
            daily_data,
            &needs.rank_state,
            context_score_date,
            &mut rank_state_ctx,
        )?;
        populate_cross_asset_context(
            py,
            daily_data,
            &needs.cross_asset,
            context_score_date,
            &mut cross_asset_ctx,
        )?;
    }

    let mut columns = Vec::with_capacity(specs.len());
    for (spec, spec_keys) in specs.iter().zip(per_spec_keys.iter()) {
        let mut values = Vec::with_capacity(keys.len());
        for key @ (_, output_code) in &keys {
            if !spec_keys.contains(key) {
                values.push(f64::NAN);
                continue;
            }
            if is_qed_spec(spec) {
                let metrics = match intraday_ctx.qed_rows.get(key) {
                    Some(rows) => exact_qed_metrics(py, rows)?,
                    None => qed_nan_metrics(),
                };
                let value = match spec.signal.as_str() {
                    "qed_prior_quote_location_dispersion" => {
                        metrics.prior_quote_location_dispersion
                    }
                    "qed_prior_quote_tail_penetration" => metrics.prior_quote_tail_penetration,
                    "qed_prior_quote_lag2_agreement" => metrics.prior_quote_lag2_agreement,
                    _ => f64::NAN,
                };
                values.push(if value.is_finite() { value } else { f64::NAN });
                continue;
            }
            if is_lrd_spec(spec) {
                let value = match intraday_ctx.lrd_rows.get(key) {
                    Some(rows) => exact_lrd_value(py, rows)?,
                    None => f64::NAN,
                };
                values.push(if value.is_finite() { value } else { f64::NAN });
                continue;
            }
            if !populate_daily {
                values.push(f64::NAN);
                continue;
            }
            let canonical_contract = canonical_contract_for_factor(&spec.factor);
            let code = canonical_market_code(Some(output_code), None, canonical_contract)
                .unwrap_or_default();
            let ctx = if canonical_contract == CanonicalContract::OptionalExchange {
                &optional_ctx
            } else {
                &strict_ctx
            };
            let value = exact_daily_kernel_value(
                py,
                ctx,
                &optional_path_ctx,
                &strict_path_ctx,
                &tracking_ctx,
                &rank_state_ctx,
                &cross_asset_ctx,
                &code,
                spec,
            )?;
            values.push(if value.is_finite() { value } else { f64::NAN });
        }
        columns.push((spec.output_col.clone(), values));
    }
    build_ordered_output(py, &keys, &columns)
}

/// Explicit typed-kernel entrypoint retained for isolated Python/Rust parity
/// fixtures.  Production callers use ``compute_factor_frame`` only.
#[pyfunction]
#[pyo3(signature = (panel_df, specs_payload, stock_df=None, map_df=None, daily_data=None, _compute_params=None))]
pub fn compute_typed_factor_frame(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    specs_payload: &Bound<'_, PyAny>,
    stock_df: Option<&Bound<'_, PyAny>>,
    map_df: Option<&Bound<'_, PyAny>>,
    daily_data: Option<&Bound<'_, PyAny>>,
    _compute_params: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    compute_typed_factor_frame_impl(
        py,
        panel_df,
        specs_payload,
        stock_df,
        map_df,
        daily_data,
        _compute_params,
    )
}

#[cfg(test)]
mod tests {
    use super::{canonical_market_code, CanonicalContract};

    #[test]
    fn canonical_market_code_keeps_catalog_optional_exchange_contract() {
        assert_eq!(
            canonical_market_code(
                Some("110001.XSHG"),
                Some("XSHG"),
                CanonicalContract::OptionalExchange,
            ),
            Some("110001.XSHG.SH".to_string())
        );
        assert_eq!(
            canonical_market_code(
                Some("110001"),
                Some("UNKNOWN"),
                CanonicalContract::OptionalExchange,
            ),
            Some("110001".to_string())
        );
        assert_eq!(
            canonical_market_code(Some("110001"), None, CanonicalContract::OptionalExchange,),
            Some("110001".to_string())
        );
        assert_eq!(
            canonical_market_code(
                Some("110001.0"),
                Some("SHSE"),
                CanonicalContract::OptionalExchange,
            ),
            Some("110001.SH".to_string())
        );
    }

    #[test]
    fn canonical_market_code_matches_strict_daily_alias_contract() {
        for (raw, expected) in [
            ("110001.XSHG", "110001.SH"),
            ("110001.SHSE", "110001.SH"),
            ("110001.XSHE", "110001.SZ"),
            ("110001.SZSE", "110001.SZ"),
            ("110001.BSE", "110001.BJ"),
            ("110001.BJSE", "110001.BJ"),
        ] {
            assert_eq!(
                canonical_market_code(
                    Some(raw),
                    Some("UNKNOWN"),
                    CanonicalContract::StrictExchange,
                ),
                Some(expected.to_string())
            );
        }
        assert_eq!(
            canonical_market_code(
                Some("110001"),
                Some("XSHG"),
                CanonicalContract::StrictExchange,
            ),
            Some("110001.SH".to_string())
        );
        assert_eq!(
            canonical_market_code(
                Some("110001.0"),
                Some("SHSE"),
                CanonicalContract::StrictExchange,
            ),
            Some("110001.SH".to_string())
        );
        assert_eq!(
            canonical_market_code(
                Some("110001"),
                Some("UNKNOWN"),
                CanonicalContract::StrictExchange,
            ),
            Some(String::new())
        );
        assert_eq!(
            canonical_market_code(Some("110001"), None, CanonicalContract::StrictExchange,),
            Some(String::new())
        );
        assert_eq!(
            canonical_market_code(Some(""), Some("XSHG"), CanonicalContract::StrictExchange,),
            Some(String::new())
        );
        assert_eq!(
            canonical_market_code(Some("nan"), Some("XSHG"), CanonicalContract::StrictExchange,),
            Some(String::new())
        );
        assert_eq!(
            canonical_market_code(None, Some("XSHG"), CanonicalContract::StrictExchange),
            None
        );
    }
}
