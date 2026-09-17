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
    lrd_cross_side_reprice_symmetry, quote_execution_typed_factor,
    rdm_joint_reprice_depth_retention, session_label, BookRow, QedTypedFactorMetrics, QuoteRow,
    QED_AT_QUOTE_TOL,
};
use crate::typed_factor_r88_daily::{
    prepare_r88_daily_values, TypedFactorR88DailyBaseRow, TypedFactorR88DailyContext,
    TypedFactorR88DailyPriceRow, TypedFactorR88DailySignal, TypedFactorR88DailyTwapRow,
};
use crate::typed_factor_r88_intraday::{
    is_r88_intraday_signal, r88_intraday_metrics, R88IntradayRow,
};
use crate::typed_factor_r88_remaining::{
    compute_r88_remaining_signal, prepare_r88_remaining_values, strict_1429_visible,
    R88RemainingBondStockMapRow, R88RemainingDailyBaseRow, R88RemainingDailyPriceRow,
    R88RemainingDailyTwapRow, R88RemainingIntradayRow, TypedFactorR88RemainingContext,
    TypedFactorR88RemainingSignal,
};
use chrono::NaiveDate;
use numpy::IntoPyArray;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::time::{Duration, Instant};

const SOURCE_BASE: &str = "market_cbond.daily_base";
const SOURCE_PRICE: &str = "market_cbond.daily_price";
const SOURCE_TWAP: &str = "market_cbond.daily_twap";
const QED_FACTOR: &str = "factor_mining_quote_execution_dynamics_v1";
const LRD_FACTOR: &str = "factor_mining_orderbook_repricing_v1";
const R88_INTRADAY_EXPANSION_FACTOR: &str = "factor_mining_intraday_expansion_v1";
const R88_INTRADAY_EXECDISC_FACTOR: &str = "factor_mining_intraday_execution_discreteness_v1";
const R88_INTRADAY_CATALOG_FACTOR: &str = "factor_mining_intraday_catalog_v1";
const R88_REMAINING_HYBRID_FACTOR: &str = "factor_mining_hybrid_catalog_v1";
const R88_REMAINING_JOINT_FACTOR: &str = "factor_mining_intraday_joint_state_v1";
const R88_REMAINING_TRANSMISSION_FACTOR: &str = "factor_mining_intraday_transmission_response_v1";
const R88_REMAINING_UNDERLYING_FACTOR: &str = "factor_mining_underlying_cohort_distribution_v1";
const R88_REMAINING_STATE_GATED_FACTOR: &str =
    "factor_mining_intraday_state_gated_microstructure_v1";
const R88_REMAINING_QUOTE_GEOMETRY_FACTOR: &str = "factor_mining_quote_geometry_microprice_v1";
const R88_REMAINING_STRUCTURAL_FACTOR: &str = "factor_mining_structural_neighborhood_v1";
const R88_REMAINING_CSN_FACTOR: &str =
    "factor_mining_cross_sectional_microstructure_neighborhood_v1";
const R88_DAILY_ASYMMETRIC_BETA_FACTOR: &str = "factor_mining_daily_asymmetric_equity_beta_v1";
const R88_DAILY_COPULA_FACTOR: &str = "factor_mining_daily_bond_stock_copula_tail_dependence_v1";
const R88_DAILY_SEASONING_FACTOR: &str = "factor_mining_daily_observable_seasoning_v1";
const R88_DAILY_RANK_COUPLING_FACTOR: &str = "factor_mining_daily_relative_rank_flow_coupling_v2";
const R88_DAILY_RANK_TAIL_FACTOR: &str = "factor_mining_daily_relative_rank_tail_contradiction_v1";
const R88_DAILY_TWAP_FACTOR: &str = "factor_mining_daily_twap_microstructure_v1";
const QED_SIGNALS: &[&str] = &[
    "qed_prior_quote_location_dispersion",
    "qed_prior_quote_tail_penetration",
    "qed_prior_quote_lag2_agreement",
];
const LRD_SIGNALS: &[&str] = &[
    "lrd_cross_side_reprice_symmetry",
    "rdm_joint_reprice_depth_retention",
];
const QED_REQUIRED_PANEL_COLUMNS: &[&str] = &[
    "trade_time",
    "last",
    "ask_price1",
    "bid_price1",
    "num_trades",
];
const R88_CATALOG_REQUIRED_PANEL_COLUMNS: &[&str] = &[
    "trade_time",
    "open",
    "last",
    "volume",
    "amount",
    "num_trades",
    "pre_close",
    "high_limited",
    "low_limited",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
    "ask_price2",
    "bid_price2",
    "ask_volume2",
    "bid_volume2",
    "ask_price3",
    "bid_price3",
    "ask_volume3",
    "bid_volume3",
    "ask_price4",
    "bid_price4",
    "ask_volume4",
    "bid_volume4",
    "ask_price5",
    "bid_price5",
    "ask_volume5",
    "bid_volume5",
];
const R88_JOINT_REQUIRED_PANEL_COLUMNS: &[&str] = &[
    "trade_time",
    "pre_close",
    "open",
    "last",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
];
const R88_STATE_GATED_REQUIRED_PANEL_COLUMNS: &[&str] = &[
    "trade_time",
    "last",
    "amount",
    "num_trades",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
];
const R88_QUOTE_GEOMETRY_REQUIRED_PANEL_COLUMNS: &[&str] = &[
    "trade_time",
    "last",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
    "ask_price2",
    "bid_price2",
    "ask_volume2",
    "bid_volume2",
    "ask_price3",
    "bid_price3",
    "ask_volume3",
    "bid_volume3",
    "ask_price4",
    "bid_price4",
    "ask_volume4",
    "bid_volume4",
    "ask_price5",
    "bid_price5",
    "ask_volume5",
    "bid_volume5",
];
const R88_CSN_REQUIRED_PANEL_COLUMNS: &[&str] = &[
    "trade_time",
    "num_trades",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
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
            | R88_INTRADAY_EXPANSION_FACTOR
            | R88_INTRADAY_EXECDISC_FACTOR
            | R88_INTRADAY_CATALOG_FACTOR
            | R88_REMAINING_HYBRID_FACTOR
            | R88_REMAINING_JOINT_FACTOR
            | R88_REMAINING_TRANSMISSION_FACTOR
            | R88_REMAINING_UNDERLYING_FACTOR
            | R88_REMAINING_STATE_GATED_FACTOR
            | R88_REMAINING_QUOTE_GEOMETRY_FACTOR
            | R88_REMAINING_STRUCTURAL_FACTOR
            | R88_REMAINING_CSN_FACTOR
            | R88_DAILY_ASYMMETRIC_BETA_FACTOR
            | R88_DAILY_COPULA_FACTOR
            | R88_DAILY_SEASONING_FACTOR
            | R88_DAILY_RANK_COUPLING_FACTOR
            | R88_DAILY_RANK_TAIL_FACTOR
            | R88_DAILY_TWAP_FACTOR
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
    family: String,
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

/// Exact physical R88 paths keyed by the labelled score-day output key.  The
/// row itself carries no label: only the dispatcher proves physical date and
/// the strict 14:29 cutoff before it reaches the formula module.
#[derive(Default)]
struct TypedFactorR88IntradayContext {
    rows: BTreeMap<(String, String), Vec<R88IntradayRow>>,
}

type R88OutputKey = (String, String);
type PreparedR88DirectValues = BTreeMap<(String, String, String), f64>;
type PreparedRdmValues = BTreeMap<(String, String), f64>;

/// Opt-in, research-only wall-clock markers for the R88 typed dispatcher.
///
/// The live profile has no R88 specs, and the timer is constructed only when
/// the explicit environment switch is set to `1`.  It deliberately observes
/// elapsed time only: no data, formula, output, or dispatch behaviour changes.
const R88_PHASE_TIMING_ENV: &str = "CBOND_ON_R88_PHASE_TIMING";
const R88_OUTPUT_SPEC_TIMING_MIN: Duration = Duration::from_millis(1);

struct R88PhaseTiming {
    started_at: Instant,
}

impl R88PhaseTiming {
    fn maybe_start(has_r88_spec: bool) -> Option<Self> {
        if has_r88_spec && matches!(std::env::var(R88_PHASE_TIMING_ENV).as_deref(), Ok("1")) {
            Some(Self {
                started_at: Instant::now(),
            })
        } else {
            None
        }
    }

    fn phase_started(&self) -> Instant {
        Instant::now()
    }

    fn record_phase(&self, phase: &str, phase_started_at: Instant) {
        eprintln!(
            "[CBOND_ON_R88_PHASE_TIMING] phase={phase} elapsed_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1_000.0
        );
    }

    /// During an R88-profile dispatch, emit the expensive output specs
    /// regardless of their factor key.  Some R38 contracts deliberately
    /// reuse generic factor keys, so filtering only on `is_r88_factor` would
    /// hide the very work this diagnostic is intended to find.
    fn record_output_spec(&self, spec: &KernelSpec, spec_started_at: Instant) {
        let elapsed = spec_started_at.elapsed();
        if elapsed <= R88_OUTPUT_SPEC_TIMING_MIN {
            return;
        }
        eprintln!(
            "[CBOND_ON_R88_PHASE_TIMING] phase=output_spec factor={} signal={} output_col={} elapsed_ms={:.3}",
            spec.factor,
            spec.signal,
            spec.output_col,
            elapsed.as_secs_f64() * 1_000.0
        );
    }

    fn finish(&self) {
        eprintln!(
            "[CBOND_ON_R88_PHASE_TIMING] phase=total_dispatch elapsed_ms={:.3}",
            self.started_at.elapsed().as_secs_f64() * 1_000.0
        );
    }
}

/// R88 direct intraday specs share the same per-code physical path.  Compute
/// the nine-metric bundle once per `(dt, code)` instead of calling the bundle
/// once for each requested output.  The catalogue path keeps its own
/// continuous-session subset exactly as the former ordered dispatch did.
fn prepare_r88_direct_intraday_values(
    specs: &[KernelSpec],
    per_spec_keys: &[BTreeSet<R88OutputKey>],
    ctx: &TypedFactorR88IntradayContext,
) -> PreparedR88DirectValues {
    let mut prepared = BTreeMap::new();
    let standard_specs: Vec<_> = specs
        .iter()
        .zip(per_spec_keys.iter())
        .filter(|(spec, _)| {
            is_r88_direct_intraday_spec(spec) && spec.factor != R88_INTRADAY_CATALOG_FACTOR
        })
        .collect();
    let catalogue_specs: Vec<_> = specs
        .iter()
        .zip(per_spec_keys.iter())
        .filter(|(spec, _)| {
            is_r88_direct_intraday_spec(spec) && spec.factor == R88_INTRADAY_CATALOG_FACTOR
        })
        .collect();

    let standard_keys: BTreeSet<_> = standard_specs
        .iter()
        .flat_map(|(_, keys)| keys.iter().cloned())
        .collect();
    for key in standard_keys {
        let metrics = ctx.rows.get(&key).map(|rows| r88_intraday_metrics(rows));
        for (spec, spec_keys) in &standard_specs {
            if spec_keys.contains(&key) {
                let value = metrics
                    .as_ref()
                    .map(|metrics| metrics.value(&spec.signal))
                    .unwrap_or(f64::NAN);
                prepared.insert(
                    (spec.output_col.clone(), key.0.clone(), key.1.clone()),
                    value,
                );
            }
        }
    }

    let catalogue_keys: BTreeSet<_> = catalogue_specs
        .iter()
        .flat_map(|(_, keys)| keys.iter().cloned())
        .collect();
    for key in catalogue_keys {
        let metrics = ctx.rows.get(&key).map(|rows| {
            let continuous = rows
                .iter()
                .copied()
                .filter(|row| session_label(row.time_ns).is_some())
                .collect::<Vec<_>>();
            r88_intraday_metrics(&continuous)
        });
        for (spec, spec_keys) in &catalogue_specs {
            if spec_keys.contains(&key) {
                let value = metrics
                    .as_ref()
                    .map(|metrics| metrics.value(&spec.signal))
                    .unwrap_or(f64::NAN);
                prepared.insert(
                    (spec.output_col.clone(), key.0.clone(), key.1.clone()),
                    value,
                );
            }
        }
    }
    prepared
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
        let family = match params.get_item("family")? {
            Some(value) if !value.is_none() => value.extract::<String>()?.trim().to_string(),
            _ => String::new(),
        };
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
            family,
            output_col,
        });
    }
    Ok(out)
}

/// The R88 research profile multiplexes many formula families behind factor
/// keys.  Keep an exact triple gate at the Rust boundary so a typo or an
/// otherwise-valid signal cannot silently select a different formula family.
/// This gate is intentionally profile-neutral: it validates only the source
/// factor key and the concrete `params.signal`/`params.family` pair.
fn r88_expected_family(factor: &str, signal: &str) -> Option<&'static str> {
    match (factor, signal) {
        (R88_DAILY_ASYMMETRIC_BETA_FACTOR, "bsab_upside_beta60")
        | (R88_DAILY_ASYMMETRIC_BETA_FACTOR, "bsab_downside_beta60") => {
            Some("prior_asymmetric_equity_beta")
        }
        (R88_DAILY_COPULA_FACTOR, "bsct_upper_tail_dependence60") => {
            Some("prior_bond_stock_copula_tail_dependence")
        }
        (R88_DAILY_SEASONING_FACTOR, "osa_terminal_amount_streak60") => {
            Some("prior_observable_market_seasoning")
        }
        (R88_DAILY_RANK_COUPLING_FACTOR, "drrc_return_trade_size_rank_spearman60") => {
            Some("prior_relative_return_flow_rank_coupling")
        }
        (R88_DAILY_RANK_TAIL_FACTOR, "drrq_return_amount_opposite_tail_excess60") => {
            Some("prior_relative_return_flow_tail_contradiction")
        }
        (R88_DAILY_TWAP_FACTOR, "dtwm_session_afternoon_late_log_slope") => {
            Some("prior_session_rotation_microstructure")
        }
        (R88_INTRADAY_EXPANSION_FACTOR, "exp_rotation_segment_return_dispersion") => {
            Some("clock_time_rotation")
        }
        (R88_INTRADAY_EXPANSION_FACTOR, "exp_exec_amount_concentration_impact") => {
            Some("execution_price_dispersion")
        }
        (R88_INTRADAY_EXPANSION_FACTOR, "exp_noise_median_mean_abs_return_ratio")
        | (R88_INTRADAY_EXPANSION_FACTOR, "exp_noise_variance_ratio_2")
        | (R88_INTRADAY_EXPANSION_FACTOR, "exp_noise_variance_ratio_5") => {
            Some("multiscale_noise_variance")
        }
        (R88_INTRADAY_EXPANSION_FACTOR, "exp_stick_quote_update_rate") => {
            Some("quote_trade_stickiness")
        }
        (R88_INTRADAY_EXECDISC_FACTOR, "execdisc_direction_reversal_rate") => {
            Some("execution_nonzero_direction_topology")
        }
        (R88_INTRADAY_EXECDISC_FACTOR, "execdisc_step_multiplicity_entropy") => {
            Some("execution_step_multiplicity_geometry")
        }
        (R88_INTRADAY_CATALOG_FACTOR, "book_quote_dislocation") => {
            Some("intraday_quote_resilience")
        }
        (R88_REMAINING_HYBRID_FACTOR, "hybrid_current_range_vs_hist_twap_curve")
        | (R88_REMAINING_HYBRID_FACTOR, "hybrid_current_flow_vs_hist_overnight_response") => {
            Some("hybrid_execution_curve")
        }
        (R88_REMAINING_JOINT_FACTOR, "joint_tail_range_coexpansion")
        | (R88_REMAINING_JOINT_FACTOR, "joint_tail_signed_cojump")
        | (R88_REMAINING_JOINT_FACTOR, "joint_tail_terminal_location_coshock") => {
            Some("joint_tail_cojump_containment")
        }
        (R88_REMAINING_TRANSMISSION_FACTOR, "itr_stock_shock_same_bin_directional_agreement") => {
            Some("intraday_stock_shock_directional_response")
        }
        (R88_REMAINING_UNDERLYING_FACTOR, "ucd_peer_stock_return_dispersion1") => {
            Some("underlying_state_distribution")
        }
        (R88_REMAINING_STATE_GATED_FACTOR, "isgm_stockvol_trade_quote_clock_center_gap") => {
            Some("stockvol_gated_trade_quote_clock_decoupling")
        }
        (R88_REMAINING_QUOTE_GEOMETRY_FACTOR, "qgeo_micro_last_next_return_sign_alignment") => {
            Some("microprice_last_execution_alignment")
        }
        (R88_REMAINING_STRUCTURAL_FACTOR, "sng_peer_return_dispersion1") => {
            Some("structural_neighborhood_geometry")
        }
        (R88_REMAINING_CSN_FACTOR, "csn_pql_churn_neighbor_gap") => {
            Some("csn_passive_queue_local_dislocation")
        }
        _ => None,
    }
}

fn is_r88_factor(factor: &str) -> bool {
    matches!(
        factor,
        R88_DAILY_ASYMMETRIC_BETA_FACTOR
            | R88_DAILY_COPULA_FACTOR
            | R88_DAILY_SEASONING_FACTOR
            | R88_DAILY_RANK_COUPLING_FACTOR
            | R88_DAILY_RANK_TAIL_FACTOR
            | R88_DAILY_TWAP_FACTOR
            | R88_INTRADAY_EXPANSION_FACTOR
            | R88_INTRADAY_EXECDISC_FACTOR
            | R88_INTRADAY_CATALOG_FACTOR
            | R88_REMAINING_HYBRID_FACTOR
            | R88_REMAINING_JOINT_FACTOR
            | R88_REMAINING_TRANSMISSION_FACTOR
            | R88_REMAINING_UNDERLYING_FACTOR
            | R88_REMAINING_STATE_GATED_FACTOR
            | R88_REMAINING_QUOTE_GEOMETRY_FACTOR
            | R88_REMAINING_STRUCTURAL_FACTOR
            | R88_REMAINING_CSN_FACTOR
    )
}

fn is_r88_daily_spec(spec: &KernelSpec) -> bool {
    matches!(
        spec.factor.as_str(),
        R88_DAILY_ASYMMETRIC_BETA_FACTOR
            | R88_DAILY_COPULA_FACTOR
            | R88_DAILY_SEASONING_FACTOR
            | R88_DAILY_RANK_COUPLING_FACTOR
            | R88_DAILY_RANK_TAIL_FACTOR
            | R88_DAILY_TWAP_FACTOR
    ) && TypedFactorR88DailySignal::parse(&spec.signal).is_some()
        && r88_expected_family(&spec.factor, &spec.signal).is_some()
}

fn is_r88_direct_intraday_spec(spec: &KernelSpec) -> bool {
    matches!(
        spec.factor.as_str(),
        R88_INTRADAY_EXPANSION_FACTOR | R88_INTRADAY_EXECDISC_FACTOR | R88_INTRADAY_CATALOG_FACTOR
    ) && is_r88_intraday_signal(&spec.signal)
        && r88_expected_family(&spec.factor, &spec.signal).is_some()
}

fn is_r88_remaining_spec(spec: &KernelSpec) -> bool {
    matches!(
        spec.factor.as_str(),
        R88_REMAINING_HYBRID_FACTOR
            | R88_REMAINING_JOINT_FACTOR
            | R88_REMAINING_TRANSMISSION_FACTOR
            | R88_REMAINING_UNDERLYING_FACTOR
            | R88_REMAINING_STATE_GATED_FACTOR
            | R88_REMAINING_QUOTE_GEOMETRY_FACTOR
            | R88_REMAINING_STRUCTURAL_FACTOR
            | R88_REMAINING_CSN_FACTOR
    ) && TypedFactorR88RemainingSignal::parse(&spec.signal).is_some()
        && r88_expected_family(&spec.factor, &spec.signal).is_some()
}

fn validate_r88_specs(specs: &[KernelSpec]) -> PyResult<()> {
    for spec in specs {
        if !is_r88_factor(&spec.factor) {
            continue;
        }
        let Some(expected_family) = r88_expected_family(&spec.factor, &spec.signal) else {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor R88 has unsupported exact factor/signal pair: factor={} signal={}",
                spec.factor, spec.signal
            )));
        };
        if spec.family != expected_family {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor R88 factor/signal requires params.family={expected_family:?}, got {:?}: factor={} signal={}",
                spec.family, spec.factor, spec.signal
            )));
        }
    }
    Ok(())
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
            "factor_mining_daily_catalog_v1",
            "base_duration_stockvol_interaction",
        ),
        (
            "factor_mining_daily_catalog_v1",
            "base_trigger_progress_ratio",
        ),
        (
            "factor_mining_daily_catalog_v1",
            "base_trigger_revision_gap",
        ),
        (
            "factor_mining_daily_catalog_v1",
            "base_stockvol_per_moneyness",
        ),
        (
            "factor_mining_daily_bond_stock_return_flow_information_v1",
            "bsfst_stock_return_bond_flow_mutual_information60",
        ),
        (
            "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1",
            "bssrc_upper_rank_tail_alignment60",
        ),
        (
            "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1",
            "bssrc_bond_stock_rank_correlation60",
        ),
        (
            "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1",
            "bssrc_lower_rank_tail_alignment60",
        ),
        (
            "factor_mining_daily_contract_stock_v1",
            "bstk_tail_cocrash_residual20",
        ),
        (
            "factor_mining_daily_contract_stock_v1",
            "rating_current_ordinal",
        ),
        (
            "factor_mining_daily_expansion_v1",
            "dliq_volume_return_corr20",
        ),
        ("factor_mining_daily_expansion_v1", "dret_momentum_5"),
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
            "factor_mining_daily_liquidity_channel_composition_v1",
            "lcc_size_frequency_coupling60",
        ),
        (
            "factor_mining_orderbook_repricing_v1",
            "lrd_cross_side_reprice_symmetry",
        ),
        (
            "factor_mining_orderbook_repricing_v1",
            "rdm_joint_reprice_depth_retention",
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
            "factor_mining_daily_return_liquidity_topology_v1",
            "rlmi_return_trade_size_sign_mutual_information60",
        ),
        (
            "factor_mining_daily_asymmetric_state_transitions_v1",
            "ydpt_yield_fall_return_beta60",
        ),
        (R88_DAILY_ASYMMETRIC_BETA_FACTOR, "bsab_upside_beta60"),
        (R88_DAILY_ASYMMETRIC_BETA_FACTOR, "bsab_downside_beta60"),
        (R88_DAILY_COPULA_FACTOR, "bsct_upper_tail_dependence60"),
        (R88_DAILY_SEASONING_FACTOR, "osa_terminal_amount_streak60"),
        (
            R88_DAILY_RANK_COUPLING_FACTOR,
            "drrc_return_trade_size_rank_spearman60",
        ),
        (
            R88_DAILY_RANK_TAIL_FACTOR,
            "drrq_return_amount_opposite_tail_excess60",
        ),
        (
            R88_DAILY_TWAP_FACTOR,
            "dtwm_session_afternoon_late_log_slope",
        ),
        (
            R88_INTRADAY_EXPANSION_FACTOR,
            "exp_rotation_segment_return_dispersion",
        ),
        (
            R88_INTRADAY_EXPANSION_FACTOR,
            "exp_exec_amount_concentration_impact",
        ),
        (
            R88_INTRADAY_EXPANSION_FACTOR,
            "exp_noise_median_mean_abs_return_ratio",
        ),
        (R88_INTRADAY_EXPANSION_FACTOR, "exp_noise_variance_ratio_2"),
        (R88_INTRADAY_EXPANSION_FACTOR, "exp_noise_variance_ratio_5"),
        (R88_INTRADAY_EXPANSION_FACTOR, "exp_stick_quote_update_rate"),
        (
            R88_INTRADAY_EXECDISC_FACTOR,
            "execdisc_direction_reversal_rate",
        ),
        (
            R88_INTRADAY_EXECDISC_FACTOR,
            "execdisc_step_multiplicity_entropy",
        ),
        (R88_INTRADAY_CATALOG_FACTOR, "book_quote_dislocation"),
        (
            R88_REMAINING_HYBRID_FACTOR,
            "hybrid_current_range_vs_hist_twap_curve",
        ),
        (
            R88_REMAINING_HYBRID_FACTOR,
            "hybrid_current_flow_vs_hist_overnight_response",
        ),
        (R88_REMAINING_JOINT_FACTOR, "joint_tail_range_coexpansion"),
        (R88_REMAINING_JOINT_FACTOR, "joint_tail_signed_cojump"),
        (
            R88_REMAINING_JOINT_FACTOR,
            "joint_tail_terminal_location_coshock",
        ),
        (
            R88_REMAINING_TRANSMISSION_FACTOR,
            "itr_stock_shock_same_bin_directional_agreement",
        ),
        (
            R88_REMAINING_UNDERLYING_FACTOR,
            "ucd_peer_stock_return_dispersion1",
        ),
        (
            R88_REMAINING_STATE_GATED_FACTOR,
            "isgm_stockvol_trade_quote_clock_center_gap",
        ),
        (
            R88_REMAINING_QUOTE_GEOMETRY_FACTOR,
            "qgeo_micro_last_next_return_sign_alignment",
        ),
        (
            R88_REMAINING_STRUCTURAL_FACTOR,
            "sng_peer_return_dispersion1",
        ),
        (R88_REMAINING_CSN_FACTOR, "csn_pql_churn_neighbor_gap"),
    ]
}

fn is_qed_spec(spec: &KernelSpec) -> bool {
    spec.factor == QED_FACTOR && QED_SIGNALS.contains(&spec.signal.as_str())
}

fn is_lrd_spec(spec: &KernelSpec) -> bool {
    spec.factor == LRD_FACTOR && LRD_SIGNALS.contains(&spec.signal.as_str())
}

fn is_intraday_spec(spec: &KernelSpec) -> bool {
    is_qed_spec(spec)
        || is_lrd_spec(spec)
        || is_r88_direct_intraday_spec(spec)
        || is_r88_remaining_intraday_spec(spec)
}

fn is_r88_remaining_intraday_spec(spec: &KernelSpec) -> bool {
    matches!(
        spec.factor.as_str(),
        R88_REMAINING_HYBRID_FACTOR
            | R88_REMAINING_JOINT_FACTOR
            | R88_REMAINING_TRANSMISSION_FACTOR
            | R88_REMAINING_STATE_GATED_FACTOR
            | R88_REMAINING_QUOTE_GEOMETRY_FACTOR
            | R88_REMAINING_CSN_FACTOR
    ) && is_r88_remaining_spec(spec)
}

fn r88_remaining_uses_daily_data(spec: &KernelSpec) -> bool {
    matches!(
        spec.factor.as_str(),
        R88_REMAINING_HYBRID_FACTOR
            | R88_REMAINING_JOINT_FACTOR
            | R88_REMAINING_UNDERLYING_FACTOR
            | R88_REMAINING_STATE_GATED_FACTOR
            | R88_REMAINING_STRUCTURAL_FACTOR
    ) && is_r88_remaining_spec(spec)
}

fn source_needs(specs: &[KernelSpec]) -> PyResult<KernelNeeds> {
    let mut needs = KernelNeeds::default();
    for spec in specs {
        // R88 has independently versioned source contexts below.  Do not
        // route a research signal through a legacy P1 source parser merely
        // because both families consume daily price rows.
        if is_r88_factor(&spec.factor) {
            continue;
        }
        match (spec.factor.as_str(), spec.signal.as_str()) {
            (QED_FACTOR, signal) if QED_SIGNALS.contains(&signal) => {
                // QED consumes only the physical score-day panel.  Missing
                // fields are deliberately handled as an all-NaN output, not
                // as a daily-source or parser exception.
            }
            (LRD_FACTOR, signal) if LRD_SIGNALS.contains(&signal) => {
                // LRD likewise has no daily source.  Its strict panel schema
                // is validated by the typed intraday parser below.
            }
            ("factor_mining_daily_catalog_v1", "base_debt_premium_floor_gap")
            | ("factor_mining_daily_catalog_v1", "base_duration_stockvol_interaction")
            | ("factor_mining_daily_catalog_v1", "base_trigger_progress_ratio")
            | ("factor_mining_daily_catalog_v1", "base_trigger_revision_gap")
            | ("factor_mining_daily_catalog_v1", "base_stockvol_per_moneyness")
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
            ("factor_mining_daily_expansion_v1", "dret_volatility_20")
            | ("factor_mining_daily_expansion_v1", "dret_momentum_5") => {
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
            ("factor_mining_daily_contract_stock_v1", "rating_current_ordinal") => {
                // The contract-stock family always declares `daily_price` as
                // its independent T-1 anchor and validates both family base
                // fields even though the selected output itself is a rating
                // ordinal only.
                add_columns(&mut needs.strict.price, &["close_price"], true, true);
                add_columns(&mut needs.strict.base, &["rating", "ytm"], true, true);
            }
            (
                "factor_mining_daily_liquidity_channel_composition_v1",
                "lcc_amount_trade_size_information60",
            )
            | (
                "factor_mining_daily_liquidity_channel_composition_v1",
                "lcc_volume_deal_information60",
            )
            | (
                "factor_mining_daily_liquidity_channel_composition_v1",
                "lcc_size_frequency_coupling60",
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
            )
            | (
                "factor_mining_daily_return_liquidity_topology_v1",
                "rlmi_return_trade_size_sign_mutual_information60",
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
                "bssrc_upper_rank_tail_alignment60",
            )
            | (
                "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1",
                "bssrc_bond_stock_rank_correlation60",
            )
            | (
                "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1",
                "bssrc_lower_rank_tail_alignment60",
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

fn r88_daily_source_needs(specs: &[KernelSpec]) -> ContractNeeds {
    let mut needs = ContractNeeds::default();
    for spec in specs.iter().filter(|spec| is_r88_daily_spec(spec)) {
        match spec.factor.as_str() {
            R88_DAILY_ASYMMETRIC_BETA_FACTOR | R88_DAILY_COPULA_FACTOR => {
                add_columns(
                    &mut needs.price,
                    &["prev_close_price", "close_price"],
                    true,
                    true,
                );
                add_columns(
                    &mut needs.base,
                    &["stk_prev_close_price", "stk_close_price"],
                    true,
                    true,
                );
            }
            R88_DAILY_SEASONING_FACTOR => {
                add_columns(&mut needs.price, &["close_price", "amount"], true, true)
            }
            R88_DAILY_RANK_COUPLING_FACTOR => add_columns(
                &mut needs.price,
                &["prev_close_price", "close_price", "amount", "deal"],
                true,
                true,
            ),
            R88_DAILY_RANK_TAIL_FACTOR => add_columns(
                &mut needs.price,
                &["prev_close_price", "close_price", "amount"],
                true,
                true,
            ),
            R88_DAILY_TWAP_FACTOR => {
                add_columns(&mut needs.price, &["close_price"], true, true);
                add_columns(
                    &mut needs.twap,
                    &["twap_1300_1330", "twap_1400_1430"],
                    true,
                    true,
                );
            }
            _ => unreachable!("validated R88 daily spec has an unmapped factor"),
        }
    }
    needs
}

fn r88_remaining_source_needs(specs: &[KernelSpec]) -> ContractNeeds {
    let mut needs = ContractNeeds::default();
    for spec in specs.iter().filter(|spec| is_r88_remaining_spec(spec)) {
        match spec.factor.as_str() {
            R88_REMAINING_HYBRID_FACTOR
                if spec.signal == "hybrid_current_range_vs_hist_twap_curve" =>
            {
                add_columns(
                    &mut needs.twap,
                    &[
                        "twap_0930_0935",
                        "twap_0935_1000",
                        "twap_1300_1330",
                        "twap_1400_1430",
                        "twap_1430_1442",
                    ],
                    true,
                    true,
                );
            }
            R88_REMAINING_HYBRID_FACTOR
                if spec.signal == "hybrid_current_flow_vs_hist_overnight_response" =>
            {
                add_columns(&mut needs.base, &["cb_amount"], true, true);
                add_columns(
                    &mut needs.twap,
                    &["twap_0930_0935", "twap_1430_1442"],
                    true,
                    true,
                );
            }
            R88_REMAINING_JOINT_FACTOR => {
                add_columns(&mut needs.price, &["close_price"], true, true);
                add_columns(&mut needs.base, &["stock_code"], true, true);
            }
            R88_REMAINING_UNDERLYING_FACTOR => {
                add_columns(
                    &mut needs.price,
                    &["prev_close_price", "close_price"],
                    true,
                    true,
                );
                add_columns(
                    &mut needs.base,
                    &[
                        "stock_code",
                        "stock_close_price",
                        "stock_volatility",
                        "stk_amount",
                    ],
                    true,
                    true,
                );
            }
            R88_REMAINING_STATE_GATED_FACTOR => {
                add_columns(&mut needs.price, &["close_price"], true, true);
                add_columns(&mut needs.base, &["stock_volatility"], true, true);
            }
            R88_REMAINING_STRUCTURAL_FACTOR => {
                add_columns(
                    &mut needs.price,
                    &["prev_close_price", "close_price"],
                    true,
                    true,
                );
                add_columns(
                    &mut needs.base,
                    &[
                        "year_to_mat",
                        "duration",
                        "bond_prem_ratio",
                        "ytm",
                        "remain_size",
                    ],
                    true,
                    true,
                );
            }
            // The other remaining kernels are strictly intraday and use no
            // daily material.
            R88_REMAINING_TRANSMISSION_FACTOR
            | R88_REMAINING_QUOTE_GEOMETRY_FACTOR
            | R88_REMAINING_CSN_FACTOR => {}
            _ => unreachable!("validated R88 remaining spec has an unmapped factor"),
        }
    }
    needs
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
    let duration = numeric_or_nan(py, &df, "duration", n)?;
    let stock_volatility = numeric_or_nan(py, &df, "stock_volatility", n)?;
    let cb_close_price = numeric_or_nan(py, &df, "cb_close_price", n)?;
    let conv_value = numeric_or_nan(py, &df, "conv_value", n)?;
    let trigger_cum_days = numeric_or_nan(py, &df, "trigger_cum_days", n)?;
    let trigger_reach_days = numeric_or_nan(py, &df, "trigger_reach_days", n)?;
    let trigger_cum_days_revise = numeric_or_nan(py, &df, "trigger_cum_days_revise", n)?;
    let rating = nullable_string_values_or_missing(py, &df, "rating", n)?;
    if [
        codes.len(),
        exchanges.len(),
        debt.len(),
        pure.len(),
        bond.len(),
        redemption.len(),
        duration.len(),
        stock_volatility.len(),
        cb_close_price.len(),
        conv_value.len(),
        trigger_cum_days.len(),
        trigger_reach_days.len(),
        trigger_cum_days_revise.len(),
        rating.len(),
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
                duration: duration[i],
                stock_volatility: stock_volatility[i],
                cb_close_price: cb_close_price[i],
                conv_value: conv_value[i],
                trigger_cum_days: trigger_cum_days[i],
                trigger_reach_days: trigger_reach_days[i],
                trigger_cum_days_revise: trigger_cum_days_revise[i],
                rating: rating[i].clone().unwrap_or_default(),
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

/// Populate the seven dedicated R88 daily formulas without reusing the older
/// P1 source structs.  These records preserve raw code/exchange pairs for the
/// formula module's own strict canonicalisation and duplicate checks.
fn populate_r88_daily_context(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    needs: &ContractNeeds,
    score_date: NaiveDate,
    ctx: &mut TypedFactorR88DailyContext,
) -> PyResult<()> {
    if !needs.price.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_PRICE, needs.price.missing_fails)? else {
            return Ok(());
        };
        require_columns(py, &df, SOURCE_PRICE, &needs.price)?;
        if !is_empty(&df)? {
            let dates = normalized_date_strings(py, &df, "trade_date")?;
            let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
            let n = dates.len();
            let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
            let prev = numeric_or_nan(py, &df, "prev_close_price", n)?;
            let close = numeric_or_nan(py, &df, "close_price", n)?;
            let amount = numeric_or_nan(py, &df, "amount", n)?;
            let deal = numeric_or_nan(py, &df, "deal", n)?;
            if [
                codes.len(),
                exchanges.len(),
                prev.len(),
                close.len(),
                amount.len(),
                deal.len(),
            ]
            .iter()
            .any(|length| *length != n)
            {
                return Err(PyErr::new::<PyValueError, _>(
                    "typed_factor R88 daily_price length mismatch",
                ));
            }
            for index in 0..n {
                let Some(trade_date) = parse_date(&dates[index]) else {
                    continue;
                };
                if trade_date >= score_date {
                    continue;
                }
                ctx.price_rows.push(TypedFactorR88DailyPriceRow {
                    trade_date,
                    code: codes[index].clone().unwrap_or_default(),
                    exchange_code: exchanges[index].clone().unwrap_or_default(),
                    prev_close_price: prev[index],
                    close_price: close[index],
                    amount: amount[index],
                    deal: deal[index],
                });
            }
        }
    }
    if !needs.base.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_BASE, needs.base.missing_fails)? else {
            return Ok(());
        };
        require_columns(py, &df, SOURCE_BASE, &needs.base)?;
        if !is_empty(&df)? {
            let dates = normalized_date_strings(py, &df, "trade_date")?;
            let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
            let n = dates.len();
            let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
            let stock_prev = numeric_or_nan(py, &df, "stk_prev_close_price", n)?;
            let stock_close = numeric_or_nan(py, &df, "stk_close_price", n)?;
            if [
                codes.len(),
                exchanges.len(),
                stock_prev.len(),
                stock_close.len(),
            ]
            .iter()
            .any(|length| *length != n)
            {
                return Err(PyErr::new::<PyValueError, _>(
                    "typed_factor R88 daily_base length mismatch",
                ));
            }
            for index in 0..n {
                let Some(trade_date) = parse_date(&dates[index]) else {
                    continue;
                };
                if trade_date >= score_date {
                    continue;
                }
                ctx.base_rows.push(TypedFactorR88DailyBaseRow {
                    trade_date,
                    code: codes[index].clone().unwrap_or_default(),
                    exchange_code: exchanges[index].clone().unwrap_or_default(),
                    stk_prev_close_price: stock_prev[index],
                    stk_close_price: stock_close[index],
                });
            }
        }
    }
    if !needs.twap.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_TWAP, needs.twap.missing_fails)? else {
            return Ok(());
        };
        require_columns(py, &df, SOURCE_TWAP, &needs.twap)?;
        if !is_empty(&df)? {
            let dates = normalized_date_strings(py, &df, "trade_date")?;
            let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
            let n = dates.len();
            let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
            let early = numeric_or_nan(py, &df, "twap_1300_1330", n)?;
            let late = numeric_or_nan(py, &df, "twap_1400_1430", n)?;
            if [codes.len(), exchanges.len(), early.len(), late.len()]
                .iter()
                .any(|length| *length != n)
            {
                return Err(PyErr::new::<PyValueError, _>(
                    "typed_factor R88 daily_twap length mismatch",
                ));
            }
            for index in 0..n {
                let Some(trade_date) = parse_date(&dates[index]) else {
                    continue;
                };
                if trade_date >= score_date {
                    continue;
                }
                ctx.twap_rows.push(TypedFactorR88DailyTwapRow {
                    trade_date,
                    code: codes[index].clone().unwrap_or_default(),
                    exchange_code: exchanges[index].clone().unwrap_or_default(),
                    twap_1300_1330: early[index],
                    twap_1400_1430: late[index],
                });
            }
        }
    }
    Ok(())
}

/// Populate the eleven R88 remaining formulas.  The narrow context is kept
/// separate from legacy typed rows because these formulas combine explicitly
/// strict daily anchors, physical intraday rows, and (for ITR) a dated map.
fn populate_r88_remaining_daily_context(
    py: Python<'_>,
    daily_data: Option<&Bound<'_, PyAny>>,
    needs: &ContractNeeds,
    score_date: NaiveDate,
    ctx: &mut TypedFactorR88RemainingContext,
) -> PyResult<()> {
    if !needs.price.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_PRICE, needs.price.missing_fails)? else {
            return Ok(());
        };
        require_columns(py, &df, SOURCE_PRICE, &needs.price)?;
        if !is_empty(&df)? {
            let dates = normalized_date_strings(py, &df, "trade_date")?;
            let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
            let n = dates.len();
            let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
            let prev = numeric_or_nan(py, &df, "prev_close_price", n)?;
            let close = numeric_or_nan(py, &df, "close_price", n)?;
            let amount = numeric_or_nan(py, &df, "amount", n)?;
            if [
                codes.len(),
                exchanges.len(),
                prev.len(),
                close.len(),
                amount.len(),
            ]
            .iter()
            .any(|length| *length != n)
            {
                return Err(PyErr::new::<PyValueError, _>(
                    "typed_factor R88 remaining daily_price length mismatch",
                ));
            }
            for index in 0..n {
                let Some(trade_date) = parse_date(&dates[index]) else {
                    continue;
                };
                if trade_date >= score_date {
                    continue;
                }
                ctx.daily_price_rows.push(R88RemainingDailyPriceRow {
                    trade_date,
                    code: codes[index].clone().unwrap_or_default(),
                    exchange_code: exchanges[index].clone().unwrap_or_default(),
                    prev_close_price: prev[index],
                    close_price: close[index],
                    amount: amount[index],
                });
            }
        }
    }
    if !needs.base.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_BASE, needs.base.missing_fails)? else {
            return Ok(());
        };
        require_columns(py, &df, SOURCE_BASE, &needs.base)?;
        if !is_empty(&df)? {
            let dates = normalized_date_strings(py, &df, "trade_date")?;
            let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
            let n = dates.len();
            let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
            let stock_code = nullable_string_values_or_missing(py, &df, "stock_code", n)?;
            let cb_amount = numeric_or_nan(py, &df, "cb_amount", n)?;
            let stock_close = numeric_or_nan(py, &df, "stock_close_price", n)?;
            let stock_volatility = numeric_or_nan(py, &df, "stock_volatility", n)?;
            let stock_amount = numeric_or_nan(py, &df, "stk_amount", n)?;
            let year_to_mat = numeric_or_nan(py, &df, "year_to_mat", n)?;
            let duration = numeric_or_nan(py, &df, "duration", n)?;
            let bond_premium = numeric_or_nan(py, &df, "bond_prem_ratio", n)?;
            let ytm = numeric_or_nan(py, &df, "ytm", n)?;
            let remain_size = numeric_or_nan(py, &df, "remain_size", n)?;
            if [
                codes.len(),
                exchanges.len(),
                stock_code.len(),
                cb_amount.len(),
                stock_close.len(),
                stock_volatility.len(),
                stock_amount.len(),
                year_to_mat.len(),
                duration.len(),
                bond_premium.len(),
                ytm.len(),
                remain_size.len(),
            ]
            .iter()
            .any(|length| *length != n)
            {
                return Err(PyErr::new::<PyValueError, _>(
                    "typed_factor R88 remaining daily_base length mismatch",
                ));
            }
            for index in 0..n {
                let Some(trade_date) = parse_date(&dates[index]) else {
                    continue;
                };
                if trade_date >= score_date {
                    continue;
                }
                ctx.daily_base_rows.push(R88RemainingDailyBaseRow {
                    trade_date,
                    code: codes[index].clone().unwrap_or_default(),
                    exchange_code: exchanges[index].clone().unwrap_or_default(),
                    cb_amount: cb_amount[index],
                    stock_code: stock_code[index].clone().unwrap_or_default(),
                    stock_close_price: stock_close[index],
                    stock_volatility: stock_volatility[index],
                    stk_amount: stock_amount[index],
                    year_to_mat: year_to_mat[index],
                    duration: duration[index],
                    bond_prem_ratio: bond_premium[index],
                    ytm: ytm[index],
                    remain_size: remain_size[index],
                });
            }
        }
    }
    if !needs.twap.columns.is_empty() {
        let Some(df) = source_frame(daily_data, SOURCE_TWAP, needs.twap.missing_fails)? else {
            return Ok(());
        };
        require_columns(py, &df, SOURCE_TWAP, &needs.twap)?;
        if !is_empty(&df)? {
            let dates = normalized_date_strings(py, &df, "trade_date")?;
            let codes = nullable_string_values_or_missing(py, &df, "code", 0)?;
            let n = dates.len();
            let exchanges = nullable_string_values_or_missing(py, &df, "exchange_code", n)?;
            let open = numeric_or_nan(py, &df, "twap_0930_0935", n)?;
            let morning = numeric_or_nan(py, &df, "twap_0935_1000", n)?;
            let early = numeric_or_nan(py, &df, "twap_1300_1330", n)?;
            let late = numeric_or_nan(py, &df, "twap_1400_1430", n)?;
            let close = numeric_or_nan(py, &df, "twap_1430_1442", n)?;
            if [
                codes.len(),
                exchanges.len(),
                open.len(),
                morning.len(),
                early.len(),
                late.len(),
                close.len(),
            ]
            .iter()
            .any(|length| *length != n)
            {
                return Err(PyErr::new::<PyValueError, _>(
                    "typed_factor R88 remaining daily_twap length mismatch",
                ));
            }
            for index in 0..n {
                let Some(trade_date) = parse_date(&dates[index]) else {
                    continue;
                };
                if trade_date >= score_date {
                    continue;
                }
                ctx.daily_twap_rows.push(R88RemainingDailyTwapRow {
                    trade_date,
                    code: codes[index].clone().unwrap_or_default(),
                    exchange_code: exchanges[index].clone().unwrap_or_default(),
                    twap_0930_0935: open[index],
                    twap_0935_1000: morning[index],
                    twap_1300_1330: early[index],
                    twap_1400_1430: late[index],
                    twap_1430_1442: close[index],
                });
            }
        }
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

fn r88_panel_columns_for_direct_specs(specs: &[KernelSpec]) -> Vec<String> {
    let mut columns = BTreeSet::new();
    for spec in specs
        .iter()
        .filter(|spec| is_r88_direct_intraday_spec(spec))
    {
        match spec.factor.as_str() {
            R88_INTRADAY_EXPANSION_FACTOR => match spec.signal.as_str() {
                "exp_rotation_segment_return_dispersion"
                | "exp_noise_median_mean_abs_return_ratio"
                | "exp_noise_variance_ratio_2"
                | "exp_noise_variance_ratio_5" => {
                    columns.extend(["trade_time", "last"].into_iter().map(str::to_string));
                }
                "exp_exec_amount_concentration_impact" => {
                    columns.extend(
                        ["trade_time", "last", "volume", "amount", "num_trades"]
                            .into_iter()
                            .map(str::to_string),
                    );
                }
                "exp_stick_quote_update_rate" => {
                    columns.extend(
                        [
                            "trade_time",
                            "last",
                            "ask_price1",
                            "bid_price1",
                            "ask_volume1",
                            "bid_volume1",
                        ]
                        .into_iter()
                        .map(str::to_string),
                    );
                }
                _ => unreachable!("validated R88 direct expansion signal"),
            },
            R88_INTRADAY_EXECDISC_FACTOR => {
                columns.extend(
                    ["trade_time", "last", "num_trades"]
                        .into_iter()
                        .map(str::to_string),
                );
            }
            R88_INTRADAY_CATALOG_FACTOR => {
                columns.extend(
                    R88_CATALOG_REQUIRED_PANEL_COLUMNS
                        .iter()
                        .copied()
                        .map(str::to_string),
                );
            }
            _ => unreachable!("validated R88 direct factor"),
        }
    }
    columns.into_iter().collect()
}

fn r88_panel_columns_for_remaining_specs(specs: &[KernelSpec], stock_side: bool) -> Vec<String> {
    let mut columns = BTreeSet::new();
    for spec in specs.iter().filter(|spec| is_r88_remaining_spec(spec)) {
        let needs_stock = matches!(
            spec.factor.as_str(),
            R88_REMAINING_JOINT_FACTOR | R88_REMAINING_TRANSMISSION_FACTOR
        );
        // Every selected intraday R88 remaining family reads the bond panel;
        // only joint/ITR additionally read the stock panel.
        if stock_side && !needs_stock {
            continue;
        }
        match spec.factor.as_str() {
            R88_REMAINING_HYBRID_FACTOR => {
                columns.extend(
                    ["trade_time", "last", "volume", "amount", "num_trades"]
                        .into_iter()
                        .map(str::to_string),
                );
            }
            R88_REMAINING_JOINT_FACTOR => {
                columns.extend(
                    R88_JOINT_REQUIRED_PANEL_COLUMNS
                        .iter()
                        .copied()
                        .map(str::to_string),
                );
            }
            R88_REMAINING_TRANSMISSION_FACTOR => {
                columns.extend(["trade_time", "last"].into_iter().map(str::to_string));
            }
            R88_REMAINING_STATE_GATED_FACTOR => {
                columns.extend(
                    R88_STATE_GATED_REQUIRED_PANEL_COLUMNS
                        .iter()
                        .copied()
                        .map(str::to_string),
                );
            }
            R88_REMAINING_QUOTE_GEOMETRY_FACTOR => {
                columns.extend(
                    R88_QUOTE_GEOMETRY_REQUIRED_PANEL_COLUMNS
                        .iter()
                        .copied()
                        .map(str::to_string),
                );
            }
            R88_REMAINING_CSN_FACTOR => {
                columns.extend(
                    R88_CSN_REQUIRED_PANEL_COLUMNS
                        .iter()
                        .copied()
                        .map(str::to_string),
                );
            }
            R88_REMAINING_UNDERLYING_FACTOR | R88_REMAINING_STRUCTURAL_FACTOR => {}
            _ => unreachable!("validated R88 remaining factor"),
        }
    }
    columns.into_iter().collect()
}

fn r88_remaining_requires_stock(specs: &[KernelSpec]) -> bool {
    specs.iter().any(|spec| {
        is_r88_remaining_spec(spec)
            && matches!(
                spec.factor.as_str(),
                R88_REMAINING_JOINT_FACTOR | R88_REMAINING_TRANSMISSION_FACTOR
            )
    })
}

fn r88_remaining_requires_map(specs: &[KernelSpec]) -> bool {
    specs
        .iter()
        .any(|spec| is_r88_remaining_spec(spec) && spec.factor == R88_REMAINING_TRANSMISSION_FACTOR)
}

/// Used by the outer one-route dispatcher to preserve legacy behavior: raw
/// stock/map frames are forwarded into typed precompute only when one exact
/// R88 remaining instance needs them.  Ordinary typed factors continue to
/// receive the historical `None` context and cannot acquire a new route.
pub(crate) fn r88_typed_spec_requires_stock_context(factor: &str, signal: Option<&str>) -> bool {
    matches!(
        (factor, signal.unwrap_or("").trim()),
        (R88_REMAINING_JOINT_FACTOR, "joint_tail_range_coexpansion")
            | (R88_REMAINING_JOINT_FACTOR, "joint_tail_signed_cojump")
            | (
                R88_REMAINING_JOINT_FACTOR,
                "joint_tail_terminal_location_coshock"
            )
            | (
                R88_REMAINING_TRANSMISSION_FACTOR,
                "itr_stock_shock_same_bin_directional_agreement",
            )
    )
}

pub(crate) fn r88_typed_spec_requires_map_context(factor: &str, signal: Option<&str>) -> bool {
    matches!(
        (factor, signal.unwrap_or("").trim()),
        (
            R88_REMAINING_TRANSMISSION_FACTOR,
            "itr_stock_shock_same_bin_directional_agreement",
        )
    )
}

/// The regular typed intraday helper is deliberately continuous-session-only
/// for QED/LRD.  R88 has both continuous and all-visible families, so retain
/// only physical score-day proof plus the common strict 14:29 cutoff here and
/// let each pure R88 formula apply its own session contract.
fn r88_physical_rows(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    score_date: NaiveDate,
) -> PyResult<Vec<PhysicalIntradayRow>> {
    let labelled_dates = normalized_date_strings(py, panel_df, "dt")?;
    let length = labelled_dates.len();
    let (physical_dates, clocks, physical_score_matches) =
        intraday_time_parts(py, panel_df, length)?;
    let labelled_score_matches = if super::has_col(py, panel_df, "__label_matches_score_day__")? {
        Some(
            super::col_to_i64_vec(py, panel_df, "__label_matches_score_day__")?
                .into_iter()
                .map(|value| value != 0)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    if labelled_score_matches
        .as_ref()
        .is_some_and(|values| values.len() != length)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor R88 labelled score-day length mismatch",
        ));
    }
    let mut rows = Vec::new();
    for index in 0..length {
        let labelled_match = parse_date(&labelled_dates[index]) == Some(score_date)
            && labelled_score_matches
                .as_ref()
                .map(|values| values[index])
                .unwrap_or(true);
        let physical_match = parse_date(&physical_dates[index]) == Some(score_date)
            && physical_score_matches
                .as_ref()
                .map(|values| values[index])
                .unwrap_or(true);
        if labelled_match && physical_match && strict_1429_visible(clocks[index]) {
            rows.push(PhysicalIntradayRow {
                index,
                clock_ns: clocks[index],
            });
        }
    }
    Ok(rows)
}

struct R88PanelVectors {
    raw_dt: Vec<String>,
    codes: Vec<String>,
    seq: Vec<i64>,
    last: Vec<f64>,
    volume: Vec<f64>,
    amount: Vec<f64>,
    num_trades: Vec<f64>,
    ask_price: Vec<Vec<f64>>,
    bid_price: Vec<Vec<f64>>,
    ask_volume: Vec<Vec<f64>>,
    bid_volume: Vec<Vec<f64>>,
}

fn r88_panel_vectors(py: Python<'_>, panel_df: &Bound<'_, PyAny>) -> PyResult<R88PanelVectors> {
    let (raw_dt, codes, seq) = intraday_panel_keys(py, panel_df)?;
    let n = raw_dt.len();
    let last = numeric_or_nan(py, panel_df, "last", n)?;
    let volume = numeric_or_nan(py, panel_df, "volume", n)?;
    let amount = numeric_or_nan(py, panel_df, "amount", n)?;
    let num_trades = numeric_or_nan(py, panel_df, "num_trades", n)?;
    let mut ask_price = Vec::with_capacity(5);
    let mut bid_price = Vec::with_capacity(5);
    let mut ask_volume = Vec::with_capacity(5);
    let mut bid_volume = Vec::with_capacity(5);
    for level in 1..=5 {
        ask_price.push(numeric_or_nan(
            py,
            panel_df,
            &format!("ask_price{level}"),
            n,
        )?);
        bid_price.push(numeric_or_nan(
            py,
            panel_df,
            &format!("bid_price{level}"),
            n,
        )?);
        ask_volume.push(numeric_or_nan(
            py,
            panel_df,
            &format!("ask_volume{level}"),
            n,
        )?);
        bid_volume.push(numeric_or_nan(
            py,
            panel_df,
            &format!("bid_volume{level}"),
            n,
        )?);
    }
    if [
        codes.len(),
        seq.len(),
        last.len(),
        volume.len(),
        amount.len(),
        num_trades.len(),
    ]
    .iter()
    .any(|length| *length != n)
        || ask_price
            .iter()
            .chain(bid_price.iter())
            .chain(ask_volume.iter())
            .chain(bid_volume.iter())
            .any(|values| values.len() != n)
    {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor R88 panel value length mismatch",
        ));
    }
    Ok(R88PanelVectors {
        raw_dt,
        codes,
        seq,
        last,
        volume,
        amount,
        num_trades,
        ask_price,
        bid_price,
        ask_volume,
        bid_volume,
    })
}

fn append_r88_direct_intraday_rows(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    score_date: NaiveDate,
    ctx: &mut TypedFactorR88IntradayContext,
) -> PyResult<()> {
    let physical = r88_physical_rows(py, panel_df, score_date)?;
    let values = r88_panel_vectors(py, panel_df)?;
    for row in physical {
        let index = row.index;
        ctx.rows
            .entry((values.raw_dt[index].clone(), values.codes[index].clone()))
            .or_default()
            .push(R88IntradayRow {
                time_ns: row.clock_ns,
                seq: values.seq[index],
                last: values.last[index],
                volume: values.volume[index],
                amount: values.amount[index],
                num_trades: values.num_trades[index],
                ask_price1: values.ask_price[0][index],
                bid_price1: values.bid_price[0][index],
                ask_volume1: values.ask_volume[0][index],
                bid_volume1: values.bid_volume[0][index],
            });
    }
    Ok(())
}

fn append_r88_remaining_intraday_rows(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    score_date: NaiveDate,
    ctx: &mut TypedFactorR88RemainingContext,
) -> PyResult<()> {
    let physical = r88_physical_rows(py, panel_df, score_date)?;
    let values = r88_panel_vectors(py, panel_df)?;
    for row in physical {
        let index = row.index;
        let mut ask_price = [f64::NAN; 5];
        let mut bid_price = [f64::NAN; 5];
        let mut ask_volume = [f64::NAN; 5];
        let mut bid_volume = [f64::NAN; 5];
        for level in 0..5 {
            ask_price[level] = values.ask_price[level][index];
            bid_price[level] = values.bid_price[level][index];
            ask_volume[level] = values.ask_volume[level][index];
            bid_volume[level] = values.bid_volume[level][index];
        }
        ctx.intraday_rows.push(R88RemainingIntradayRow {
            trade_date: score_date,
            code: values.codes[index].clone(),
            exchange_code: String::new(),
            time_ns: row.clock_ns,
            seq: values.seq[index],
            last: values.last[index],
            volume: values.volume[index],
            amount: values.amount[index],
            num_trades: values.num_trades[index],
            ask_price,
            bid_price,
            ask_volume,
            bid_volume,
        });
    }
    Ok(())
}

fn populate_r88_remaining_map_context(
    py: Python<'_>,
    map_df: Option<&Bound<'_, PyAny>>,
    score_date: NaiveDate,
    ctx: &mut TypedFactorR88RemainingContext,
) -> PyResult<()> {
    let Some(df) = map_df else {
        return Err(PyErr::new::<PyKeyError, _>(
            "typed_factor R88 ITR requires bond_stock_map",
        ));
    };
    for column in ["code", "stock_code"] {
        if !super::has_col(py, df, column)? {
            return Err(PyErr::new::<PyKeyError, _>(format!(
                "typed_factor R88 bond_stock_map missing required column: {column}"
            )));
        }
    }
    let date_column = if super::has_col(py, df, "as_of_date")? {
        "as_of_date"
    } else if super::has_col(py, df, "trade_date")? {
        "trade_date"
    } else {
        return Err(PyErr::new::<PyKeyError, _>(
            "typed_factor R88 bond_stock_map requires as_of_date or trade_date",
        ));
    };
    let codes = nullable_string_values_or_missing(py, df, "code", 0)?;
    let stocks = nullable_string_values_or_missing(py, df, "stock_code", codes.len())?;
    let dates = normalized_date_strings(py, df, date_column)?;
    if stocks.len() != codes.len() || dates.len() != codes.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor R88 bond_stock_map length mismatch",
        ));
    }
    for index in 0..codes.len() {
        let Some(as_of_date) = parse_date(&dates[index]) else {
            continue;
        };
        if as_of_date > score_date {
            continue;
        }
        ctx.bond_stock_map.push(R88RemainingBondStockMapRow {
            code: codes[index].clone().unwrap_or_default(),
            stock_code: stocks[index].clone().unwrap_or_default(),
            as_of_date,
        });
    }
    Ok(())
}

fn build_ordered_output(
    py: Python<'_>,
    keys: Vec<(String, String)>,
    values: Vec<(String, Vec<f64>)>,
    r88_phase_timing: Option<&R88PhaseTiming>,
) -> PyResult<PyObject> {
    // Validate before moving the numerical Vecs into NumPy.  Apart from
    // avoiding a partially-built Python object on error, this preserves the
    // former first-mismatched-column failure contract.
    for (column, column_values) in &values {
        if column_values.len() != keys.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "typed_factor kernel output length mismatch for {column}"
            )));
        }
    }

    let dt_code_phase_started = r88_phase_timing.map(|timing| timing.phase_started());
    let data = PyDict::new_bound(py);
    let mut dts = Vec::with_capacity(keys.len());
    let mut codes = Vec::with_capacity(keys.len());
    for (dt, code) in keys {
        dts.push(dt);
        codes.push(code);
    }
    data.set_item("dt", dts)?;
    data.set_item("code", codes)?;
    if let (Some(timing), Some(phase_started_at)) = (r88_phase_timing, dt_code_phase_started) {
        timing.record_phase("output_dt_code", phase_started_at);
    }

    let numeric_columns_phase_started = r88_phase_timing.map(|timing| timing.phase_started());
    for (column, column_values) in values {
        // `IntoPyArray` transfers the Vec allocation to NumPy.  The previous
        // PyO3 conversion created a Python list first, causing one Python
        // object per scalar before pandas could make its float64 column.
        // A NumPy-owned f64 buffer keeps column order, dtype, NaN payloads,
        // and values while avoiding that element-wise bridge.
        data.set_item(column, column_values.into_pyarray_bound(py))?;
    }
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing, numeric_columns_phase_started)
    {
        timing.record_phase("output_numeric_columns", phase_started_at);
    }

    let dataframe_phase_started = r88_phase_timing.map(|timing| timing.phase_started());
    let pandas = py.import_bound("pandas")?;
    let kwargs = PyDict::new_bound(py);
    // The NumPy arrays are solely owned by this output path, so avoiding a
    // second pandas-side copy does not introduce an observable alias for
    // callers.  Their Python base object retains the transferred Rust buffer.
    kwargs.set_item("copy", false)?;
    let output = pandas
        .call_method("DataFrame", (data,), Some(&kwargs))?
        .into_py(py);
    if let (Some(timing), Some(phase_started_at)) = (r88_phase_timing, dataframe_phase_started) {
        timing.record_phase("output_dataframe", phase_started_at);
    }
    Ok(output)
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

/// NumPy-finalized `rdm_joint_reprice_depth_retention`.  The typed parser and
/// LRD direction classifier own source/session/PIT handling; only the final
/// average follows NumPy's reduction path to preserve the reference's float
/// contract.
#[cfg(test)]
fn exact_rdm_joint_depth_retention(py: Python<'_>, rows: &[BookRow]) -> PyResult<f64> {
    let _ = rdm_joint_reprice_depth_retention(rows);
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
    let mut retention = Vec::new();
    for (index, pair) in rows.windows(2).enumerate() {
        if bid_direction[index] == 0.0 || ask_direction[index] == 0.0 {
            continue;
        }
        let previous = pair[0]
            .0
            .bid_volume
            .iter()
            .chain(pair[0].0.ask_volume.iter())
            .sum::<f64>();
        let current = pair[1]
            .0
            .bid_volume
            .iter()
            .chain(pair[1].0.ask_volume.iter())
            .sum::<f64>();
        if !previous.is_finite() || !current.is_finite() || previous <= 1e-12 || current < 0.0 {
            return Ok(f64::NAN);
        }
        let value = previous.min(current) / previous;
        if !value.is_finite() {
            return Ok(f64::NAN);
        }
        retention.push(value);
    }
    if retention.len() < 3 {
        return Ok(f64::NAN);
    }
    let value = numpy_mean(py, &retention)?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

/// RDM hot-path evaluator.  It keeps the public pure-Rust
/// kernel as the fail-closed contract gate (physical filtering, stable order,
/// book validity and global ladder-log checks), reconstructs its event
/// selection in Rust, and delegates only the final mean to NumPy.  It is
/// covered by exact-reference bitwise-or-NaN regression tests because the Rust
/// direction thresholds must pick exactly the same intervals as NumPy.
fn rdm_numpy_mean_with_rust_directions(py: Python<'_>, input_rows: &[BookRow]) -> PyResult<f64> {
    // This invokes the production pure-Rust implementation first so every
    // rejected input retains its existing fail-closed result.  Its scalar mean
    // is intentionally not returned: NumPy owns the promotion reduction path.
    if !rdm_joint_reprice_depth_retention(input_rows).is_finite() {
        return Ok(f64::NAN);
    }
    let rows = ordered_lrd_rows(input_rows);
    let mut retention = Vec::with_capacity(rows.len().saturating_sub(1));
    for pair in rows.windows(2) {
        let (previous, previous_session) = pair[0];
        let (current, current_session) = pair[1];
        let same_session = previous_session == current_session;
        let Some(bid_direction) =
            rdm_rust_ladder_direction(&previous.bid_price, &current.bid_price, same_session)
        else {
            return Ok(f64::NAN);
        };
        let Some(ask_direction) =
            rdm_rust_ladder_direction(&previous.ask_price, &current.ask_price, same_session)
        else {
            return Ok(f64::NAN);
        };
        if bid_direction == 0.0 || ask_direction == 0.0 {
            continue;
        }
        let previous_depth = previous
            .bid_volume
            .iter()
            .chain(previous.ask_volume.iter())
            .sum::<f64>();
        let current_depth = current
            .bid_volume
            .iter()
            .chain(current.ask_volume.iter())
            .sum::<f64>();
        if !previous_depth.is_finite()
            || !current_depth.is_finite()
            || previous_depth <= 1e-12
            || current_depth < 0.0
        {
            return Ok(f64::NAN);
        }
        let value = previous_depth.min(current_depth) / previous_depth;
        if !value.is_finite() {
            return Ok(f64::NAN);
        }
        retention.push(value);
    }
    if retention.len() < 3 {
        return Ok(f64::NAN);
    }
    let value = numpy_mean(py, &retention)?;
    Ok(if value.is_finite() { value } else { f64::NAN })
}

/// Rust mirror of the pure RDM direction classifier.  The log is evaluated
/// before the session gate because Python materializes all ladder log moves
/// before masking lunch-crossing intervals.
fn rdm_rust_ladder_direction(
    previous: &[f64; 5],
    current: &[f64; 5],
    same_session: bool,
) -> Option<f64> {
    let mut positive = 0usize;
    let mut negative = 0usize;
    for level in 0..5 {
        let log_move = (current[level] / previous[level]).ln();
        if !log_move.is_finite() {
            return None;
        }
        if same_session {
            if log_move > 1e-12 {
                positive += 1;
            } else if log_move < -1e-12 {
                negative += 1;
            }
        }
    }
    if !same_session {
        return Some(0.0);
    }
    Some(if positive >= 3 && positive > negative {
        1.0
    } else if negative >= 3 && negative > positive {
        -1.0
    } else {
        0.0
    })
}

/// Prepare the RDM output once for every labelled `(dt, code)` requested by
/// the RDM spec.  The R38-only backfill requests this one new factor, while a
/// mixed request can still retain its existing QED/LRD output paths.
fn prepare_rdm_intraday_values(
    py: Python<'_>,
    specs: &[KernelSpec],
    per_spec_keys: &[BTreeSet<(String, String)>],
    ctx: &TypedFactorIntradayContext,
) -> PyResult<PreparedRdmValues> {
    let rdm_keys: BTreeSet<_> = specs
        .iter()
        .zip(per_spec_keys.iter())
        .filter(|(spec, _)| {
            spec.factor == LRD_FACTOR && spec.signal == "rdm_joint_reprice_depth_retention"
        })
        .flat_map(|(_, keys)| keys.iter().cloned())
        .collect();
    let mut prepared = BTreeMap::new();
    for key in rdm_keys {
        let value = match ctx.lrd_rows.get(&key) {
            Some(rows) => rdm_numpy_mean_with_rust_directions(py, rows)?,
            None => f64::NAN,
        };
        prepared.insert(key, if value.is_finite() { value } else { f64::NAN });
    }
    Ok(prepared)
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

#[derive(Clone, Copy, Debug)]
enum BssrcMetric {
    UpperAlignment,
    Correlation,
    LowerAlignment,
}

/// Score-day BSSRC output cache.  The two tail factors and the correlation
/// share the same strict source universe, cross-sectional ranks, terminal
/// gates, and 60-row paths.  Keeping values keyed by the already canonical
/// bond code makes the ordinary output loop a lookup only.
#[derive(Debug, Default)]
struct PreparedBssrcValues {
    upper_alignment: BTreeMap<String, f64>,
    correlation: BTreeMap<String, f64>,
    lower_alignment: BTreeMap<String, f64>,
}

impl PreparedBssrcValues {
    fn insert(&mut self, metric: BssrcMetric, code: String, value: f64) {
        match metric {
            BssrcMetric::UpperAlignment => {
                self.upper_alignment.insert(code, value);
            }
            BssrcMetric::Correlation => {
                self.correlation.insert(code, value);
            }
            BssrcMetric::LowerAlignment => {
                self.lower_alignment.insert(code, value);
            }
        }
    }

    fn lookup(&self, metric: BssrcMetric, code: &str) -> Option<f64> {
        match metric {
            BssrcMetric::UpperAlignment => self.upper_alignment.get(code).copied(),
            BssrcMetric::Correlation => self.correlation.get(code).copied(),
            BssrcMetric::LowerAlignment => self.lower_alignment.get(code).copied(),
        }
    }
}

/// Return only the two R38 research additions that share a cache.  The live50
/// `bssrc_bond_stock_rank_correlation60` instance intentionally continues
/// through its established direct path: R88 research performance work must
/// not alter a live factor's execution semantics.
fn r38_bssrc_metric_for_spec(spec: &KernelSpec) -> Option<BssrcMetric> {
    (spec.factor == "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1")
        .then(|| match spec.signal.as_str() {
            "bssrc_upper_rank_tail_alignment60" => Some(BssrcMetric::UpperAlignment),
            "bssrc_lower_rank_tail_alignment60" => Some(BssrcMetric::LowerAlignment),
            _ => None,
        })
        .flatten()
}

/// Prepare every requested BSSRC metric for one score day.  This deliberately
/// retains the exact Python/NumPy reductions of `exact_bssrc_metric`: NumPy
/// owns log-return and Pearson arithmetic, while the cache merely moves the
/// score-day-global rank construction outside the `(spec, code)` output loop.
///
/// `exact_bssrc_metric` remains the direct reference implementation below.
fn prepare_bssrc_values(
    py: Python<'_>,
    ctx: &TypedFactorDailyCrossAssetContext,
    requested: &[(BssrcMetric, String)],
) -> PyResult<PreparedBssrcValues> {
    let mut prepared = PreparedBssrcValues::default();
    if requested.is_empty() {
        return Ok(prepared);
    }
    let Some((price, base, _sessions, anchor)) = strict_cross_sources(ctx) else {
        return Ok(prepared);
    };

    // These blocks intentionally match the direct reference's global source
    // construction and ordering.  In particular, each underlying is checked
    // before target-code selection so its fail-closed error surface stays
    // global rather than dependent on the requested output universe.
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

    let requested_by_code: BTreeMap<String, Vec<BssrcMetric>> = requested
        .iter()
        .filter_map(|(metric, raw_code)| {
            canonical_market_code(
                Some(raw_code.as_str()),
                None,
                CanonicalContract::StrictExchange,
            )
            .filter(|code| !code.is_empty())
            .map(|code| (code, *metric))
        })
        .fold(BTreeMap::new(), |mut output, (code, metric)| {
            output.entry(code).or_default().push(metric);
            output
        });
    if requested_by_code.is_empty() {
        return Ok(prepared);
    }

    let mut paths: BTreeMap<String, Vec<(NaiveDate, f64, f64)>> = requested_by_code
        .keys()
        .cloned()
        .map(|code| (code, Vec::new()))
        .collect();
    for row in price.iter().copied() {
        let Some(path) = paths.get_mut(&row.code) else {
            continue;
        };
        let key = (row.trade_date, row.code.clone());
        if let Some(stock_rank) = stock_rank_by_bond.get(&key) {
            path.push((
                row.trade_date,
                bond_rank.get(&key).copied().unwrap_or(f64::NAN),
                *stock_rank,
            ));
        }
    }

    for (code, metrics) in requested_by_code {
        let Some(path) = paths.get(&code) else {
            continue;
        };
        if path.last().map(|(trade_date, _, _)| *trade_date) != Some(anchor) {
            continue;
        }
        let start = path.len().saturating_sub(60);
        let recent = &path[start..];
        let Some((_, terminal_bond, terminal_stock)) = recent.last() else {
            continue;
        };
        if !terminal_bond.is_finite() || !terminal_stock.is_finite() {
            continue;
        }
        let pairs: Vec<(f64, f64)> = recent
            .iter()
            .filter_map(|(_, bond, stock)| {
                (bond.is_finite() && stock.is_finite()).then_some((*bond, *stock))
            })
            .collect();
        if pairs.len() < 45 {
            continue;
        }
        let upper = pairs.iter().filter(|(_, stock)| *stock >= 0.75).count();
        let lower = pairs.iter().filter(|(_, stock)| *stock <= 0.25).count();
        if upper < 8 || lower < 8 {
            continue;
        }
        let bond: Vec<f64> = pairs.iter().map(|(bond, _)| *bond).collect();
        let stock: Vec<f64> = pairs.iter().map(|(_, stock)| *stock).collect();
        let correlation = numpy_centered_correlation(py, &bond, &stock)?;
        if !correlation.is_finite() {
            continue;
        }
        let upper_alignment = pairs
            .iter()
            .filter(|(bond, stock)| *stock >= 0.75 && *bond >= 0.75)
            .count() as f64
            / upper as f64;
        let lower_alignment = pairs
            .iter()
            .filter(|(bond, stock)| *stock <= 0.25 && *bond <= 0.25)
            .count() as f64
            / lower as f64;
        for metric in metrics {
            let value = match metric {
                BssrcMetric::UpperAlignment => upper_alignment,
                BssrcMetric::Correlation => correlation,
                BssrcMetric::LowerAlignment => lower_alignment,
            };
            prepared.insert(metric, code.clone(), value);
        }
    }
    Ok(prepared)
}

fn exact_bssrc_metric(
    py: Python<'_>,
    ctx: &TypedFactorDailyCrossAssetContext,
    code: &str,
    metric: BssrcMetric,
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
    let correlation = numpy_centered_correlation(py, &bond, &stock)?;
    if !correlation.is_finite() {
        return Ok(f64::NAN);
    }
    let value = match metric {
        BssrcMetric::Correlation => correlation,
        BssrcMetric::UpperAlignment => {
            pairs
                .iter()
                .filter(|(bond, stock)| *stock >= 0.75 && *bond >= 0.75)
                .count() as f64
                / upper as f64
        }
        BssrcMetric::LowerAlignment => {
            pairs
                .iter()
                .filter(|(bond, stock)| *stock <= 0.25 && *bond <= 0.25)
                .count() as f64
                / lower as f64
        }
    };
    Ok(if value.is_finite() { value } else { f64::NAN })
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
        "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1" => {
            let metric = match spec.signal.as_str() {
                "bssrc_upper_rank_tail_alignment60" => BssrcMetric::UpperAlignment,
                "bssrc_bond_stock_rank_correlation60" => BssrcMetric::Correlation,
                "bssrc_lower_rank_tail_alignment60" => BssrcMetric::LowerAlignment,
                _ => {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "typed_factor daily BSSRC kernel has unknown signal={}",
                        spec.signal
                    )));
                }
            };
            exact_bssrc_metric(py, cross_asset_ctx, code, metric)
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
        "factor_mining_daily_contract_stock_v1" if spec.signal == "rating_current_ordinal" => {
            // This calls the typed context's independent daily-price anchor
            // guard before mapping the terminal rating string to its ordinal.
            compute_p1_daily_signal(ctx, code, &spec.signal)
                .map_err(|error| PyErr::new::<PyRuntimeError, _>(error.to_string()))
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
    let specs = parse_specs(specs_payload)?;
    validate_r88_specs(&specs)?;
    let has_r88_spec = specs.iter().any(|spec| is_r88_factor(&spec.factor));
    let r88_phase_timing = R88PhaseTiming::maybe_start(has_r88_spec);
    let shared_dispatch_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
    let has_r88_remaining = specs.iter().any(is_r88_remaining_spec);
    // Existing typed families do not consume panel/map context.  Keep their
    // historical rejection exactly; the only narrow exception is the
    // research-only R88 remaining adapter below.
    if (stock_df.is_some() || map_df.is_some()) && !has_r88_remaining {
        return Err(PyErr::new::<PyValueError, _>(
            "typed_factor kernel does not accept stock_df or map_df",
        ));
    }
    let has_qed = specs.iter().any(is_qed_spec);
    let has_lrd = specs.iter().any(is_lrd_spec);
    let has_r88_direct_intraday = specs.iter().any(is_r88_direct_intraday_spec);
    let has_r88_remaining_intraday = specs.iter().any(is_r88_remaining_intraday_spec);
    let has_intraday = has_qed || has_lrd || has_r88_direct_intraday || has_r88_remaining_intraday;
    let has_daily = specs
        .iter()
        .any(|spec| !is_intraday_spec(spec) || r88_remaining_uses_daily_data(spec));

    // Python LRD calls `ensure_trade_time` before it resolves __build_day__.
    // Preserve that error precedence without pre-validating the rest of the
    // LRD book schema (those fields are checked after score-day resolution by
    // the Python reference).
    if has_lrd {
        require_panel_columns(py, panel_df, LRD_FACTOR, ["trade_time".to_string()])?;
    }
    if has_r88_direct_intraday {
        require_panel_columns(
            py,
            panel_df,
            "typed_factor R88 direct intraday",
            r88_panel_columns_for_direct_specs(&specs),
        )?;
    }
    if has_r88_remaining_intraday {
        require_panel_columns(
            py,
            panel_df,
            "typed_factor R88 remaining intraday",
            r88_panel_columns_for_remaining_specs(&specs, false),
        )?;
        if r88_remaining_requires_stock(&specs) {
            let stock = stock_df.ok_or_else(|| {
                PyErr::new::<PyKeyError, _>("typed_factor R88 remaining requires stock_df")
            })?;
            if is_empty(stock)? {
                return Err(PyErr::new::<PyValueError, _>(
                    "typed_factor R88 remaining requires non-empty stock_df",
                ));
            }
            require_panel_columns(
                py,
                stock,
                "typed_factor R88 remaining stock intraday",
                r88_panel_columns_for_remaining_specs(&specs, true),
            )?;
        }
    }
    if r88_remaining_requires_map(&specs) && map_df.is_none() {
        return Err(PyErr::new::<PyKeyError, _>(
            "typed_factor R88 ITR requires bond_stock_map",
        ));
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
    if has_r88_direct_intraday || has_r88_remaining || specs.iter().any(is_r88_daily_spec) {
        if let (Some(daily), Some(intraday)) = (daily_score_date, intraday_score_date) {
            if daily != intraday {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "typed_factor R88 score-date provenance mismatch: daily={daily}, intraday={intraday}"
                )));
            }
        }
    }
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
    let r88_intraday_keys: BTreeSet<(String, String)> =
        if has_r88_direct_intraday || has_r88_remaining_intraday {
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
            } else if is_r88_direct_intraday_spec(spec) || is_r88_remaining_intraday_spec(spec) {
                r88_intraday_keys.clone()
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
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing.as_ref(), shared_dispatch_phase_started)
    {
        timing.record_phase("shared_dispatch_prep", phase_started_at);
    }
    // LRD's strict schema has already been checked above.  Every other family
    // exits before daily-source validation when its output universe is empty.
    if keys.is_empty() {
        let output_loop_phase_started = r88_phase_timing
            .as_ref()
            .map(|timing| timing.phase_started());
        let columns = specs
            .iter()
            .map(|spec| (spec.output_col.clone(), Vec::new()))
            .collect::<Vec<_>>();
        let output = build_ordered_output(py, keys, columns, r88_phase_timing.as_ref());
        if let (Some(timing), Some(phase_started_at)) =
            (r88_phase_timing.as_ref(), output_loop_phase_started)
        {
            timing.record_phase("output_loop", phase_started_at);
            timing.finish();
        }
        return output;
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
    let r88_daily_needs = r88_daily_source_needs(&specs);
    let r88_remaining_needs = r88_remaining_source_needs(&specs);
    let mut r88_daily_ctx = TypedFactorR88DailyContext::new(context_score_date);
    let mut r88_intraday_ctx = TypedFactorR88IntradayContext::default();
    let mut r88_remaining_ctx = TypedFactorR88RemainingContext::new(context_score_date);
    let legacy_daily_prep_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
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
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing.as_ref(), legacy_daily_prep_phase_started)
    {
        timing.record_phase("legacy_daily_prep", phase_started_at);
    }

    let r88_daily_prep_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
    if specs.iter().any(is_r88_daily_spec) {
        populate_r88_daily_context(
            py,
            daily_data,
            &r88_daily_needs,
            context_score_date,
            &mut r88_daily_ctx,
        )?;
    }
    // The seven daily R88 formulas share strict-prior source parsing, joins
    // and rank construction.  Prepare their score-day values once before the
    // ordinary spec/key loop; direct per-code functions remain the reference
    // implementation in the daily module and test suite.
    let r88_daily_prepared = prepare_r88_daily_values(
        &r88_daily_ctx,
        specs
            .iter()
            .zip(per_spec_keys.iter())
            .filter(|(spec, spec_keys)| is_r88_daily_spec(spec) && !spec_keys.is_empty())
            .filter_map(|(spec, _)| TypedFactorR88DailySignal::parse(&spec.signal)),
        specs
            .iter()
            .zip(per_spec_keys.iter())
            .filter(|(spec, _)| is_r88_daily_spec(spec))
            .flat_map(|(_, spec_keys)| spec_keys.iter().map(|(_, code)| code.clone())),
    )
    .map_err(|error| PyErr::new::<PyRuntimeError, _>(error.to_string()))?;
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing.as_ref(), r88_daily_prep_phase_started)
    {
        timing.record_phase("r88_daily_prep", phase_started_at);
    }

    let r88_remaining_daily_prep_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
    if specs.iter().any(r88_remaining_uses_daily_data) {
        populate_r88_remaining_daily_context(
            py,
            daily_data,
            &r88_remaining_needs,
            context_score_date,
            &mut r88_remaining_ctx,
        )?;
    }
    if let (Some(timing), Some(phase_started_at)) = (
        r88_phase_timing.as_ref(),
        r88_remaining_daily_prep_phase_started,
    ) {
        timing.record_phase("r88_remaining_daily_prep", phase_started_at);
    }
    let r88_remaining_output_codes: BTreeSet<String> = specs
        .iter()
        .zip(per_spec_keys.iter())
        .filter(|(spec, _)| is_r88_remaining_spec(spec))
        .flat_map(|(_, spec_keys)| spec_keys.iter().map(|(_, code)| code.clone()))
        .collect();
    r88_remaining_ctx.panel_codes = r88_remaining_output_codes.into_iter().collect();
    if let Some(score_date) = intraday_score_date {
        let append_bond_parsing_phase_started = r88_phase_timing
            .as_ref()
            .map(|timing| timing.phase_started());
        if has_r88_direct_intraday {
            append_r88_direct_intraday_rows(py, panel_df, score_date, &mut r88_intraday_ctx)?;
        }
        if has_r88_remaining_intraday {
            append_r88_remaining_intraday_rows(py, panel_df, score_date, &mut r88_remaining_ctx)?;
        }
        if let (Some(timing), Some(phase_started_at)) =
            (r88_phase_timing.as_ref(), append_bond_parsing_phase_started)
        {
            timing.record_phase("append_bond_parsing", phase_started_at);
        }
        if has_r88_remaining_intraday && r88_remaining_requires_stock(&specs) {
            let append_stock_parsing_phase_started = r88_phase_timing
                .as_ref()
                .map(|timing| timing.phase_started());
            let stock = stock_df.expect("validated R88 stock_df presence");
            append_r88_remaining_intraday_rows(py, stock, score_date, &mut r88_remaining_ctx)?;
            if let (Some(timing), Some(phase_started_at)) = (
                r88_phase_timing.as_ref(),
                append_stock_parsing_phase_started,
            ) {
                timing.record_phase("append_stock_parsing", phase_started_at);
            }
        }
        if has_r88_remaining_intraday && r88_remaining_requires_map(&specs) {
            let append_map_parsing_phase_started = r88_phase_timing
                .as_ref()
                .map(|timing| timing.phase_started());
            populate_r88_remaining_map_context(py, map_df, score_date, &mut r88_remaining_ctx)?;
            if let (Some(timing), Some(phase_started_at)) =
                (r88_phase_timing.as_ref(), append_map_parsing_phase_started)
            {
                timing.record_phase("append_map_parsing", phase_started_at);
            }
        }
    }
    let r88_direct_prep_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
    let r88_direct_prepared =
        prepare_r88_direct_intraday_values(&specs, &per_spec_keys, &r88_intraday_ctx);
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing.as_ref(), r88_direct_prep_phase_started)
    {
        timing.record_phase("r88_direct_prep", phase_started_at);
    }
    // UCD/SNG/CSN are cross-sectional functions.  Build every requested
    // score-day result exactly once before the ordinary spec/key loop; direct
    // functions remain the fall-through reference for non-cached signals.
    let r88_remaining_prep_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
    let r88_remaining_prepared = prepare_r88_remaining_values(
        &r88_remaining_ctx,
        specs
            .iter()
            .zip(per_spec_keys.iter())
            .filter(|(spec, spec_keys)| is_r88_remaining_spec(spec) && !spec_keys.is_empty())
            .filter_map(|(spec, _)| TypedFactorR88RemainingSignal::parse(&spec.signal)),
    )
    .map_err(|error| PyErr::new::<PyRuntimeError, _>(error.to_string()))?;
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing.as_ref(), r88_remaining_prep_phase_started)
    {
        timing.record_phase("r88_remaining_prep", phase_started_at);
    }

    // RDM is the R38-only order-book addition.  Prepare it after every source
    // and context validation but before output assembly, so the ordinary
    // `(spec, key)` loop is a lookup rather than a per-bond NumPy direction
    // construction.  The legacy LRD output keeps its established exact
    // reference path unchanged.
    let rdm_prep_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
    let rdm_prepared = prepare_rdm_intraday_values(py, &specs, &per_spec_keys, &intraday_ctx)?;
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing.as_ref(), rdm_prep_phase_started)
    {
        timing.record_phase("rdm_prep", phase_started_at);
    }

    // BSSRC's upper/lower tail outputs and correlation all rebuild the same
    // score-day cross-sectional ranks in the direct reference.  Prepare that
    // state once here so the output loop only resolves a canonical-code value.
    let bssrc_prep_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
    let bssrc_requested: Vec<_> = specs
        .iter()
        .zip(per_spec_keys.iter())
        .filter_map(|(spec, spec_keys)| {
            r38_bssrc_metric_for_spec(spec).map(|metric| (metric, spec_keys))
        })
        .flat_map(|(metric, spec_keys)| {
            spec_keys
                .iter()
                .map(move |(_, code)| (metric, code.clone()))
        })
        .collect();
    let bssrc_prepared = if populate_daily {
        prepare_bssrc_values(py, &cross_asset_ctx, &bssrc_requested)?
    } else {
        PreparedBssrcValues::default()
    };
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing.as_ref(), bssrc_prep_phase_started)
    {
        timing.record_phase("bssrc_rank_prep", phase_started_at);
    }

    let output_loop_phase_started = r88_phase_timing
        .as_ref()
        .map(|timing| timing.phase_started());
    let mut columns = Vec::with_capacity(specs.len());
    for (spec, spec_keys) in specs.iter().zip(per_spec_keys.iter()) {
        let output_spec_phase_started = r88_phase_timing
            .as_ref()
            .map(|timing| timing.phase_started());
        let mut values = Vec::with_capacity(keys.len());
        for key @ (_, output_code) in &keys {
            if !spec_keys.contains(key) {
                values.push(f64::NAN);
            } else if is_qed_spec(spec) {
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
            } else if is_lrd_spec(spec) {
                let value = match intraday_ctx.lrd_rows.get(key) {
                    Some(rows) => match spec.signal.as_str() {
                        "lrd_cross_side_reprice_symmetry" => exact_lrd_value(py, rows)?,
                        "rdm_joint_reprice_depth_retention" => {
                            rdm_prepared.get(key).copied().unwrap_or(f64::NAN)
                        }
                        _ => f64::NAN,
                    },
                    None => f64::NAN,
                };
                values.push(if value.is_finite() { value } else { f64::NAN });
            } else if is_r88_direct_intraday_spec(spec) {
                let value = r88_direct_prepared
                    .get(&(spec.output_col.clone(), key.0.clone(), key.1.clone()))
                    .copied()
                    .unwrap_or(f64::NAN);
                values.push(if value.is_finite() { value } else { f64::NAN });
            } else if is_r88_daily_spec(spec) {
                let signal = TypedFactorR88DailySignal::parse(&spec.signal)
                    .expect("validated R88 daily signal");
                let value = r88_daily_prepared
                    .lookup(signal, output_code)
                    .unwrap_or(f64::NAN);
                values.push(if value.is_finite() { value } else { f64::NAN });
            } else if is_r88_remaining_spec(spec) {
                let signal = TypedFactorR88RemainingSignal::parse(&spec.signal)
                    .expect("validated R88 remaining signal");
                let value = if let Some(value) = r88_remaining_prepared.lookup(signal, output_code)
                {
                    value
                } else {
                    compute_r88_remaining_signal(&r88_remaining_ctx, output_code, &spec.signal)
                        .map_err(|error| PyErr::new::<PyRuntimeError, _>(error.to_string()))?
                };
                values.push(if value.is_finite() { value } else { f64::NAN });
            } else if let Some(metric) = r38_bssrc_metric_for_spec(spec) {
                let code = canonical_market_code(
                    Some(output_code),
                    None,
                    CanonicalContract::StrictExchange,
                )
                .unwrap_or_default();
                let value = bssrc_prepared.lookup(metric, &code).unwrap_or(f64::NAN);
                values.push(if value.is_finite() { value } else { f64::NAN });
            } else if !populate_daily {
                values.push(f64::NAN);
            } else {
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
        }
        columns.push((spec.output_col.clone(), values));
        if let (Some(timing), Some(spec_started_at)) =
            (r88_phase_timing.as_ref(), output_spec_phase_started)
        {
            timing.record_output_spec(spec, spec_started_at);
        }
    }
    let output = build_ordered_output(py, keys, columns, r88_phase_timing.as_ref());
    if let (Some(timing), Some(phase_started_at)) =
        (r88_phase_timing.as_ref(), output_loop_phase_started)
    {
        timing.record_phase("output_loop", phase_started_at);
        timing.finish();
    }
    output
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
    use super::{
        build_ordered_output, canonical_market_code, exact_bssrc_metric,
        exact_rdm_joint_depth_retention, is_supported_typed_spec, prepare_bssrc_values,
        prepare_r88_direct_intraday_values, prepare_rdm_intraday_values, r88_expected_family,
        rdm_numpy_mean_with_rust_directions, session_label, validate_r88_specs, BookRow,
        BssrcMetric, CanonicalContract, KernelSpec, TypedFactorCrossAssetBaseRow,
        TypedFactorCrossAssetPriceRow, TypedFactorDailyCrossAssetContext,
        TypedFactorIntradayContext, TypedFactorR88IntradayContext, LRD_FACTOR,
    };
    use crate::typed_factor_r88_intraday::{r88_intraday_metrics, R88IntradayRow};
    use chrono::{Duration, NaiveDate};
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
    use pyo3::Python;
    use std::collections::BTreeSet;

    fn direct_spec(output_col: &str, factor: &str, signal: &str) -> KernelSpec {
        KernelSpec {
            factor: factor.to_string(),
            signal: signal.to_string(),
            family: String::new(),
            output_col: output_col.to_string(),
        }
    }

    fn assert_bitwise_or_nan(actual: f64, expected: f64) {
        assert!(
            (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits(),
            "actual={actual:?} ({:#x}), expected={expected:?} ({:#x})",
            actual.to_bits(),
            expected.to_bits(),
        );
    }

    #[test]
    fn ordered_output_numpy_columns_match_legacy_list_dataframe_exactly() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let keys = vec![
                ("2026-08-25 14:30:00".to_string(), "110001.SH".to_string()),
                ("2026-08-25 14:30:00".to_string(), "110002.SH".to_string()),
                ("2026-08-25 14:30:00".to_string(), "110003.SH".to_string()),
            ];
            let nan_payload = f64::from_bits(0x7ff8_0000_0000_0042);
            let values = vec![
                ("first".to_string(), vec![1.25, nan_payload, -0.0]),
                ("second".to_string(), vec![f64::INFINITY, -3.5, 9.0]),
            ];

            let actual = build_ordered_output(py, keys.clone(), values.clone(), None).unwrap();
            let actual = actual.bind(py);

            // Reproduce the prior list-backed construction literally.  This
            // locks names, order, dtypes, and pandas' NaN semantics while the
            // optimized implementation changes only its numeric bridge.
            let pandas = py.import_bound("pandas").unwrap();
            let expected_data = PyDict::new_bound(py);
            expected_data
                .set_item(
                    "dt",
                    keys.iter().map(|(dt, _)| dt.clone()).collect::<Vec<_>>(),
                )
                .unwrap();
            expected_data
                .set_item(
                    "code",
                    keys.iter()
                        .map(|(_, code)| code.clone())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            for (column, column_values) in &values {
                expected_data.set_item(column, column_values).unwrap();
            }
            let expected = pandas.call_method1("DataFrame", (expected_data,)).unwrap();
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("check_dtype", true).unwrap();
            kwargs.set_item("check_exact", true).unwrap();
            pandas
                .getattr("testing")
                .unwrap()
                .call_method("assert_frame_equal", (actual, &expected), Some(&kwargs))
                .unwrap();

            let actual_columns = actual
                .getattr("columns")
                .unwrap()
                .call_method0("tolist")
                .unwrap()
                .extract::<Vec<String>>()
                .unwrap();
            assert_eq!(actual_columns, vec!["dt", "code", "first", "second"]);
            for (column, expected_values) in &values {
                let actual_values = actual
                    .call_method1("__getitem__", (column,))
                    .unwrap()
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f64>>()
                    .unwrap();
                assert_eq!(actual_values.len(), expected_values.len());
                for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
                    assert_bitwise_or_nan(*actual_value, *expected_value);
                }
            }
        });
    }

    fn rdm_book_row(index: i64, price_scale: f64, depth: f64) -> BookRow {
        let bid_anchor = 100.0 * price_scale;
        let ask_anchor = 101.0 * price_scale;
        let mut bid_price = [0.0; 5];
        let mut ask_price = [0.0; 5];
        for level in 0..5 {
            bid_price[level] = bid_anchor - level as f64 * 0.01;
            ask_price[level] = ask_anchor + level as f64 * 0.01;
        }
        BookRow {
            time_ns: (9 * 60 * 60 + 30 * 60 + index) * 1_000_000_000,
            seq: index,
            ask_price,
            bid_price,
            ask_volume: [depth; 5],
            bid_volume: [depth * 1.5; 5],
        }
    }

    #[test]
    fn rdm_numpy_mean_with_rust_directions_matches_exact_reference_bitwise_or_nan() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let regular = (0..12_i64)
                .map(|index| {
                    rdm_book_row(index, 1.0 + index as f64 * 2e-4, 20.0 - index as f64 * 0.5)
                })
                .collect::<Vec<_>>();
            let expected = exact_rdm_joint_depth_retention(py, &regular).unwrap();
            let actual = rdm_numpy_mean_with_rust_directions(py, &regular).unwrap();
            assert_bitwise_or_nan(actual, expected);

            // Values immediately below and above the 1e-12 direction cutoff
            // exercise the only point where Rust and NumPy log paths may
            // select different reprice intervals.
            let near_threshold = (0..12_i64)
                .map(|index| {
                    let scale = match index {
                        0 => 1.0,
                        1 => 1.0 + 5e-13,
                        _ => 1.0 + index as f64 * 3e-12,
                    };
                    rdm_book_row(index, scale, 25.0 - index as f64 * 0.25)
                })
                .collect::<Vec<_>>();
            let expected = exact_rdm_joint_depth_retention(py, &near_threshold).unwrap();
            let actual = rdm_numpy_mean_with_rust_directions(py, &near_threshold).unwrap();
            assert_bitwise_or_nan(actual, expected);

            let mut invalid = regular.clone();
            invalid[4].ask_volume[2] = -0.1;
            let expected = exact_rdm_joint_depth_retention(py, &invalid).unwrap();
            let actual = rdm_numpy_mean_with_rust_directions(py, &invalid).unwrap();
            assert_bitwise_or_nan(actual, expected);
        });
    }

    #[test]
    fn rdm_numpy_mean_with_rust_directions_matches_exact_reference_across_order_session_and_contract_edges(
    ) {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let assert_matches = |rows: &[BookRow]| {
                let expected = exact_rdm_joint_depth_retention(py, rows).unwrap();
                let actual = rdm_numpy_mean_with_rust_directions(py, rows).unwrap();
                assert_bitwise_or_nan(actual, expected);
            };

            // Alternating joint up/down reprices with non-constant retention.
            let alternating = (0..14_i64)
                .map(|index| {
                    let scale = if index % 2 == 0 {
                        1.0 + index as f64 * 2e-4
                    } else {
                        1.0 - index as f64 * 1e-4
                    };
                    rdm_book_row(index, scale, 30.0 - index as f64 * 0.9)
                })
                .collect::<Vec<_>>();
            assert_matches(&alternating);

            // Stable sorting uses (time, seq, original position), not input
            // order; include a lunch boundary to prove the same-session gate.
            let mut shuffled = alternating.clone();
            for (index, row) in shuffled.iter_mut().enumerate() {
                if index >= 7 {
                    row.time_ns = (13 * 60 * 60 + (index as i64 - 7) * 60) * 1_000_000_000;
                }
            }
            shuffled.reverse();
            assert_matches(&shuffled);

            // Every ladder move is below the strict direction threshold.
            let below_threshold = (0..12_i64)
                .map(|index| rdm_book_row(index, 1.0 + index as f64 * 5e-13, 20.0))
                .collect::<Vec<_>>();
            assert_matches(&below_threshold);

            let mut invalid_ladder = alternating.clone();
            invalid_ladder[5].ask_price[3] = invalid_ladder[5].ask_price[2] - 0.01;
            assert_matches(&invalid_ladder);

            let insufficient = alternating[..11].to_vec();
            assert_matches(&insufficient);
        });
    }

    #[test]
    fn rdm_prepared_cache_matches_exact_reference_for_each_output_key() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let key_a = ("2026-08-25 14:30:00".to_string(), "110001.SH".to_string());
            let key_b = ("2026-08-25 14:30:00".to_string(), "110002.SH".to_string());
            let rows_a = (0..12_i64)
                .map(|index| {
                    rdm_book_row(index, 1.0 + index as f64 * 2e-4, 20.0 - index as f64 * 0.5)
                })
                .collect::<Vec<_>>();
            let rows_b = (0..12_i64)
                .map(|index| {
                    rdm_book_row(index, 1.0 - index as f64 * 1e-4, 25.0 - index as f64 * 0.25)
                })
                .collect::<Vec<_>>();
            let mut ctx = TypedFactorIntradayContext::default();
            ctx.lrd_rows.insert(key_a.clone(), rows_a.clone());
            ctx.lrd_rows.insert(key_b.clone(), rows_b.clone());
            let specs = vec![direct_spec(
                "rdm_joint_reprice_depth_retention",
                LRD_FACTOR,
                "rdm_joint_reprice_depth_retention",
            )];
            let per_spec_keys = vec![BTreeSet::from([key_a.clone(), key_b.clone()])];
            let prepared = prepare_rdm_intraday_values(py, &specs, &per_spec_keys, &ctx).unwrap();
            assert_eq!(prepared.len(), 2);
            for (key, rows) in [(key_a, rows_a), (key_b, rows_b)] {
                let expected = exact_rdm_joint_depth_retention(py, &rows).unwrap();
                let actual = prepared.get(&key).copied().expect("RDM key was cached");
                assert_bitwise_or_nan(actual, expected);
            }
        });
    }

    #[test]
    fn r88_direct_bundle_cache_matches_each_named_metric() {
        let key = ("2026-01-02 14:30:00".to_string(), "110001.SH".to_string());
        let rows: Vec<_> = (0..12_i64)
            .map(|index| R88IntradayRow {
                time_ns: (9 * 60 * 60 + 30 * 60 + index) * 1_000_000_000,
                seq: index,
                last: 100.0 + index as f64,
                volume: index as f64,
                amount: 10.0 * index as f64,
                num_trades: index as f64,
                ask_price1: 101.0 + index as f64,
                bid_price1: 99.0 + index as f64,
                ask_volume1: 10.0 + index as f64,
                bid_volume1: 20.0 + index as f64,
            })
            .collect();
        let mut ctx = TypedFactorR88IntradayContext::default();
        ctx.rows.insert(key.clone(), rows.clone());
        let specs = vec![
            direct_spec(
                "exp_noise_variance_ratio_2",
                "factor_mining_intraday_expansion_v1",
                "exp_noise_variance_ratio_2",
            ),
            direct_spec(
                "execdisc_step_multiplicity_entropy",
                "factor_mining_intraday_execution_discreteness_v1",
                "execdisc_step_multiplicity_entropy",
            ),
            direct_spec(
                "book_quote_dislocation",
                "factor_mining_intraday_catalog_v1",
                "book_quote_dislocation",
            ),
        ];
        let keys = vec![BTreeSet::from([key.clone()]); 3];
        let cached = prepare_r88_direct_intraday_values(&specs, &keys, &ctx);
        for spec in &specs[..2] {
            let expected = r88_intraday_metrics(&rows).value(&spec.signal);
            let actual = cached
                .get(&(spec.output_col.clone(), key.0.clone(), key.1.clone()))
                .copied()
                .expect("direct signal is cached exactly once per output key");
            assert!(
                (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
            );
        }
        let continuous: Vec<_> = rows
            .iter()
            .copied()
            .filter(|row| session_label(row.time_ns).is_some())
            .collect();
        let expected = r88_intraday_metrics(&continuous).value("book_quote_dislocation");
        let actual = cached
            .get(&(
                "book_quote_dislocation".to_string(),
                key.0.clone(),
                key.1.clone(),
            ))
            .copied()
            .expect("catalogue signal is cached exactly once per output key");
        assert!((actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits());
    }

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

    fn bssrc_cache_day(offset: i64) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid date") + Duration::days(offset)
    }

    fn bssrc_cache_context() -> TypedFactorDailyCrossAssetContext {
        let mut ctx = TypedFactorDailyCrossAssetContext::new(bssrc_cache_day(65));
        let codes = [
            "110001.SH",
            "110002.SH",
            "110003.SH",
            "110004.SH",
            "110005.SH",
        ];
        for offset in 0_i64..65 {
            for (index, code) in codes.iter().enumerate() {
                let bond_return = 0.019 * (0.29 * offset as f64 + 0.71 * index as f64).sin()
                    + 0.004 * (0.11 * offset as f64 + 0.17 * index as f64).cos();
                let group = if index == 0 {
                    (offset as usize) % 4
                } else {
                    index - 1
                };
                let stock_return = -0.024 + group as f64 * 0.016 + 0.001 * (offset as f64).sin();
                let bond_previous = 100.0 + index as f64;
                let stock_previous = 50.0 + group as f64;
                ctx.price_rows.push(TypedFactorCrossAssetPriceRow {
                    trade_date: bssrc_cache_day(offset),
                    code: (*code).to_string(),
                    exchange_code: String::new(),
                    prev_close_price: bond_previous,
                    close_price: bond_previous * bond_return.exp(),
                    amount: f64::NAN,
                });
                ctx.base_rows.push(TypedFactorCrossAssetBaseRow {
                    trade_date: bssrc_cache_day(offset),
                    code: (*code).to_string(),
                    exchange_code: String::new(),
                    stock_code: format!("60000{}.SH", group + 1),
                    stk_prev_close_price: stock_previous,
                    stk_close_price: stock_previous * stock_return.exp(),
                });
            }
        }
        ctx
    }

    #[test]
    fn bssrc_score_day_cache_matches_direct_reference_across_metrics_and_fail_closed_edges() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let ctx = bssrc_cache_context();
            let metrics = [
                BssrcMetric::UpperAlignment,
                BssrcMetric::Correlation,
                BssrcMetric::LowerAlignment,
            ];
            let codes = ["110001.SH", "110002.SH", "110003.SH", "110004.SH"];
            let requested = codes
                .iter()
                .flat_map(|code| {
                    metrics
                        .iter()
                        .copied()
                        .map(move |metric| (metric, (*code).to_string()))
                })
                .collect::<Vec<_>>();
            let prepared = prepare_bssrc_values(py, &ctx, &requested).unwrap();
            let mut finite = 0usize;
            for code in codes {
                for metric in metrics {
                    let expected = exact_bssrc_metric(py, &ctx, code, metric).unwrap();
                    let actual = prepared.lookup(metric, code).unwrap_or(f64::NAN);
                    finite += usize::from(expected.is_finite());
                    assert_bitwise_or_nan(actual, expected);
                }
            }
            assert!(finite > 0, "fixture must exercise non-missing BSSRC values");

            // Missing terminal mapping is a per-code NaN, while an anchor
            // mismatch invalidates the entire score-day cache.
            let mut missing_terminal = ctx.clone();
            missing_terminal
                .base_rows
                .iter_mut()
                .find(|row| row.code == "110001.SH" && row.trade_date == bssrc_cache_day(64))
                .expect("seeded target terminal row")
                .stock_code
                .clear();
            let prepared = prepare_bssrc_values(py, &missing_terminal, &requested).unwrap();
            for metric in metrics {
                assert_bitwise_or_nan(
                    prepared.lookup(metric, "110001.SH").unwrap_or(f64::NAN),
                    exact_bssrc_metric(py, &missing_terminal, "110001.SH", metric).unwrap(),
                );
            }
            let mut anchor_mismatch = ctx.clone();
            anchor_mismatch
                .base_rows
                .retain(|row| row.trade_date != bssrc_cache_day(64));
            let prepared = prepare_bssrc_values(py, &anchor_mismatch, &requested).unwrap();
            for metric in metrics {
                assert_bitwise_or_nan(
                    prepared.lookup(metric, "110001.SH").unwrap_or(f64::NAN),
                    exact_bssrc_metric(py, &anchor_mismatch, "110001.SH", metric).unwrap(),
                );
            }

            // A conflicting shared-underlying return remains a global error,
            // even when the output code itself is otherwise valid.
            let mut inconsistent = ctx.clone();
            inconsistent
                .base_rows
                .iter_mut()
                .find(|row| row.code == "110002.SH" && row.trade_date == bssrc_cache_day(8))
                .expect("seeded shared-underlying row")
                .stk_close_price *= 1.05;
            let direct_error =
                exact_bssrc_metric(py, &inconsistent, "110001.SH", BssrcMetric::Correlation)
                    .expect_err("direct reference must fail globally");
            let cache_error = prepare_bssrc_values(
                py,
                &inconsistent,
                &[(BssrcMetric::Correlation, "110001.SH".to_string())],
            )
            .expect_err("cache must preserve global failure");
            assert_eq!(cache_error.to_string(), direct_error.to_string());
        });
    }

    #[test]
    fn r88_all_27_factor_signal_family_pairs_are_exact_and_advertised() {
        let expected = [
            (
                "factor_mining_daily_asymmetric_equity_beta_v1",
                "bsab_upside_beta60",
                "prior_asymmetric_equity_beta",
            ),
            (
                "factor_mining_daily_asymmetric_equity_beta_v1",
                "bsab_downside_beta60",
                "prior_asymmetric_equity_beta",
            ),
            (
                "factor_mining_daily_bond_stock_copula_tail_dependence_v1",
                "bsct_upper_tail_dependence60",
                "prior_bond_stock_copula_tail_dependence",
            ),
            (
                "factor_mining_daily_observable_seasoning_v1",
                "osa_terminal_amount_streak60",
                "prior_observable_market_seasoning",
            ),
            (
                "factor_mining_daily_relative_rank_flow_coupling_v2",
                "drrc_return_trade_size_rank_spearman60",
                "prior_relative_return_flow_rank_coupling",
            ),
            (
                "factor_mining_daily_relative_rank_tail_contradiction_v1",
                "drrq_return_amount_opposite_tail_excess60",
                "prior_relative_return_flow_tail_contradiction",
            ),
            (
                "factor_mining_daily_twap_microstructure_v1",
                "dtwm_session_afternoon_late_log_slope",
                "prior_session_rotation_microstructure",
            ),
            (
                "factor_mining_intraday_expansion_v1",
                "exp_rotation_segment_return_dispersion",
                "clock_time_rotation",
            ),
            (
                "factor_mining_intraday_expansion_v1",
                "exp_exec_amount_concentration_impact",
                "execution_price_dispersion",
            ),
            (
                "factor_mining_intraday_expansion_v1",
                "exp_noise_median_mean_abs_return_ratio",
                "multiscale_noise_variance",
            ),
            (
                "factor_mining_intraday_expansion_v1",
                "exp_noise_variance_ratio_2",
                "multiscale_noise_variance",
            ),
            (
                "factor_mining_intraday_expansion_v1",
                "exp_noise_variance_ratio_5",
                "multiscale_noise_variance",
            ),
            (
                "factor_mining_intraday_expansion_v1",
                "exp_stick_quote_update_rate",
                "quote_trade_stickiness",
            ),
            (
                "factor_mining_intraday_execution_discreteness_v1",
                "execdisc_direction_reversal_rate",
                "execution_nonzero_direction_topology",
            ),
            (
                "factor_mining_intraday_execution_discreteness_v1",
                "execdisc_step_multiplicity_entropy",
                "execution_step_multiplicity_geometry",
            ),
            (
                "factor_mining_intraday_catalog_v1",
                "book_quote_dislocation",
                "intraday_quote_resilience",
            ),
            (
                "factor_mining_hybrid_catalog_v1",
                "hybrid_current_range_vs_hist_twap_curve",
                "hybrid_execution_curve",
            ),
            (
                "factor_mining_hybrid_catalog_v1",
                "hybrid_current_flow_vs_hist_overnight_response",
                "hybrid_execution_curve",
            ),
            (
                "factor_mining_intraday_joint_state_v1",
                "joint_tail_range_coexpansion",
                "joint_tail_cojump_containment",
            ),
            (
                "factor_mining_intraday_joint_state_v1",
                "joint_tail_signed_cojump",
                "joint_tail_cojump_containment",
            ),
            (
                "factor_mining_intraday_joint_state_v1",
                "joint_tail_terminal_location_coshock",
                "joint_tail_cojump_containment",
            ),
            (
                "factor_mining_intraday_transmission_response_v1",
                "itr_stock_shock_same_bin_directional_agreement",
                "intraday_stock_shock_directional_response",
            ),
            (
                "factor_mining_underlying_cohort_distribution_v1",
                "ucd_peer_stock_return_dispersion1",
                "underlying_state_distribution",
            ),
            (
                "factor_mining_intraday_state_gated_microstructure_v1",
                "isgm_stockvol_trade_quote_clock_center_gap",
                "stockvol_gated_trade_quote_clock_decoupling",
            ),
            (
                "factor_mining_quote_geometry_microprice_v1",
                "qgeo_micro_last_next_return_sign_alignment",
                "microprice_last_execution_alignment",
            ),
            (
                "factor_mining_structural_neighborhood_v1",
                "sng_peer_return_dispersion1",
                "structural_neighborhood_geometry",
            ),
            (
                "factor_mining_cross_sectional_microstructure_neighborhood_v1",
                "csn_pql_churn_neighbor_gap",
                "csn_passive_queue_local_dislocation",
            ),
        ];
        assert_eq!(expected.len(), 27);
        for (factor, signal, family) in expected {
            assert_eq!(r88_expected_family(factor, signal), Some(family));
            assert!(is_supported_typed_spec(factor, Some(signal)));
            assert!(validate_r88_specs(&[KernelSpec {
                factor: factor.to_string(),
                signal: signal.to_string(),
                family: family.to_string(),
                output_col: signal.to_string(),
            }])
            .is_ok());
        }
    }

    #[test]
    fn r88_rejects_wrong_family_and_cross_family_signal_reuse() {
        let wrong_family = KernelSpec {
            factor: "factor_mining_intraday_expansion_v1".to_string(),
            signal: "exp_noise_variance_ratio_2".to_string(),
            family: "clock_time_rotation".to_string(),
            output_col: "candidate".to_string(),
        };
        assert!(validate_r88_specs(&[wrong_family]).is_err());
        let crossed_signal = KernelSpec {
            factor: "factor_mining_daily_asymmetric_equity_beta_v1".to_string(),
            signal: "bsct_upper_tail_dependence60".to_string(),
            family: "prior_bond_stock_copula_tail_dependence".to_string(),
            output_col: "candidate".to_string(),
        };
        assert!(validate_r88_specs(&[crossed_signal]).is_err());
    }
}
