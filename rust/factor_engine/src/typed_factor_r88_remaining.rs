//! Strict-PIT, unregistered Rust kernels for the remaining eleven R88
//! research factors.
//!
//! This module is intentionally **not** named from `lib.rs` and does not
//! depend on the typed-factor dispatcher.  Adding it therefore cannot change
//! the loaded Python extension, an active factor profile, a model input, or a
//! live path.  A future research-only dispatcher hook must construct
//! [`TypedFactorR88RemainingContext`] from its physical score-day inputs and
//! preserve the exact signal/family pairs exposed by
//! [`TypedFactorR88RemainingSignal`].
//!
//! ## Point-in-time contract
//!
//! * Daily records are filtered with `trade_date < score_date` before every
//!   anchor, join, rank, or historical calculation.
//! * Intraday records must carry their physical date.  They are retained only
//!   for `trade_date == score_date`, a continuous auction session, and no
//!   later than local `14:29:00`.  The comparison deliberately uses
//!   pandas-compatible microsecond flooring: `14:29:00.000000999` is visible,
//!   while `14:29:00.000001000`, `14:29:30`, and `14:30` are not.
//! * The legacy Python hybrid catalogue historically allowed data through
//!   14:30.  These Rust research kernels intentionally prefilter to the
//!   project-wide strict 14:29 boundary; parity tests must apply the same
//!   prefilter to an old Python reference before comparing it.
//!
//! Missing / malformed data fails closed to `NaN`; duplicate strict-prior
//! daily keys and ambiguous maps are explicit errors, matching the Python
//! factor families rather than silently selecting a row.

use chrono::NaiveDate;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const EPS: f64 = 1e-12;
const NS_PER_SECOND: i64 = 1_000_000_000;
const NS_PER_MICROSECOND: i64 = 1_000;
const NS_PER_MINUTE: i64 = 60 * NS_PER_SECOND;
const NS_PER_DAY: i64 = 86_400 * NS_PER_SECOND;

const MORNING_START: i64 = clock_ns(9, 30, 0, 0);
const MORNING_END: i64 = clock_ns(11, 30, 0, 0);
const AFTERNOON_START: i64 = clock_ns(13, 0, 0, 0);
const LATE_START: i64 = clock_ns(13, 30, 0, 0);
const STRICT_CUTOFF: i64 = clock_ns(14, 29, 0, 0);

const MIN_HYBRID_ROWS: usize = 5;
const HYBRID_HISTORY_WINDOW: usize = 20;
const HYBRID_HISTORY_MIN: usize = 15;
const JOINT_MIN_ROWS: usize = 3;
const ITR_EVENT_BIN: i64 = 5 * NS_PER_MINUTE;
const ITR_MIN_COMMON_RETURNS: usize = 12;
const ITR_MIN_SHOCKS: usize = 4;
const ISGM_MIN_CROSS_SECTION: usize = 30;
const ISGM_MIN_ENDPOINTS: usize = 12;
const ISGM_MIN_QUOTE_EVENTS: usize = 4;
const ISGM_MIN_TRADE_EVENTS: usize = 6;
const QGEO_MIN_ROWS: usize = 8;
const QGEO_MIN_PAIRS: usize = 4;
const PEER_COUNT: usize = 7;
const PEER_MIN: usize = 3;
const CSN_MIN_PATH_ROWS: usize = 12;
const CSN_MIN_CROSS_SECTION: usize = 30;
const CSN_NEIGHBOR_COUNT: usize = 12;
const ISGM_QUOTE_REL_TOL: f64 = 1e-10;
const CONTINUOUS_SECONDS: f64 = 210.0 * 60.0;

/// Make a local Asia/Shanghai clock value expressed as nanoseconds after
/// midnight.  It is public so isolated tests and a future adapter can avoid
/// repeating time arithmetic.
pub const fn clock_ns(hour: i64, minute: i64, second: i64, nanos: i64) -> i64 {
    ((hour * 60 * 60 + minute * 60 + second) * NS_PER_SECOND) + nanos
}

/// Project-wide strict 14:29 visibility with pandas `Timestamp.dt.time`
/// microsecond-resolution comparison semantics.
pub fn strict_1429_visible(time_ns: i64) -> bool {
    let python_clock = time_ns.div_euclid(NS_PER_MICROSECOND) * NS_PER_MICROSECOND;
    (0..NS_PER_DAY).contains(&python_clock) && python_clock <= STRICT_CUTOFF
}

fn continuous_session(time_ns: i64) -> Option<u8> {
    let clock = time_ns.div_euclid(NS_PER_MICROSECOND) * NS_PER_MICROSECOND;
    if (MORNING_START..=MORNING_END).contains(&clock) {
        Some(0)
    } else if (AFTERNOON_START..=STRICT_CUTOFF).contains(&clock) {
        Some(1)
    } else {
        None
    }
}

fn canonical_exchange(value: &str) -> String {
    match value.trim().to_ascii_uppercase().as_str() {
        "XSHG" | "SHSE" => "SH".to_string(),
        "XSHE" | "SZSE" => "SZ".to_string(),
        "BSE" | "BJSE" => "BJ".to_string(),
        other => other.to_string(),
    }
}

fn valid_exchange(value: &str) -> bool {
    matches!(value, "SH" | "SZ" | "BJ")
}

/// Exact bond-code handling used by the Python research families: a bare code
/// needs an explicit valid exchange from the source record.  A caller-provided
/// panel code normally already includes its suffix.
fn canonical_market_code(value: &str, exchange: &str) -> String {
    let mut text = value.trim().to_ascii_uppercase();
    if text.is_empty() || matches!(text.as_str(), "NAN" | "NONE" | "<NA>") {
        return String::new();
    }
    if text.ends_with(".0") {
        text.truncate(text.len() - 2);
    }
    if let Some((bare, suffix)) = text.rsplit_once('.') {
        let suffix = canonical_exchange(suffix);
        if !bare.is_empty() && valid_exchange(&suffix) {
            return format!("{bare}.{suffix}");
        }
    }
    let suffix = canonical_exchange(exchange);
    if valid_exchange(&suffix) {
        format!("{text}.{suffix}")
    } else {
        String::new()
    }
}

/// The joint-state Python family permits an unqualified underlying stock only
/// when its leading exchange digit makes the mapping unambiguous.
fn canonical_stock_code(value: &str) -> String {
    let mut text = value.trim().to_ascii_uppercase();
    if text.is_empty() || matches!(text.as_str(), "NAN" | "NONE" | "<NA>") {
        return String::new();
    }
    if text.ends_with(".0") {
        text.truncate(text.len() - 2);
    }
    if let Some((bare, suffix)) = text.rsplit_once('.') {
        let suffix = canonical_exchange(suffix);
        return if !bare.is_empty() && valid_exchange(&suffix) {
            format!("{bare}.{suffix}")
        } else {
            String::new()
        };
    }
    if !text.chars().all(|value| value.is_ascii_digit()) {
        return String::new();
    }
    while text.len() < 6 {
        text.insert(0, '0');
    }
    if text.len() != 6 {
        return String::new();
    }
    let suffix = match text.as_bytes()[0] {
        b'6' => "SH",
        b'0' | b'3' => "SZ",
        b'4' | b'8' => "BJ",
        _ => return String::new(),
    };
    format!("{text}.{suffix}")
}

fn finite(value: f64) -> bool {
    value.is_finite()
}

fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if finite(numerator) && finite(denominator) && denominator.abs() > EPS {
        let value = numerator / denominator;
        if finite(value) {
            value
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    }
}

fn safe_log_return(current: f64, previous: f64) -> f64 {
    if finite(current) && finite(previous) && current > EPS && previous > EPS {
        let value = (current / previous).ln();
        if finite(value) {
            value
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    }
}

fn sanitize(value: f64) -> f64 {
    if finite(value) {
        value
    } else {
        f64::NAN
    }
}

#[derive(Clone, Debug)]
pub struct R88RemainingIntradayRow {
    /// Physical exchange date, not a rolling-panel label.
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    /// Asia/Shanghai local time after midnight in nanoseconds.
    pub time_ns: i64,
    /// Stable source-order tie breaker after timestamp.
    pub seq: i64,
    pub last: f64,
    /// Cumulative session fields where used by the Python source family.
    pub volume: f64,
    pub amount: f64,
    pub num_trades: f64,
    /// L1--L5 arrays are required for the quote-geometry contract.  Factors
    /// that need only L1 read index zero but retain the full validation.
    pub ask_price: [f64; 5],
    pub bid_price: [f64; 5],
    pub ask_volume: [f64; 5],
    pub bid_volume: [f64; 5],
}

#[derive(Clone, Debug)]
pub struct R88RemainingDailyPriceRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    pub prev_close_price: f64,
    pub close_price: f64,
    #[allow(dead_code)]
    pub amount: f64,
}

#[derive(Clone, Debug)]
pub struct R88RemainingDailyBaseRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    /// Required by the hybrid current-flow/history interaction.
    pub cb_amount: f64,
    /// Required by the strict T-1 stock mapping and underlying cohort state.
    pub stock_code: String,
    pub stock_close_price: f64,
    pub stock_volatility: f64,
    pub stk_amount: f64,
    /// Required by the structural-neighborhood state.
    pub year_to_mat: f64,
    pub duration: f64,
    pub bond_prem_ratio: f64,
    pub ytm: f64,
    pub remain_size: f64,
}

#[derive(Clone, Debug)]
pub struct R88RemainingDailyTwapRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    pub twap_0930_0935: f64,
    pub twap_0935_1000: f64,
    pub twap_1300_1330: f64,
    pub twap_1400_1430: f64,
    pub twap_1430_1442: f64,
}

/// The ITR Python source consumes a supplied point-in-time map, not a daily
/// inferred map.  Duplicate valid bond keys are treated as ambiguity.
#[derive(Clone, Debug)]
pub struct R88RemainingBondStockMapRow {
    pub code: String,
    pub stock_code: String,
    pub as_of_date: NaiveDate,
}

/// Complete raw, research-only material for the eleven kernels.  `panel_codes`
/// is the score-day output universe used for cross-sectional cohort/rank
/// factors; it must contain explicit exchange-qualified bond codes.
#[derive(Clone, Debug)]
pub struct TypedFactorR88RemainingContext {
    pub score_date: NaiveDate,
    pub panel_codes: Vec<String>,
    pub intraday_rows: Vec<R88RemainingIntradayRow>,
    pub daily_price_rows: Vec<R88RemainingDailyPriceRow>,
    pub daily_base_rows: Vec<R88RemainingDailyBaseRow>,
    pub daily_twap_rows: Vec<R88RemainingDailyTwapRow>,
    pub bond_stock_map: Vec<R88RemainingBondStockMapRow>,
}

impl TypedFactorR88RemainingContext {
    pub fn new(score_date: NaiveDate) -> Self {
        Self {
            score_date,
            panel_codes: Vec::new(),
            intraday_rows: Vec::new(),
            daily_price_rows: Vec::new(),
            daily_base_rows: Vec::new(),
            daily_twap_rows: Vec::new(),
            bond_stock_map: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypedFactorR88RemainingError {
    DuplicateStrictPriorDate {
        source: &'static str,
        code: String,
        trade_date: NaiveDate,
    },
    AmbiguousBondStockMap {
        code: String,
    },
    UnknownSignal(String),
}

impl fmt::Display for TypedFactorR88RemainingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStrictPriorDate {
                source,
                code,
                trade_date,
            } => write!(
                formatter,
                "typed_factor R88 remaining {source} has duplicate strict-prior row for {code} on {trade_date}"
            ),
            Self::AmbiguousBondStockMap { code } => write!(
                formatter,
                "typed_factor R88 remaining has ambiguous point-in-time bond-stock mapping for {code}"
            ),
            Self::UnknownSignal(signal) => {
                write!(formatter, "unknown typed_factor R88 remaining signal: {signal}")
            }
        }
    }
}

impl Error for TypedFactorR88RemainingError {}

/// Exact signal names owned by this isolated file.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TypedFactorR88RemainingSignal {
    HybridCurrentRangeVsHistTwapCurve,
    HybridCurrentFlowVsHistOvernightResponse,
    JointTailRangeCoexpansion,
    JointTailSignedCojump,
    JointTailTerminalLocationCoshock,
    ItrStockShockSameBinDirectionalAgreement,
    UcdPeerStockReturnDispersion1,
    IsgmStockvolTradeQuoteClockCenterGap,
    QgeoMicroLastNextReturnSignAlignment,
    SngPeerReturnDispersion1,
    CsnPqlChurnNeighborGap,
}

impl TypedFactorR88RemainingSignal {
    pub fn parse(signal: &str) -> Option<Self> {
        match signal {
            "hybrid_current_range_vs_hist_twap_curve" => {
                Some(Self::HybridCurrentRangeVsHistTwapCurve)
            }
            "hybrid_current_flow_vs_hist_overnight_response" => {
                Some(Self::HybridCurrentFlowVsHistOvernightResponse)
            }
            "joint_tail_range_coexpansion" => Some(Self::JointTailRangeCoexpansion),
            "joint_tail_signed_cojump" => Some(Self::JointTailSignedCojump),
            "joint_tail_terminal_location_coshock" => Some(Self::JointTailTerminalLocationCoshock),
            "itr_stock_shock_same_bin_directional_agreement" => {
                Some(Self::ItrStockShockSameBinDirectionalAgreement)
            }
            "ucd_peer_stock_return_dispersion1" => Some(Self::UcdPeerStockReturnDispersion1),
            "isgm_stockvol_trade_quote_clock_center_gap" => {
                Some(Self::IsgmStockvolTradeQuoteClockCenterGap)
            }
            "qgeo_micro_last_next_return_sign_alignment" => {
                Some(Self::QgeoMicroLastNextReturnSignAlignment)
            }
            "sng_peer_return_dispersion1" => Some(Self::SngPeerReturnDispersion1),
            "csn_pql_churn_neighbor_gap" => Some(Self::CsnPqlChurnNeighborGap),
            _ => None,
        }
    }

    pub fn signal(self) -> &'static str {
        match self {
            Self::HybridCurrentRangeVsHistTwapCurve => "hybrid_current_range_vs_hist_twap_curve",
            Self::HybridCurrentFlowVsHistOvernightResponse => {
                "hybrid_current_flow_vs_hist_overnight_response"
            }
            Self::JointTailRangeCoexpansion => "joint_tail_range_coexpansion",
            Self::JointTailSignedCojump => "joint_tail_signed_cojump",
            Self::JointTailTerminalLocationCoshock => "joint_tail_terminal_location_coshock",
            Self::ItrStockShockSameBinDirectionalAgreement => {
                "itr_stock_shock_same_bin_directional_agreement"
            }
            Self::UcdPeerStockReturnDispersion1 => "ucd_peer_stock_return_dispersion1",
            Self::IsgmStockvolTradeQuoteClockCenterGap => {
                "isgm_stockvol_trade_quote_clock_center_gap"
            }
            Self::QgeoMicroLastNextReturnSignAlignment => {
                "qgeo_micro_last_next_return_sign_alignment"
            }
            Self::SngPeerReturnDispersion1 => "sng_peer_return_dispersion1",
            Self::CsnPqlChurnNeighborGap => "csn_pql_churn_neighbor_gap",
        }
    }

    pub fn family(self) -> &'static str {
        match self {
            Self::HybridCurrentRangeVsHistTwapCurve
            | Self::HybridCurrentFlowVsHistOvernightResponse => "hybrid_execution_curve",
            Self::JointTailRangeCoexpansion
            | Self::JointTailSignedCojump
            | Self::JointTailTerminalLocationCoshock => "joint_tail_cojump_containment",
            Self::ItrStockShockSameBinDirectionalAgreement => {
                "intraday_stock_shock_directional_response"
            }
            Self::UcdPeerStockReturnDispersion1 => "underlying_state_distribution",
            Self::IsgmStockvolTradeQuoteClockCenterGap => {
                "stockvol_gated_trade_quote_clock_decoupling"
            }
            Self::QgeoMicroLastNextReturnSignAlignment => "microprice_last_execution_alignment",
            Self::SngPeerReturnDispersion1 => "structural_neighborhood_geometry",
            Self::CsnPqlChurnNeighborGap => "csn_passive_queue_local_dislocation",
        }
    }

    pub fn source_contract(self) -> &'static str {
        match self {
            Self::HybridCurrentRangeVsHistTwapCurve => {
                "intraday(last,volume,amount,num_trades); daily_twap(twap_0930_0935,twap_0935_1000,twap_1300_1330,twap_1400_1430,twap_1430_1442)"
            }
            Self::HybridCurrentFlowVsHistOvernightResponse => {
                "intraday(last,volume,amount,num_trades); daily_base(cb_amount); daily_twap(twap_0930_0935,twap_1430_1442)"
            }
            Self::JointTailRangeCoexpansion
            | Self::JointTailSignedCojump
            | Self::JointTailTerminalLocationCoshock => {
                "intraday(last); daily_price(close_price); daily_base(stock_code)"
            }
            Self::ItrStockShockSameBinDirectionalAgreement => {
                "intraday(last); bond_stock_map(as_of_date,code,stock_code)"
            }
            Self::UcdPeerStockReturnDispersion1 => {
                "daily_price(prev_close_price,close_price); daily_base(stock_code,stock_close_price,stock_volatility,stk_amount)"
            }
            Self::IsgmStockvolTradeQuoteClockCenterGap => {
                "intraday(num_trades,ask_price1,bid_price1); daily_price(close_price); daily_base(stock_volatility)"
            }
            Self::QgeoMicroLastNextReturnSignAlignment => {
                "intraday(last,ask_price1..5,bid_price1..5,ask_volume1..5,bid_volume1..5)"
            }
            Self::SngPeerReturnDispersion1 => {
                "daily_price(prev_close_price,close_price); daily_base(year_to_mat,duration,bond_prem_ratio,ytm,remain_size)"
            }
            Self::CsnPqlChurnNeighborGap => {
                "intraday(num_trades,ask_price1,bid_price1,ask_volume1,bid_volume1)"
            }
        }
    }
}

#[derive(Clone, Debug)]
struct PriceRow {
    trade_date: NaiveDate,
    code: String,
    prev_close_price: f64,
    close_price: f64,
}

#[derive(Clone, Debug)]
struct BaseRow {
    trade_date: NaiveDate,
    code: String,
    cb_amount: f64,
    stock_code: String,
    stock_close_price: f64,
    stock_volatility: f64,
    stk_amount: f64,
    year_to_mat: f64,
    duration: f64,
    bond_prem_ratio: f64,
    ytm: f64,
    remain_size: f64,
}

#[derive(Clone, Debug)]
struct TwapRow {
    trade_date: NaiveDate,
    code: String,
    twap_0930_0935: f64,
    twap_0935_1000: f64,
    twap_1300_1330: f64,
    twap_1400_1430: f64,
    twap_1430_1442: f64,
}

fn strict_price_rows(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<Vec<PriceRow>, TypedFactorR88RemainingError> {
    let mut rows: Vec<_> = ctx
        .daily_price_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .filter_map(|row| {
            let code = canonical_market_code(&row.code, &row.exchange_code);
            (!code.is_empty()).then_some(PriceRow {
                trade_date: row.trade_date,
                code,
                prev_close_price: row.prev_close_price,
                close_price: row.close_price,
            })
        })
        .collect();
    rows.sort_by(|left, right| {
        left.trade_date
            .cmp(&right.trade_date)
            .then_with(|| left.code.cmp(&right.code))
    });
    for pair in rows.windows(2) {
        if pair[0].trade_date == pair[1].trade_date && pair[0].code == pair[1].code {
            return Err(TypedFactorR88RemainingError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_price",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

fn strict_base_rows(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<Vec<BaseRow>, TypedFactorR88RemainingError> {
    let mut rows: Vec<_> = ctx
        .daily_base_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .filter_map(|row| {
            let code = canonical_market_code(&row.code, &row.exchange_code);
            (!code.is_empty()).then_some(BaseRow {
                trade_date: row.trade_date,
                code,
                cb_amount: row.cb_amount,
                stock_code: row.stock_code.clone(),
                stock_close_price: row.stock_close_price,
                stock_volatility: row.stock_volatility,
                stk_amount: row.stk_amount,
                year_to_mat: row.year_to_mat,
                duration: row.duration,
                bond_prem_ratio: row.bond_prem_ratio,
                ytm: row.ytm,
                remain_size: row.remain_size,
            })
        })
        .collect();
    rows.sort_by(|left, right| {
        left.trade_date
            .cmp(&right.trade_date)
            .then_with(|| left.code.cmp(&right.code))
    });
    for pair in rows.windows(2) {
        if pair[0].trade_date == pair[1].trade_date && pair[0].code == pair[1].code {
            return Err(TypedFactorR88RemainingError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_base",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

fn strict_twap_rows(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<Vec<TwapRow>, TypedFactorR88RemainingError> {
    let mut rows: Vec<_> = ctx
        .daily_twap_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .filter_map(|row| {
            let code = canonical_market_code(&row.code, &row.exchange_code);
            (!code.is_empty()).then_some(TwapRow {
                trade_date: row.trade_date,
                code,
                twap_0930_0935: row.twap_0930_0935,
                twap_0935_1000: row.twap_0935_1000,
                twap_1300_1330: row.twap_1300_1330,
                twap_1400_1430: row.twap_1400_1430,
                twap_1430_1442: row.twap_1430_1442,
            })
        })
        .collect();
    rows.sort_by(|left, right| {
        left.trade_date
            .cmp(&right.trade_date)
            .then_with(|| left.code.cmp(&right.code))
    });
    for pair in rows.windows(2) {
        if pair[0].trade_date == pair[1].trade_date && pair[0].code == pair[1].code {
            return Err(TypedFactorR88RemainingError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_twap",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

fn normalized_panel_code(raw_panel_code: &str) -> String {
    canonical_market_code(raw_panel_code, "")
}

fn physical_rows_for_code(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Vec<R88RemainingIntradayRow> {
    let code = normalized_panel_code(raw_panel_code);
    if code.is_empty() {
        return Vec::new();
    }
    let mut rows: Vec<_> = ctx
        .intraday_rows
        .iter()
        .filter(|row| row.trade_date == ctx.score_date)
        .filter(|row| continuous_session(row.time_ns).is_some())
        .filter(|row| canonical_market_code(&row.code, &row.exchange_code) == code)
        .cloned()
        .collect();
    rows.sort_by(|left, right| {
        left.time_ns
            .cmp(&right.time_ns)
            .then_with(|| left.seq.cmp(&right.seq))
    });
    rows
}

fn physical_rows_by_code(
    ctx: &TypedFactorR88RemainingContext,
) -> BTreeMap<String, Vec<R88RemainingIntradayRow>> {
    let mut out: BTreeMap<String, Vec<R88RemainingIntradayRow>> = BTreeMap::new();
    for row in &ctx.intraday_rows {
        if row.trade_date != ctx.score_date || continuous_session(row.time_ns).is_none() {
            continue;
        }
        let code = canonical_market_code(&row.code, &row.exchange_code);
        if !code.is_empty() {
            out.entry(code).or_default().push(row.clone());
        }
    }
    for rows in out.values_mut() {
        rows.sort_by(|left, right| {
            left.time_ns
                .cmp(&right.time_ns)
                .then_with(|| left.seq.cmp(&right.seq))
        });
    }
    out
}

/// Hybrid-catalogue paths historically used all physical score-day snapshots
/// through the score cutoff, rather than imposing a continuous-session mask.
/// Keep that Python-family distinction while replacing its former 14:30 bound
/// with the strict project 14:29 boundary.
fn strict_visible_rows_for_code(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Vec<R88RemainingIntradayRow> {
    let code = normalized_panel_code(raw_panel_code);
    if code.is_empty() {
        return Vec::new();
    }
    let mut rows: Vec<_> = ctx
        .intraday_rows
        .iter()
        .filter(|row| row.trade_date == ctx.score_date && strict_1429_visible(row.time_ns))
        .filter(|row| canonical_market_code(&row.code, &row.exchange_code) == code)
        .cloned()
        .collect();
    rows.sort_by(|left, right| {
        left.time_ns
            .cmp(&right.time_ns)
            .then_with(|| left.seq.cmp(&right.seq))
    });
    rows
}

fn strict_visible_rows_by_code(
    ctx: &TypedFactorR88RemainingContext,
) -> BTreeMap<String, Vec<R88RemainingIntradayRow>> {
    let mut out: BTreeMap<String, Vec<R88RemainingIntradayRow>> = BTreeMap::new();
    for row in &ctx.intraday_rows {
        if row.trade_date != ctx.score_date || !strict_1429_visible(row.time_ns) {
            continue;
        }
        let code = canonical_market_code(&row.code, &row.exchange_code);
        if !code.is_empty() {
            out.entry(code).or_default().push(row.clone());
        }
    }
    for rows in out.values_mut() {
        rows.sort_by(|left, right| {
            left.time_ns
                .cmp(&right.time_ns)
                .then_with(|| left.seq.cmp(&right.seq))
        });
    }
    out
}

fn target_base_rows<'a>(rows: &'a [BaseRow], code: &str) -> Vec<&'a BaseRow> {
    rows.iter().filter(|row| row.code == code).collect()
}

fn target_twap_rows<'a>(rows: &'a [TwapRow], code: &str) -> Vec<&'a TwapRow> {
    rows.iter().filter(|row| row.code == code).collect()
}

fn finite_tail_mean(values: impl IntoIterator<Item = f64>, count: usize, min_count: usize) -> f64 {
    let finite_values: Vec<f64> = values.into_iter().filter(|value| finite(*value)).collect();
    if finite_values.len() < min_count {
        return f64::NAN;
    }
    let tail_start = finite_values.len().saturating_sub(count);
    let tail = &finite_values[tail_start..];
    let value = tail.iter().sum::<f64>() / tail.len() as f64;
    sanitize(value)
}

fn last_finite(values: impl IntoIterator<Item = f64>) -> f64 {
    let values: Vec<_> = values.into_iter().collect();
    values
        .into_iter()
        .rev()
        .find(|value| finite(*value))
        .unwrap_or(f64::NAN)
}

fn incremental(values: &[f64]) -> Option<Vec<f64>> {
    if values.len() < 3 || values.iter().any(|value| !finite(*value)) {
        return None;
    }
    let maximum = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let tolerance = 1e-8_f64.max(maximum * 1e-10);
    let increments: Vec<f64> = values.windows(2).map(|pair| pair[1] - pair[0]).collect();
    if increments.iter().any(|value| *value < -tolerance)
        || !increments.iter().any(|value| *value > tolerance)
    {
        None
    } else {
        Some(increments)
    }
}

fn hybrid_current_range(rows: &[R88RemainingIntradayRow]) -> f64 {
    if rows.len() < MIN_HYBRID_ROWS || rows.iter().any(|row| !finite(row.last) || row.last <= EPS) {
        return f64::NAN;
    }
    let first = rows[0].last;
    let high = rows
        .iter()
        .map(|row| row.last)
        .fold(f64::NEG_INFINITY, f64::max);
    let low = rows
        .iter()
        .map(|row| row.last)
        .fold(f64::INFINITY, f64::min);
    safe_div(high - low, first)
}

/// Exact `__intra_amount_total` prerequisite from the Python hybrid source:
/// the whole cumulative volume/amount/trade-count path must be valid, and
/// interval amount is summed only where all three per-interval flow criteria
/// hold.  This intentionally does not substitute a raw terminal amount.
fn hybrid_current_total_amount(rows: &[R88RemainingIntradayRow]) -> f64 {
    if rows.len() < MIN_HYBRID_ROWS || rows.iter().any(|row| !finite(row.last) || row.last <= EPS) {
        return f64::NAN;
    }
    let volume: Vec<f64> = rows.iter().map(|row| row.volume).collect();
    let amount: Vec<f64> = rows.iter().map(|row| row.amount).collect();
    let deals: Vec<f64> = rows.iter().map(|row| row.num_trades).collect();
    let (Some(volume_inc), Some(amount_inc), Some(deal_inc)) = (
        incremental(&volume),
        incremental(&amount),
        incremental(&deals),
    ) else {
        return f64::NAN;
    };
    let valid: Vec<bool> = volume_inc
        .iter()
        .zip(amount_inc.iter())
        .zip(deal_inc.iter())
        .map(|((volume, amount), deal)| *volume > 0.0 && *amount > 0.0 && *deal >= 0.0)
        .collect();
    if valid.iter().filter(|item| **item).count() < 3 {
        return f64::NAN;
    }
    sanitize(
        amount_inc
            .iter()
            .zip(valid.iter())
            .filter_map(|(value, keep)| (*keep).then_some(*value))
            .sum(),
    )
}

fn ratio_minus_one(numerator: f64, denominator: f64) -> f64 {
    if finite(numerator) && finite(denominator) && numerator > 0.0 && denominator > 0.0 {
        sanitize(numerator / denominator - 1.0)
    } else {
        f64::NAN
    }
}

fn hist_twap_curve_mean20(rows: &[&TwapRow]) -> f64 {
    let values = rows.iter().map(|row| {
        let morning = ratio_minus_one(row.twap_0935_1000, row.twap_0930_0935);
        let afternoon = ratio_minus_one(row.twap_1400_1430, row.twap_1300_1330);
        let late = ratio_minus_one(row.twap_1430_1442, row.twap_1400_1430);
        sanitize(morning + afternoon - 2.0 * late)
    });
    finite_tail_mean(values, HYBRID_HISTORY_WINDOW, HYBRID_HISTORY_MIN)
}

fn hist_overnight_mean20(rows: &[&TwapRow]) -> f64 {
    let mut values = Vec::with_capacity(rows.len());
    let mut prior_late = f64::NAN;
    for row in rows {
        values.push(ratio_minus_one(row.twap_0930_0935, prior_late));
        // Python's `.shift(1)` uses the raw historical ordered field; a
        // malformed current late value therefore cannot be "fixed" by a
        // prior finite-value compaction.
        prior_late = row.twap_1430_1442;
    }
    finite_tail_mean(values, HYBRID_HISTORY_WINDOW, HYBRID_HISTORY_MIN)
}

/// Current strict-14:29 range multiplied by the completed historical
/// 20-observation TWAP curve mean.
pub fn hybrid_current_range_vs_hist_twap_curve(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let code = normalized_panel_code(raw_panel_code);
    if code.is_empty() {
        return Ok(f64::NAN);
    }
    let twap = strict_twap_rows(ctx)?;
    let range = hybrid_current_range(&strict_visible_rows_for_code(ctx, raw_panel_code));
    Ok(sanitize(
        range * hist_twap_curve_mean20(&target_twap_rows(&twap, &code)),
    ))
}

/// Strict-14:29 current-flow / completed-prior `cb_amount` ratio multiplied
/// by the historical 20-observation overnight response mean.
pub fn hybrid_current_flow_vs_hist_overnight_response(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let code = normalized_panel_code(raw_panel_code);
    if code.is_empty() {
        return Ok(f64::NAN);
    }
    let base = strict_base_rows(ctx)?;
    let twap = strict_twap_rows(ctx)?;
    let current_amount =
        hybrid_current_total_amount(&strict_visible_rows_for_code(ctx, raw_panel_code));
    let previous_cb_amount = last_finite(
        target_base_rows(&base, &code)
            .into_iter()
            .map(|row| row.cb_amount),
    );
    let amount_ratio = safe_div(current_amount, previous_cb_amount);
    Ok(sanitize(
        amount_ratio * hist_overnight_mean20(&target_twap_rows(&twap, &code)),
    ))
}

#[derive(Clone, Copy, Debug)]
struct JointTailState {
    late_return: f64,
    tail_range: f64,
    tail_location: f64,
}

impl JointTailState {
    fn nan() -> Self {
        Self {
            late_return: f64::NAN,
            tail_range: f64::NAN,
            tail_location: f64::NAN,
        }
    }
}

fn joint_tail_state(rows: &[R88RemainingIntradayRow]) -> JointTailState {
    // `_asset_summary` first accepts three physical last-price observations,
    // then derives the tail path from 13:30 onward.  Invalid last values are
    // excluded only for this price-path subfamily, as in the Python source.
    let valid: Vec<_> = rows
        .iter()
        .filter(|row| finite(row.last) && row.last > 0.0)
        .collect();
    if valid.len() < JOINT_MIN_ROWS {
        return JointTailState::nan();
    }
    let tail: Vec<_> = valid
        .into_iter()
        .filter(|row| row.time_ns >= LATE_START)
        .collect();
    if tail.len() < JOINT_MIN_ROWS {
        return JointTailState::nan();
    }
    let first = tail[0].last;
    let last = tail[tail.len() - 1].last;
    let high = tail
        .iter()
        .map(|row| row.last)
        .fold(f64::NEG_INFINITY, f64::max);
    let low = tail
        .iter()
        .map(|row| row.last)
        .fold(f64::INFINITY, f64::min);
    let range = high - low;
    JointTailState {
        late_return: safe_div(last - first, first),
        tail_range: safe_div(range, first),
        tail_location: safe_div(last - low, range),
    }
}

/// Strict T-1 stock mapping identical to the joint-state Python family: the
/// global latest daily-price date is the certificate, then price/base need an
/// exact `(date, code)` match before `stock_code` is usable.
fn strict_tminus1_stock_mapping(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<BTreeMap<String, String>, TypedFactorR88RemainingError> {
    let price = strict_price_rows(ctx)?;
    let base = strict_base_rows(ctx)?;
    let Some(anchor) = price.iter().map(|row| row.trade_date).max() else {
        return Ok(BTreeMap::new());
    };
    let price_codes: BTreeSet<_> = price
        .iter()
        .filter(|row| row.trade_date == anchor)
        .map(|row| row.code.clone())
        .collect();
    let mut out = BTreeMap::new();
    for row in base
        .iter()
        .filter(|row| row.trade_date == anchor && price_codes.contains(&row.code))
    {
        let stock = canonical_stock_code(&row.stock_code);
        if !stock.is_empty() {
            out.insert(row.code.clone(), stock);
        }
    }
    Ok(out)
}

fn mapped_stock_rows(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<Option<Vec<R88RemainingIntradayRow>>, TypedFactorR88RemainingError> {
    let code = normalized_panel_code(raw_panel_code);
    if code.is_empty() {
        return Ok(None);
    }
    let mapping = strict_tminus1_stock_mapping(ctx)?;
    let Some(stock_code) = mapping.get(&code) else {
        return Ok(None);
    };
    let rows = physical_rows_by_code(ctx)
        .remove(stock_code)
        .unwrap_or_default();
    Ok((!rows.is_empty()).then_some(rows))
}

/// Strict-14:29 product of mapped bond and stock tail returns.
pub fn joint_tail_signed_cojump(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let bond = joint_tail_state(&physical_rows_for_code(ctx, raw_panel_code));
    let Some(stock_rows) = mapped_stock_rows(ctx, raw_panel_code)? else {
        return Ok(f64::NAN);
    };
    let stock = joint_tail_state(&stock_rows);
    Ok(sanitize(bond.late_return * stock.late_return))
}

/// Strict-14:29 product of mapped bond and stock tail ranges, each normalized
/// by that asset's first valid tail price.
pub fn joint_tail_range_coexpansion(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let bond = joint_tail_state(&physical_rows_for_code(ctx, raw_panel_code));
    let Some(stock_rows) = mapped_stock_rows(ctx, raw_panel_code)? else {
        return Ok(f64::NAN);
    };
    let stock = joint_tail_state(&stock_rows);
    Ok(sanitize(bond.tail_range * stock.tail_range))
}

/// Strict-14:29 tail terminal-location difference multiplied by the signed
/// mapped tail cojump.
pub fn joint_tail_terminal_location_coshock(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let bond = joint_tail_state(&physical_rows_for_code(ctx, raw_panel_code));
    let Some(stock_rows) = mapped_stock_rows(ctx, raw_panel_code)? else {
        return Ok(f64::NAN);
    };
    let stock = joint_tail_state(&stock_rows);
    Ok(sanitize(
        (bond.tail_location - stock.tail_location) * (bond.late_return * stock.late_return),
    ))
}

type JointTailValueMaps = (
    BTreeMap<String, f64>,
    BTreeMap<String, f64>,
    BTreeMap<String, f64>,
);

/// Compute all three joint-tail outputs over the one immutable score-day
/// cross-section.  The direct public functions above intentionally retain the
/// per-code reference path; this helper only moves their shared mapping and
/// group construction outside the output-code loop.
fn joint_tail_value_maps(
    ctx: &TypedFactorR88RemainingContext,
    groups: &BTreeMap<String, Vec<R88RemainingIntradayRow>>,
) -> Result<JointTailValueMaps, TypedFactorR88RemainingError> {
    let mapping = strict_tminus1_stock_mapping(ctx)?;
    let mut signed = BTreeMap::new();
    let mut range = BTreeMap::new();
    let mut location = BTreeMap::new();
    for code in panel_codes(ctx) {
        let Some(stock_code) = mapping.get(&code) else {
            continue;
        };
        let bond = groups
            .get(&code)
            .map(|rows| joint_tail_state(rows))
            .unwrap_or_else(JointTailState::nan);
        let stock = groups
            .get(stock_code)
            .map(|rows| joint_tail_state(rows))
            .unwrap_or_else(JointTailState::nan);
        let cojump = sanitize(bond.late_return * stock.late_return);
        signed.insert(code.clone(), cojump);
        range.insert(code.clone(), sanitize(bond.tail_range * stock.tail_range));
        location.insert(
            code,
            sanitize((bond.tail_location - stock.tail_location) * cojump),
        );
    }
    Ok((signed, range, location))
}

fn point_in_time_map(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<BTreeMap<String, String>, TypedFactorR88RemainingError> {
    let mut out = BTreeMap::new();
    let mut duplicate = BTreeSet::new();
    for row in &ctx.bond_stock_map {
        if row.as_of_date > ctx.score_date {
            continue;
        }
        // Unlike daily-source records, this map contract requires explicit
        // exchange-qualified keys.  An empty exchange argument enforces it.
        let bond = canonical_market_code(&row.code, "");
        let stock = canonical_market_code(&row.stock_code, "");
        if bond.is_empty() || stock.is_empty() {
            continue;
        }
        if out.insert(bond.clone(), stock).is_some() {
            duplicate.insert(bond);
        }
    }
    if let Some(code) = duplicate.into_iter().next() {
        return Err(TypedFactorR88RemainingError::AmbiguousBondStockMap { code });
    }
    Ok(out)
}

fn five_minute_endpoints(rows: &[R88RemainingIntradayRow]) -> Vec<(i64, f64)> {
    let mut bins: BTreeMap<i64, (i64, i64, f64)> = BTreeMap::new();
    for row in rows {
        if !finite(row.last) || row.last <= EPS {
            continue;
        }
        let bin = row.time_ns.div_euclid(ITR_EVENT_BIN) * ITR_EVENT_BIN;
        let replace = bins
            .get(&bin)
            .map(|(time, seq, _)| (row.time_ns, row.seq) >= (*time, *seq))
            .unwrap_or(true);
        if replace {
            bins.insert(bin, (row.time_ns, row.seq, row.last));
        }
    }
    bins.into_iter()
        .map(|(bin, (_, _, last))| (bin, last))
        .collect()
}

fn five_minute_returns(rows: &[R88RemainingIntradayRow]) -> BTreeMap<i64, f64> {
    let endpoints = five_minute_endpoints(rows);
    let mut out = BTreeMap::new();
    for pair in endpoints.windows(2) {
        let (previous_time, previous_last) = pair[0];
        let (time, last) = pair[1];
        out.insert(
            time,
            if time - previous_time == ITR_EVENT_BIN {
                safe_log_return(last, previous_last)
            } else {
                f64::NAN
            },
        );
    }
    out
}

fn percentile_linear(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() || !(0.0..=1.0).contains(&probability) {
        return f64::NAN;
    }
    let position = (sorted.len() - 1) as f64 * probability;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        sorted[low]
    } else {
        sorted[low] + (position - low as f64) * (sorted[high] - sorted[low])
    }
}

/// Shock-conditioned five-minute same-bin stock/bond directional agreement.
/// The map is accepted only through `score_date`; returns are built only from
/// consecutive observed five-minute endpoint bins.
fn itr_stock_shock_same_bin_directional_agreement_from_groups(
    mapping: &BTreeMap<String, String>,
    groups: &BTreeMap<String, Vec<R88RemainingIntradayRow>>,
    bond_code: &str,
) -> f64 {
    let Some(stock_code) = mapping.get(bond_code) else {
        return f64::NAN;
    };
    let bond = five_minute_returns(groups.get(bond_code).map(Vec::as_slice).unwrap_or(&[]));
    let stock = five_minute_returns(groups.get(stock_code).map(Vec::as_slice).unwrap_or(&[]));
    let joint: Vec<(f64, f64)> = bond
        .iter()
        .filter_map(|(time, bond_return)| {
            let stock_return = stock.get(time).copied().unwrap_or(f64::NAN);
            (finite(*bond_return) && finite(stock_return)).then_some((*bond_return, stock_return))
        })
        .collect();
    if joint.len() < ITR_MIN_COMMON_RETURNS {
        return f64::NAN;
    }
    let mut absolute_stock: Vec<_> = joint.iter().map(|(_, stock)| stock.abs()).collect();
    absolute_stock.sort_by(|left, right| left.total_cmp(right));
    let threshold = percentile_linear(&absolute_stock, 0.75);
    if !finite(threshold) || threshold <= EPS {
        return f64::NAN;
    }
    let shocks: Vec<_> = joint
        .into_iter()
        .filter(|(_, stock)| stock.abs() >= threshold)
        .collect();
    if shocks.len() < ITR_MIN_SHOCKS {
        return f64::NAN;
    }
    sanitize(
        shocks
            .iter()
            .map(|(bond, stock)| bond.signum() * stock.signum())
            .sum::<f64>()
            / shocks.len() as f64,
    )
}

pub fn itr_stock_shock_same_bin_directional_agreement(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let bond_code = normalized_panel_code(raw_panel_code);
    if bond_code.is_empty() {
        return Ok(f64::NAN);
    }
    let mapping = point_in_time_map(ctx)?;
    let groups = physical_rows_by_code(ctx);
    Ok(itr_stock_shock_same_bin_directional_agreement_from_groups(
        &mapping, &groups, &bond_code,
    ))
}

#[derive(Clone, Debug)]
struct DailyAnchorState {
    anchor: NaiveDate,
    calendar: Vec<NaiveDate>,
    price_by_code: BTreeMap<String, Vec<PriceRow>>,
    base_by_code: BTreeMap<String, Vec<BaseRow>>,
}

fn daily_anchor_state(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<Option<DailyAnchorState>, TypedFactorR88RemainingError> {
    let price = strict_price_rows(ctx)?;
    let base = strict_base_rows(ctx)?;
    let Some(anchor) = price.iter().map(|row| row.trade_date).max() else {
        return Ok(None);
    };
    if base.is_empty() {
        return Ok(None);
    }
    let mut calendar: Vec<_> = price.iter().map(|row| row.trade_date).collect();
    calendar.sort_unstable();
    calendar.dedup();
    let price_keys: BTreeSet<_> = price
        .iter()
        .map(|row| (row.trade_date, row.code.clone()))
        .collect();
    let mut price_by_code: BTreeMap<String, Vec<PriceRow>> = BTreeMap::new();
    for row in price {
        price_by_code.entry(row.code.clone()).or_default().push(row);
    }
    let mut base_by_code: BTreeMap<String, Vec<BaseRow>> = BTreeMap::new();
    for row in base {
        if price_keys.contains(&(row.trade_date, row.code.clone())) {
            base_by_code.entry(row.code.clone()).or_default().push(row);
        }
    }
    for rows in price_by_code.values_mut() {
        rows.sort_by_key(|row| row.trade_date);
    }
    for rows in base_by_code.values_mut() {
        rows.sort_by_key(|row| row.trade_date);
    }
    Ok(Some(DailyAnchorState {
        anchor,
        calendar,
        price_by_code,
        base_by_code,
    }))
}

fn panel_codes(ctx: &TypedFactorR88RemainingContext) -> Vec<String> {
    let mut out: Vec<_> = ctx
        .panel_codes
        .iter()
        .map(|code| normalized_panel_code(code))
        .filter(|code| !code.is_empty())
        .collect();
    out.sort();
    out.dedup();
    out
}

fn session_index(calendar: &[NaiveDate], day: NaiveDate) -> Option<i64> {
    calendar.binary_search(&day).ok().map(|index| index as i64)
}

fn price_at_anchor<'a>(state: &'a DailyAnchorState, code: &str) -> Option<&'a PriceRow> {
    let row = state.price_by_code.get(code)?.last()?;
    (row.trade_date == state.anchor).then_some(row)
}

fn base_at_anchor<'a>(state: &'a DailyAnchorState, code: &str) -> Option<&'a BaseRow> {
    let row = state.base_by_code.get(code)?.last()?;
    (row.trade_date == state.anchor).then_some(row)
}

fn complete_base_tail<'a>(
    state: &'a DailyAnchorState,
    code: &str,
    count: usize,
) -> Option<Vec<&'a BaseRow>> {
    let rows = state.base_by_code.get(code)?;
    if rows.len() < count {
        return None;
    }
    let tail = &rows[rows.len() - count..];
    let positions: Vec<_> = tail
        .iter()
        .map(|row| session_index(&state.calendar, row.trade_date))
        .collect::<Option<_>>()?;
    let first = *positions.first()?;
    if positions
        .iter()
        .enumerate()
        .any(|(offset, position)| *position != first + offset as i64)
    {
        return None;
    }
    Some(tail.iter().collect())
}

fn complete_price_tail<'a>(
    state: &'a DailyAnchorState,
    code: &str,
    count: usize,
) -> Option<Vec<&'a PriceRow>> {
    let rows = state.price_by_code.get(code)?;
    if rows.len() < count {
        return None;
    }
    let tail = &rows[rows.len() - count..];
    let positions: Vec<_> = tail
        .iter()
        .map(|row| session_index(&state.calendar, row.trade_date))
        .collect::<Option<_>>()?;
    let first = *positions.first()?;
    if positions
        .iter()
        .enumerate()
        .any(|(offset, position)| *position != first + offset as i64)
    {
        return None;
    }
    Some(tail.iter().collect())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| !finite(*value)) {
        f64::NAN
    } else {
        sanitize(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 || values.iter().any(|value| !finite(*value)) {
        return f64::NAN;
    }
    let center = mean(values);
    let variance = values
        .iter()
        .map(|value| (*value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    if variance >= 0.0 && finite(variance) {
        variance.sqrt()
    } else {
        f64::NAN
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() || values.iter().any(|value| !finite(*value)) {
        return f64::NAN;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn z_last_log_positive(values: &[f64]) -> f64 {
    if values.len() < 3 || values.iter().any(|value| !finite(*value) || *value <= EPS) {
        return f64::NAN;
    }
    let logged: Vec<_> = values.iter().map(|value| value.ln()).collect();
    let prior = &logged[..logged.len() - 1];
    let scale = sample_std(prior);
    if !finite(scale) || scale <= EPS {
        f64::NAN
    } else {
        sanitize((logged[logged.len() - 1] - mean(prior)) / scale)
    }
}

fn simple_return(current: f64, previous: f64) -> f64 {
    if finite(current) && finite(previous) && current > EPS && previous > EPS {
        sanitize(current / previous - 1.0)
    } else {
        f64::NAN
    }
}

#[derive(Clone, Debug)]
struct UnderlyingSnapshotRow {
    code: String,
    stock_return1: f64,
    stock_volatility: f64,
    stock_liquidity_z20: f64,
}

fn underlying_snapshot(
    state: &DailyAnchorState,
    ctx: &TypedFactorR88RemainingContext,
) -> Vec<UnderlyingSnapshotRow> {
    let mut out = Vec::new();
    for code in panel_codes(ctx) {
        if price_at_anchor(state, &code).is_none() || base_at_anchor(state, &code).is_none() {
            continue;
        }
        let Some(base_last) = complete_base_tail(state, &code, 1) else {
            continue;
        };
        let base_last = base_last[0];
        let stock_mapping = base_last.stock_code.trim().to_ascii_uppercase();
        if stock_mapping.is_empty() || matches!(stock_mapping.as_str(), "NAN" | "NONE" | "<NA>") {
            continue;
        }
        let Some(close_tail) = complete_base_tail(state, &code, 2) else {
            continue;
        };
        let Some(amount_tail) = complete_base_tail(state, &code, 21) else {
            continue;
        };
        let stock_return1 = simple_return(
            close_tail[1].stock_close_price,
            close_tail[0].stock_close_price,
        );
        let stock_liquidity_z20 = z_last_log_positive(
            &amount_tail
                .iter()
                .map(|row| row.stk_amount)
                .collect::<Vec<_>>(),
        );
        if !(finite(base_last.stock_volatility)
            && base_last.stock_volatility > EPS
            && finite(stock_return1)
            && finite(stock_liquidity_z20)
            && finite(base_last.stk_amount)
            && base_last.stk_amount > EPS)
        {
            continue;
        }
        out.push(UnderlyingSnapshotRow {
            code,
            stock_return1,
            stock_volatility: base_last.stock_volatility,
            stock_liquidity_z20,
        });
    }
    out.sort_by(|left, right| left.code.cmp(&right.code));
    out
}

fn standardized_neighbor_indices(
    codes: &[String],
    coordinates: &[Vec<f64>],
    neighbor_count: usize,
    min_neighbors: usize,
) -> BTreeMap<String, Vec<(usize, f64)>> {
    if codes.is_empty() || coordinates.is_empty() || codes.len() != coordinates.len() {
        return BTreeMap::new();
    }
    let dimension = coordinates[0].len();
    if dimension == 0 || coordinates.iter().any(|row| row.len() != dimension) {
        return BTreeMap::new();
    }
    let mut centers = Vec::with_capacity(dimension);
    let mut scales = Vec::with_capacity(dimension);
    for column in 0..dimension {
        let values: Vec<_> = coordinates
            .iter()
            .map(|row| row[column])
            .filter(|value| finite(*value))
            .collect();
        if values.len() < min_neighbors + 1 {
            return BTreeMap::new();
        }
        let center = median(values.clone());
        let scale = sample_std(&values);
        if !finite(center) || !finite(scale) || scale <= EPS {
            return BTreeMap::new();
        }
        centers.push(center);
        scales.push(scale);
    }
    let standardized: Vec<Vec<f64>> = coordinates
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(column, value)| (*value - centers[column]) / scales[column])
                .collect()
        })
        .collect();
    let available: Vec<_> = standardized
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row.iter().all(|value| finite(*value)).then_some(index))
        .collect();
    if available.len() < min_neighbors + 1 {
        return BTreeMap::new();
    }
    let mut out = BTreeMap::new();
    for &position in &available {
        let mut candidates: Vec<(usize, f64)> = available
            .iter()
            .copied()
            .filter(|candidate| *candidate != position)
            .map(|candidate| {
                let squared = standardized[position]
                    .iter()
                    .zip(standardized[candidate].iter())
                    .map(|(left, right)| (*left - *right).powi(2))
                    .sum::<f64>();
                (candidate, (squared / dimension as f64).sqrt())
            })
            .collect();
        candidates.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| codes[left.0].cmp(&codes[right.0]))
        });
        candidates.truncate(neighbor_count.min(candidates.len()));
        if candidates.len() >= min_neighbors
            && candidates.iter().all(|(_, distance)| finite(*distance))
        {
            out.insert(codes[position].clone(), candidates);
        }
    }
    out
}

/// Immutable score-day cache for the remaining R88 signals whose formula is a
/// full cross-section.  The public per-code functions below remain the direct
/// formula reference.  The typed dispatcher may build this once and then do
/// O(1) lookups for every output key instead of reconstructing the identical
/// cross-section once per bond.
#[derive(Clone, Debug, Default)]
pub struct PreparedR88RemainingValues {
    values: BTreeMap<TypedFactorR88RemainingSignal, BTreeMap<String, f64>>,
}

impl PreparedR88RemainingValues {
    /// Return a cached result for a requested signal.  A code missing from an
    /// already-prepared cross-section is semantically the same as the direct
    /// formula's fail-closed `NaN` result.
    pub fn lookup(
        &self,
        signal: TypedFactorR88RemainingSignal,
        raw_panel_code: &str,
    ) -> Option<f64> {
        self.values.get(&signal).map(|by_code| {
            let code = normalized_panel_code(raw_panel_code);
            by_code.get(&code).copied().unwrap_or(f64::NAN)
        })
    }
}

fn ucd_peer_stock_return_dispersion_values(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<BTreeMap<String, f64>, TypedFactorR88RemainingError> {
    let Some(state) = daily_anchor_state(ctx)? else {
        return Ok(BTreeMap::new());
    };
    let snapshot = underlying_snapshot(&state, ctx);
    let codes: Vec<_> = snapshot.iter().map(|row| row.code.clone()).collect();
    let coordinates: Vec<_> = snapshot
        .iter()
        .map(|row| {
            vec![
                row.stock_return1,
                row.stock_volatility,
                row.stock_liquidity_z20,
            ]
        })
        .collect();
    let neighbors = standardized_neighbor_indices(&codes, &coordinates, PEER_COUNT, PEER_MIN);
    let mut out = BTreeMap::new();
    for (code, picked) in neighbors {
        let values: Vec<_> = picked
            .iter()
            .map(|(index, distance)| (snapshot[*index].stock_return1, 1.0 / (1.0 + *distance)))
            .collect();
        out.insert(code, weighted_std(&values, PEER_MIN));
    }
    Ok(out)
}

fn weighted_std(values: &[(f64, f64)], min_neighbors: usize) -> f64 {
    let selected: Vec<_> = values
        .iter()
        .copied()
        .filter(|(value, weight)| finite(*value) && finite(*weight) && *weight > 0.0)
        .collect();
    if selected.len() < min_neighbors {
        return f64::NAN;
    }
    let total = selected.iter().map(|(_, weight)| *weight).sum::<f64>();
    if total <= EPS {
        return f64::NAN;
    }
    let center = selected
        .iter()
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / total;
    let variance = selected
        .iter()
        .map(|(value, weight)| weight * (value - center).powi(2))
        .sum::<f64>()
        / total;
    if finite(variance) && variance >= 0.0 {
        variance.sqrt()
    } else {
        f64::NAN
    }
}

/// T-1 underlying-stock-state neighborhood weighted dispersion.  The target
/// itself is excluded, then up to seven neighbors are weighted `1/(1+d)` in
/// standardized `(stock return, stock vol, stock liquidity shock)` space.
pub fn ucd_peer_stock_return_dispersion1(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let code = normalized_panel_code(raw_panel_code);
    let Some(state) = daily_anchor_state(ctx)? else {
        return Ok(f64::NAN);
    };
    let snapshot = underlying_snapshot(&state, ctx);
    let codes: Vec<_> = snapshot.iter().map(|row| row.code.clone()).collect();
    let coordinates: Vec<_> = snapshot
        .iter()
        .map(|row| {
            vec![
                row.stock_return1,
                row.stock_volatility,
                row.stock_liquidity_z20,
            ]
        })
        .collect();
    let neighbors = standardized_neighbor_indices(&codes, &coordinates, PEER_COUNT, PEER_MIN);
    let Some(picked) = neighbors.get(&code) else {
        return Ok(f64::NAN);
    };
    let values: Vec<_> = picked
        .iter()
        .map(|(index, distance)| (snapshot[*index].stock_return1, 1.0 / (1.0 + *distance)))
        .collect();
    Ok(weighted_std(&values, PEER_MIN))
}

#[derive(Clone, Debug)]
struct StructuralSnapshotRow {
    code: String,
    year_to_mat: f64,
    duration: f64,
    bond_prem_ratio: f64,
    ytm: f64,
    log_remain_size: f64,
    bond_return1: f64,
}

fn structural_snapshot(
    state: &DailyAnchorState,
    ctx: &TypedFactorR88RemainingContext,
) -> Vec<StructuralSnapshotRow> {
    let mut out = Vec::new();
    for code in panel_codes(ctx) {
        let (Some(_price), Some(_base)) =
            (price_at_anchor(state, &code), base_at_anchor(state, &code))
        else {
            continue;
        };
        let (Some(price_tail), Some(base_tail)) = (
            complete_price_tail(state, &code, 1),
            complete_base_tail(state, &code, 1),
        ) else {
            continue;
        };
        let price = price_tail[0];
        let base = base_tail[0];
        // `price`/`base` local bindings certify the exact anchor join; retain
        // them so a future refactor cannot weaken this boundary.
        if price.trade_date != state.anchor || base.trade_date != state.anchor {
            continue;
        }
        let log_remain_size = if finite(base.remain_size) && base.remain_size > EPS {
            base.remain_size.ln()
        } else {
            f64::NAN
        };
        if !(finite(base.year_to_mat)
            && base.year_to_mat >= 0.0
            && finite(base.duration)
            && base.duration >= 0.0
            && finite(base.bond_prem_ratio)
            && finite(base.ytm)
            && finite(log_remain_size))
        {
            continue;
        }
        out.push(StructuralSnapshotRow {
            code,
            year_to_mat: base.year_to_mat,
            duration: base.duration,
            bond_prem_ratio: base.bond_prem_ratio,
            ytm: base.ytm,
            log_remain_size,
            bond_return1: simple_return(price.close_price, price.prev_close_price),
        });
    }
    out.sort_by(|left, right| left.code.cmp(&right.code));
    out
}

/// T-1 structural-neighborhood weighted peer bond-return dispersion.  It is
/// deliberately a dispersion, never a return mean or a fitted residual.
pub fn sng_peer_return_dispersion1(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let code = normalized_panel_code(raw_panel_code);
    let Some(state) = daily_anchor_state(ctx)? else {
        return Ok(f64::NAN);
    };
    let snapshot = structural_snapshot(&state, ctx);
    let codes: Vec<_> = snapshot.iter().map(|row| row.code.clone()).collect();
    let coordinates: Vec<_> = snapshot
        .iter()
        .map(|row| {
            vec![
                row.year_to_mat,
                row.duration,
                row.bond_prem_ratio,
                row.ytm,
                row.log_remain_size,
            ]
        })
        .collect();
    let neighbors = standardized_neighbor_indices(&codes, &coordinates, PEER_COUNT, PEER_MIN);
    let Some(picked) = neighbors.get(&code) else {
        return Ok(f64::NAN);
    };
    let values: Vec<_> = picked
        .iter()
        .map(|(index, distance)| (snapshot[*index].bond_return1, 1.0 / (1.0 + *distance)))
        .collect();
    Ok(weighted_std(&values, PEER_MIN))
}

fn sng_peer_return_dispersion_values(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<BTreeMap<String, f64>, TypedFactorR88RemainingError> {
    let Some(state) = daily_anchor_state(ctx)? else {
        return Ok(BTreeMap::new());
    };
    let snapshot = structural_snapshot(&state, ctx);
    let codes: Vec<_> = snapshot.iter().map(|row| row.code.clone()).collect();
    let coordinates: Vec<_> = snapshot
        .iter()
        .map(|row| {
            vec![
                row.year_to_mat,
                row.duration,
                row.bond_prem_ratio,
                row.ytm,
                row.log_remain_size,
            ]
        })
        .collect();
    let neighbors = standardized_neighbor_indices(&codes, &coordinates, PEER_COUNT, PEER_MIN);
    let mut out = BTreeMap::new();
    for (code, picked) in neighbors {
        let values: Vec<_> = picked
            .iter()
            .map(|(index, distance)| (snapshot[*index].bond_return1, 1.0 / (1.0 + *distance)))
            .collect();
        out.insert(code, weighted_std(&values, PEER_MIN));
    }
    Ok(out)
}

fn population_std(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| !finite(*value)) {
        return f64::NAN;
    }
    let center = mean(values);
    let variance = values
        .iter()
        .map(|value| (*value - center).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    if finite(variance) && variance >= 0.0 {
        variance.sqrt()
    } else {
        f64::NAN
    }
}

/// Pandas `rank(method="average")`, returned on the [0, 1] interval used by
/// the state-gated source (`(rank - 1) / (n - 1)`).
fn rank01_average(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    let mut order: Vec<_> = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| finite(value).then_some((index, value)))
        .collect();
    if order.len() < ISGM_MIN_CROSS_SECTION
        || population_std(&order.iter().map(|(_, value)| *value).collect::<Vec<_>>()) <= EPS
    {
        return out;
    }
    order.sort_by(|left, right| left.1.total_cmp(&right.1));
    let denominator = (order.len() - 1) as f64;
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && order[end].1.total_cmp(&order[start].1) == Ordering::Equal {
            end += 1;
        }
        // Pandas ranks from one.  Equal values receive the average rank.
        let average_rank = ((start + 1 + end) as f64) / 2.0;
        for (index, _) in &order[start..end] {
            out[*index] = (average_rank - 1.0) / denominator;
        }
        start = end;
    }
    out
}

fn smoothstep(value: f64) -> f64 {
    if finite(value) {
        let clipped = value.clamp(0.0, 1.0);
        3.0 * clipped.powi(2) - 2.0 * clipped.powi(3)
    } else {
        f64::NAN
    }
}

fn isgm_stockvol_gates(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<BTreeMap<String, f64>, TypedFactorR88RemainingError> {
    let price = strict_price_rows(ctx)?;
    let base = strict_base_rows(ctx)?;
    let price: Vec<_> = price
        .into_iter()
        .filter(|row| finite(row.close_price) && row.close_price > EPS)
        .collect();
    let Some(anchor) = price.iter().map(|row| row.trade_date).max() else {
        return Ok(BTreeMap::new());
    };
    let price_codes: BTreeSet<_> = price
        .iter()
        .filter(|row| row.trade_date == anchor)
        .map(|row| row.code.clone())
        .collect();
    let mut records: Vec<(String, f64)> = base
        .iter()
        .filter(|row| row.trade_date == anchor && price_codes.contains(&row.code))
        .map(|row| (row.code.clone(), row.stock_volatility))
        .collect();
    records.sort_by(|left, right| left.0.cmp(&right.0));
    let values: Vec<_> = records.iter().map(|(_, value)| *value).collect();
    let ranks = rank01_average(&values);
    let mut out = BTreeMap::new();
    for ((code, _), rank) in records.into_iter().zip(ranks) {
        if finite(rank) {
            out.insert(code, smoothstep(rank));
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct EndpointRow {
    bin: i64,
    row: R88RemainingIntradayRow,
}

fn endpoint_rows(rows: &[R88RemainingIntradayRow]) -> Vec<EndpointRow> {
    let mut endpoints: BTreeMap<i64, R88RemainingIntradayRow> = BTreeMap::new();
    for row in rows {
        let bin = row.time_ns.div_euclid(ITR_EVENT_BIN) * ITR_EVENT_BIN;
        let replace = endpoints
            .get(&bin)
            .map(|existing| (row.time_ns, row.seq) >= (existing.time_ns, existing.seq))
            .unwrap_or(true);
        if replace {
            endpoints.insert(bin, row.clone());
        }
    }
    endpoints
        .into_iter()
        .map(|(bin, row)| EndpointRow { bin, row })
        .collect()
}

fn clock_position(bin: i64) -> f64 {
    let clock = bin.div_euclid(NS_PER_MICROSECOND) * NS_PER_MICROSECOND;
    let seconds = clock as f64 / NS_PER_SECOND as f64;
    if (MORNING_START..=MORNING_END).contains(&clock) {
        (seconds - (MORNING_START as f64 / NS_PER_SECOND as f64)) / CONTINUOUS_SECONDS
    } else if (AFTERNOON_START..=STRICT_CUTOFF).contains(&clock) {
        (120.0 * 60.0 + seconds - (AFTERNOON_START as f64 / NS_PER_SECOND as f64))
            / CONTINUOUS_SECONDS
    } else {
        f64::NAN
    }
}

fn isgm_trade_quote_center_gap(rows: &[R88RemainingIntradayRow]) -> f64 {
    let endpoints = endpoint_rows(rows);
    let mut quote: Vec<(i64, f64, f64, f64)> = endpoints
        .into_iter()
        .filter_map(|endpoint| {
            let ask = endpoint.row.ask_price[0];
            let bid = endpoint.row.bid_price[0];
            (finite(ask) && finite(bid) && ask > bid && bid > EPS).then_some((
                endpoint.bin,
                ask,
                bid,
                endpoint.row.num_trades,
            ))
        })
        .collect();
    if quote.len() < ISGM_MIN_ENDPOINTS {
        return f64::NAN;
    }
    quote.sort_by_key(|(bin, _, _, _)| *bin);
    let mut revisions = vec![false; quote.len()];
    for index in 1..quote.len() {
        let adjacent = quote[index].0 - quote[index - 1].0 == ITR_EVENT_BIN;
        let bid_changed =
            (quote[index].2.ln() - quote[index - 1].2.ln()).abs() > ISGM_QUOTE_REL_TOL;
        let ask_changed =
            (quote[index].1.ln() - quote[index - 1].1.ln()).abs() > ISGM_QUOTE_REL_TOL;
        revisions[index] = adjacent && (bid_changed || ask_changed);
    }
    let mut counts = Vec::new();
    let mut retained_revisions = Vec::new();
    let mut clocks = Vec::new();
    let mut bins = Vec::new();
    for (index, (bin, _, _, count)) in quote.into_iter().enumerate() {
        if finite(count) {
            bins.push(bin);
            counts.push(count);
            retained_revisions.push(revisions[index]);
            clocks.push(clock_position(bin));
        }
    }
    if counts.len() < ISGM_MIN_ENDPOINTS || clocks.iter().any(|value| !finite(*value)) {
        return f64::NAN;
    }
    let mut increments = vec![0.0; counts.len()];
    let mut usable = vec![false; counts.len()];
    for index in 1..counts.len() {
        let increment = counts[index] - counts[index - 1];
        usable[index] = bins[index] - bins[index - 1] == ITR_EVENT_BIN && finite(increment);
        if usable[index] {
            increments[index] = increment;
        }
    }
    if usable.iter().filter(|value| **value).count() < ISGM_MIN_TRADE_EVENTS
        || increments
            .iter()
            .zip(usable.iter())
            .any(|(increment, usable)| *usable && *increment < -EPS)
    {
        return f64::NAN;
    }
    for (increment, usable) in increments.iter_mut().zip(usable.iter()) {
        if !*usable {
            *increment = 0.0;
        } else {
            *increment = increment.max(0.0);
        }
    }
    if increments.iter().filter(|value| **value > EPS).count() < ISGM_MIN_TRADE_EVENTS
        || retained_revisions.iter().filter(|value| **value).count() < ISGM_MIN_QUOTE_EVENTS
    {
        return f64::NAN;
    }
    let total = increments.iter().sum::<f64>();
    if total <= EPS {
        return f64::NAN;
    }
    let trade_center = increments
        .iter()
        .zip(clocks.iter())
        .map(|(weight, clock)| weight * clock)
        .sum::<f64>()
        / total;
    let quote_values: Vec<_> = clocks
        .iter()
        .zip(retained_revisions.iter())
        .filter_map(|(clock, revision)| (*revision).then_some(*clock))
        .collect();
    sanitize(trade_center - mean(&quote_values))
}

/// Strict-T-1 stock-volatility smooth gate times T-day trade/quote clock
/// center separation.  The gate is undefined unless the anchored state cross
/// section has at least thirty valid, nonconstant stock-volatility values.
pub fn isgm_stockvol_trade_quote_clock_center_gap(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let code = normalized_panel_code(raw_panel_code);
    let gates = isgm_stockvol_gates(ctx)?;
    let Some(gate) = gates.get(&code).copied() else {
        return Ok(f64::NAN);
    };
    Ok(sanitize(
        gate * isgm_trade_quote_center_gap(&physical_rows_for_code(ctx, raw_panel_code)),
    ))
}

fn isgm_stockvol_trade_quote_clock_center_gap_values(
    ctx: &TypedFactorR88RemainingContext,
    groups: &BTreeMap<String, Vec<R88RemainingIntradayRow>>,
) -> Result<BTreeMap<String, f64>, TypedFactorR88RemainingError> {
    let gates = isgm_stockvol_gates(ctx)?;
    let mut out = BTreeMap::new();
    for code in panel_codes(ctx) {
        let Some(gate) = gates.get(&code).copied() else {
            continue;
        };
        let rows = groups.get(&code).map(Vec::as_slice).unwrap_or(&[]);
        out.insert(code, sanitize(gate * isgm_trade_quote_center_gap(rows)));
    }
    Ok(out)
}

fn valid_qgeo_path(rows: &[R88RemainingIntradayRow]) -> bool {
    if rows.len() < QGEO_MIN_ROWS {
        return false;
    }
    if rows
        .windows(2)
        .any(|pair| pair[0].time_ns >= pair[1].time_ns)
    {
        return false;
    }
    let mut sequences = BTreeSet::new();
    for row in rows {
        if !sequences.insert(row.seq) || !finite(row.last) || row.last <= EPS {
            return false;
        }
        if row
            .ask_price
            .iter()
            .chain(row.bid_price.iter())
            .any(|value| !finite(*value) || *value <= EPS)
            || row
                .ask_volume
                .iter()
                .chain(row.bid_volume.iter())
                .any(|value| !finite(*value) || *value < 0.0)
            || row.ask_price[0] < row.bid_price[0]
            || row.ask_price.windows(2).any(|pair| pair[1] < pair[0])
            || row.bid_price.windows(2).any(|pair| pair[1] > pair[0])
        {
            return false;
        }
        let touch_spread = row.ask_price[0] - row.bid_price[0];
        let touch_depth = row.ask_volume[0] + row.bid_volume[0];
        let ask_total = row.ask_volume.iter().sum::<f64>();
        let bid_total = row.bid_volume.iter().sum::<f64>();
        if touch_spread <= EPS || touch_depth <= EPS || ask_total <= EPS || bid_total <= EPS {
            return false;
        }
        // `_book_geometry` computes the depth entropy and L1--L5 ladder
        // convexity before it exposes the microprice family.  Preserve those
        // shared fail-closed prerequisites even though this signal uses only
        // the L1 dislocation afterwards.
        for prices in [&row.ask_price, &row.bid_price] {
            let gaps: Vec<_> = if std::ptr::eq(prices, &row.ask_price) {
                prices
                    .windows(2)
                    .map(|pair| (pair[1] / pair[0]).ln())
                    .collect()
            } else {
                prices
                    .windows(2)
                    .map(|pair| (pair[0] / pair[1]).ln())
                    .collect()
            };
            if gaps.iter().any(|gap| !finite(*gap) || *gap < 0.0) {
                return false;
            }
            let inner = (gaps[0] + gaps[1]) / 2.0;
            let outer = (gaps[2] + gaps[3]) / 2.0;
            if !finite(inner + outer) || inner + outer <= EPS {
                return false;
            }
        }
    }
    true
}

/// L1 microprice dislocation's sign against the following same-session log
/// return.  It deliberately ignores the noon bridge, exactly as the Python
/// quote-geometry source does.
fn qgeo_micro_last_next_return_sign_alignment_rows(rows: &[R88RemainingIntradayRow]) -> f64 {
    if !valid_qgeo_path(rows) {
        return f64::NAN;
    }
    let displacement: Vec<_> = rows
        .iter()
        .map(|row| {
            let touch_depth = row.ask_volume[0] + row.bid_volume[0];
            let microprice = (row.ask_price[0] * row.bid_volume[0]
                + row.bid_price[0] * row.ask_volume[0])
                / touch_depth;
            (microprice - row.last) / (row.ask_price[0] - row.bid_price[0])
        })
        .collect();
    let mut agreement = Vec::new();
    for index in 0..rows.len() - 1 {
        if continuous_session(rows[index].time_ns) != continuous_session(rows[index + 1].time_ns) {
            continue;
        }
        let next_return = safe_log_return(rows[index + 1].last, rows[index].last);
        if finite(displacement[index])
            && finite(next_return)
            && displacement[index].abs() > EPS
            && next_return.abs() > EPS
        {
            agreement.push(displacement[index].signum() * next_return.signum());
        }
    }
    if agreement.len() >= QGEO_MIN_PAIRS {
        mean(&agreement)
    } else {
        f64::NAN
    }
}

pub fn qgeo_micro_last_next_return_sign_alignment(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> f64 {
    qgeo_micro_last_next_return_sign_alignment_rows(&physical_rows_for_code(ctx, raw_panel_code))
}

#[derive(Clone, Copy, Debug)]
struct CsnPassiveSummary {
    churn_per_trade: f64,
    trade_event_rate: f64,
    top_depth_update_intensity: f64,
}

impl CsnPassiveSummary {
    fn nan() -> Self {
        Self {
            churn_per_trade: f64::NAN,
            trade_event_rate: f64::NAN,
            top_depth_update_intensity: f64::NAN,
        }
    }
}

fn valid_csn_path(rows: &[R88RemainingIntradayRow]) -> bool {
    if rows.len() < CSN_MIN_PATH_ROWS
        || rows
            .windows(2)
            .any(|pair| pair[0].time_ns >= pair[1].time_ns)
    {
        return false;
    }
    let mut sequences = BTreeSet::new();
    rows.iter().all(|row| sequences.insert(row.seq))
}

fn csn_top_book_valid(rows: &[R88RemainingIntradayRow]) -> bool {
    rows.iter().all(|row| {
        finite(row.ask_price[0])
            && finite(row.bid_price[0])
            && row.ask_price[0] > 0.0
            && row.bid_price[0] > 0.0
            && row.ask_price[0] >= row.bid_price[0]
            && finite(row.ask_volume[0])
            && finite(row.bid_volume[0])
            && row.ask_volume[0] >= 0.0
            && row.bid_volume[0] >= 0.0
    })
}

fn csn_top_depth_update_intensity(ask: &[f64], bid: &[f64]) -> f64 {
    let mut values = Vec::new();
    for index in 1..ask.len() {
        let ask_sum = ask[index] + ask[index - 1];
        let bid_sum = bid[index] + bid[index - 1];
        if ask_sum > EPS && bid_sum > EPS {
            values.push((2.0 * (ask[index] - ask[index - 1]) / ask_sum).abs());
            values.push((2.0 * (bid[index] - bid[index - 1]) / bid_sum).abs());
        }
    }
    // Python's `valid.sum()` counts interval rows, not both side values.
    let valid_intervals = values.len() / 2;
    if valid_intervals < 3 || values.iter().any(|value| !finite(*value)) {
        f64::NAN
    } else {
        mean(&values)
    }
}

fn csn_passive_summary(rows: &[R88RemainingIntradayRow]) -> CsnPassiveSummary {
    if !valid_csn_path(rows) || !csn_top_book_valid(rows) {
        return CsnPassiveSummary::nan();
    }
    let trades: Vec<_> = rows.iter().map(|row| row.num_trades).collect();
    if trades.iter().any(|value| !finite(*value) || *value < 0.0) {
        return CsnPassiveSummary::nan();
    }
    let increments: Vec<_> = trades.windows(2).map(|pair| pair[1] - pair[0]).collect();
    if increments
        .iter()
        .any(|value| !finite(*value) || *value < 0.0)
    {
        return CsnPassiveSummary::nan();
    }
    let ask: Vec<_> = rows.iter().map(|row| row.ask_volume[0]).collect();
    let bid: Vec<_> = rows.iter().map(|row| row.bid_volume[0]).collect();
    let mut churn = Vec::new();
    let mut event_trades = Vec::new();
    for index in 1..rows.len() {
        let prior_depth = ask[index - 1] + bid[index - 1];
        let current_depth = ask[index] + bid[index];
        let event = increments[index - 1] > 0.0;
        if event && prior_depth > EPS && finite(current_depth) {
            churn.push((bid[index] - bid[index - 1]).abs() + (ask[index] - ask[index - 1]).abs());
            event_trades.push(increments[index - 1]);
        }
    }
    if churn.len() < 3 || churn.iter().any(|value| !finite(*value)) {
        return CsnPassiveSummary::nan();
    }
    let churn_per_trade = safe_div(churn.iter().sum(), event_trades.iter().sum());
    CsnPassiveSummary {
        churn_per_trade,
        trade_event_rate: increments.iter().filter(|value| **value > 0.0).count() as f64
            / increments.len() as f64,
        top_depth_update_intensity: csn_top_depth_update_intensity(&ask, &bid),
    }
}

/// Pands `rank(method="average", pct=True)` exactly: rank starts at one and
/// is divided by the number of valid cross-sectional observations.
fn percentile_ranks(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    let mut order: Vec<_> = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| finite(value).then_some((index, value)))
        .collect();
    if order.is_empty() {
        return out;
    }
    order.sort_by(|left, right| left.1.total_cmp(&right.1));
    let denominator = order.len() as f64;
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && order[end].1.total_cmp(&order[start].1) == Ordering::Equal {
            end += 1;
        }
        let average_rank = ((start + 1 + end) as f64) / 2.0;
        for (index, _) in &order[start..end] {
            out[*index] = average_rank / denominator;
        }
        start = end;
    }
    out
}

/// Same-day local cross-sectional churn rank minus the median churn rank of
/// its twelve nearest peers in `(trade_event_rate, top_depth_update_intensity)`
/// percentile-rank space.
pub fn csn_pql_churn_neighbor_gap(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
) -> f64 {
    let target = normalized_panel_code(raw_panel_code);
    let groups = physical_rows_by_code(ctx);
    let mut records: Vec<(String, CsnPassiveSummary)> = panel_codes(ctx)
        .into_iter()
        .filter_map(|code| {
            groups
                .get(&code)
                .map(|rows| (code, csn_passive_summary(rows)))
        })
        .filter(|(_, summary)| {
            finite(summary.churn_per_trade)
                && finite(summary.trade_event_rate)
                && finite(summary.top_depth_update_intensity)
        })
        .collect();
    records.sort_by(|left, right| left.0.cmp(&right.0));
    if records.len() < CSN_MIN_CROSS_SECTION
        || records.iter().any(|(_, summary)| {
            !finite(summary.churn_per_trade)
                || !finite(summary.trade_event_rate)
                || !finite(summary.top_depth_update_intensity)
        })
    {
        return f64::NAN;
    }
    let churn: Vec<_> = records
        .iter()
        .map(|(_, value)| value.churn_per_trade)
        .collect();
    let trade_rate: Vec<_> = records
        .iter()
        .map(|(_, value)| value.trade_event_rate)
        .collect();
    let intensity: Vec<_> = records
        .iter()
        .map(|(_, value)| value.top_depth_update_intensity)
        .collect();
    // Python rejects any requested series with fewer than two distinct values.
    let distinct = |values: &[f64]| {
        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.total_cmp(right));
        sorted.dedup_by(|left, right| left.total_cmp(right) == Ordering::Equal);
        sorted.len()
    };
    if distinct(&churn) < 2 || distinct(&trade_rate) < 2 || distinct(&intensity) < 2 {
        return f64::NAN;
    }
    let churn_rank = percentile_ranks(&churn);
    let trade_rank = percentile_ranks(&trade_rate);
    let intensity_rank = percentile_ranks(&intensity);
    let Some(position) = records.iter().position(|(code, _)| *code == target) else {
        return f64::NAN;
    };
    let mut candidates: Vec<(usize, f64)> = (0..records.len())
        .filter(|index| *index != position)
        .map(|index| {
            let distance = ((trade_rank[position] - trade_rank[index]).powi(2)
                + (intensity_rank[position] - intensity_rank[index]).powi(2))
            .sqrt();
            (index, distance)
        })
        .collect();
    candidates.sort_by(|left, right| left.1.total_cmp(&right.1));
    candidates.truncate(CSN_NEIGHBOR_COUNT);
    if candidates.len() < CSN_NEIGHBOR_COUNT {
        return f64::NAN;
    }
    let peer_ranks: Vec<_> = candidates
        .iter()
        .map(|(index, _)| churn_rank[*index])
        .collect();
    sanitize(churn_rank[position] - median(peer_ranks))
}

fn csn_pql_churn_neighbor_gap_values(
    ctx: &TypedFactorR88RemainingContext,
    groups: &BTreeMap<String, Vec<R88RemainingIntradayRow>>,
) -> BTreeMap<String, f64> {
    let mut records: Vec<(String, CsnPassiveSummary)> = panel_codes(ctx)
        .into_iter()
        .filter_map(|code| {
            groups
                .get(&code)
                .map(|rows| (code, csn_passive_summary(rows)))
        })
        .filter(|(_, summary)| {
            finite(summary.churn_per_trade)
                && finite(summary.trade_event_rate)
                && finite(summary.top_depth_update_intensity)
        })
        .collect();
    records.sort_by(|left, right| left.0.cmp(&right.0));
    if records.len() < CSN_MIN_CROSS_SECTION
        || records.iter().any(|(_, summary)| {
            !finite(summary.churn_per_trade)
                || !finite(summary.trade_event_rate)
                || !finite(summary.top_depth_update_intensity)
        })
    {
        return BTreeMap::new();
    }
    let churn: Vec<_> = records
        .iter()
        .map(|(_, value)| value.churn_per_trade)
        .collect();
    let trade_rate: Vec<_> = records
        .iter()
        .map(|(_, value)| value.trade_event_rate)
        .collect();
    let intensity: Vec<_> = records
        .iter()
        .map(|(_, value)| value.top_depth_update_intensity)
        .collect();
    let distinct = |values: &[f64]| {
        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.total_cmp(right));
        sorted.dedup_by(|left, right| left.total_cmp(right) == Ordering::Equal);
        sorted.len()
    };
    if distinct(&churn) < 2 || distinct(&trade_rate) < 2 || distinct(&intensity) < 2 {
        return BTreeMap::new();
    }
    let churn_rank = percentile_ranks(&churn);
    let trade_rank = percentile_ranks(&trade_rate);
    let intensity_rank = percentile_ranks(&intensity);
    let mut out = BTreeMap::new();
    for position in 0..records.len() {
        let mut candidates: Vec<(usize, f64)> = (0..records.len())
            .filter(|index| *index != position)
            .map(|index| {
                let distance = ((trade_rank[position] - trade_rank[index]).powi(2)
                    + (intensity_rank[position] - intensity_rank[index]).powi(2))
                .sqrt();
                (index, distance)
            })
            .collect();
        candidates.sort_by(|left, right| left.1.total_cmp(&right.1));
        candidates.truncate(CSN_NEIGHBOR_COUNT);
        if candidates.len() < CSN_NEIGHBOR_COUNT {
            continue;
        }
        let peer_ranks: Vec<_> = candidates
            .iter()
            .map(|(index, _)| churn_rank[*index])
            .collect();
        out.insert(
            records[position].0.clone(),
            sanitize(churn_rank[position] - median(peer_ranks)),
        );
    }
    out
}

fn hybrid_value_maps(
    ctx: &TypedFactorR88RemainingContext,
) -> Result<(BTreeMap<String, f64>, BTreeMap<String, f64>), TypedFactorR88RemainingError> {
    let visible = strict_visible_rows_by_code(ctx);
    let twap = strict_twap_rows(ctx)?;
    let base = strict_base_rows(ctx)?;
    let mut twap_by_code: BTreeMap<String, Vec<&TwapRow>> = BTreeMap::new();
    for row in &twap {
        twap_by_code.entry(row.code.clone()).or_default().push(row);
    }
    let mut base_by_code: BTreeMap<String, Vec<&BaseRow>> = BTreeMap::new();
    for row in &base {
        base_by_code.entry(row.code.clone()).or_default().push(row);
    }
    let mut curve = BTreeMap::new();
    let mut flow = BTreeMap::new();
    for code in panel_codes(ctx) {
        let rows = visible.get(&code).map(Vec::as_slice).unwrap_or(&[]);
        let twap_rows = twap_by_code.get(&code).map(Vec::as_slice).unwrap_or(&[]);
        curve.insert(
            code.clone(),
            sanitize(hybrid_current_range(rows) * hist_twap_curve_mean20(twap_rows)),
        );
        let base_rows = base_by_code.get(&code).map(Vec::as_slice).unwrap_or(&[]);
        let previous_cb_amount = last_finite(base_rows.iter().map(|row| row.cb_amount));
        flow.insert(
            code,
            sanitize(
                safe_div(hybrid_current_total_amount(rows), previous_cb_amount)
                    * hist_overnight_mean20(twap_rows),
            ),
        );
    }
    Ok((curve, flow))
}

fn itr_stock_shock_same_bin_directional_agreement_values(
    ctx: &TypedFactorR88RemainingContext,
    groups: &BTreeMap<String, Vec<R88RemainingIntradayRow>>,
) -> Result<BTreeMap<String, f64>, TypedFactorR88RemainingError> {
    let mapping = point_in_time_map(ctx)?;
    let mut out = BTreeMap::new();
    for code in panel_codes(ctx) {
        out.insert(
            code.clone(),
            itr_stock_shock_same_bin_directional_agreement_from_groups(&mapping, &groups, &code),
        );
    }
    Ok(out)
}

fn qgeo_micro_last_next_return_sign_alignment_values(
    ctx: &TypedFactorR88RemainingContext,
    groups: &BTreeMap<String, Vec<R88RemainingIntradayRow>>,
) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for code in panel_codes(ctx) {
        let rows = groups.get(&code).map(Vec::as_slice).unwrap_or(&[]);
        out.insert(code, qgeo_micro_last_next_return_sign_alignment_rows(rows));
    }
    out
}

/// Construct only the heavy, cross-sectional signal maps requested by a
/// score-day dispatch.  The direct per-code formula functions intentionally
/// remain below as the parity reference; this cache changes only reuse, never
/// their inputs, ordering, or formula arithmetic.
pub fn prepare_r88_remaining_values<I>(
    ctx: &TypedFactorR88RemainingContext,
    requested_signals: I,
) -> Result<PreparedR88RemainingValues, TypedFactorR88RemainingError>
where
    I: IntoIterator<Item = TypedFactorR88RemainingSignal>,
{
    let requested: BTreeSet<_> = requested_signals.into_iter().collect();
    let mut values = BTreeMap::new();
    let needs_physical_groups = requested.iter().any(|signal| {
        matches!(
            signal,
            TypedFactorR88RemainingSignal::JointTailRangeCoexpansion
                | TypedFactorR88RemainingSignal::JointTailSignedCojump
                | TypedFactorR88RemainingSignal::JointTailTerminalLocationCoshock
                | TypedFactorR88RemainingSignal::ItrStockShockSameBinDirectionalAgreement
                | TypedFactorR88RemainingSignal::IsgmStockvolTradeQuoteClockCenterGap
                | TypedFactorR88RemainingSignal::QgeoMicroLastNextReturnSignAlignment
                | TypedFactorR88RemainingSignal::CsnPqlChurnNeighborGap
        )
    });
    let physical_groups = needs_physical_groups.then(|| physical_rows_by_code(ctx));
    if requested.iter().any(|signal| {
        matches!(
            signal,
            TypedFactorR88RemainingSignal::HybridCurrentRangeVsHistTwapCurve
                | TypedFactorR88RemainingSignal::HybridCurrentFlowVsHistOvernightResponse
        )
    }) {
        let (curve, flow) = hybrid_value_maps(ctx)?;
        if requested.contains(&TypedFactorR88RemainingSignal::HybridCurrentRangeVsHistTwapCurve) {
            values.insert(
                TypedFactorR88RemainingSignal::HybridCurrentRangeVsHistTwapCurve,
                curve,
            );
        }
        if requested
            .contains(&TypedFactorR88RemainingSignal::HybridCurrentFlowVsHistOvernightResponse)
        {
            values.insert(
                TypedFactorR88RemainingSignal::HybridCurrentFlowVsHistOvernightResponse,
                flow,
            );
        }
    }
    if requested.iter().any(|signal| {
        matches!(
            signal,
            TypedFactorR88RemainingSignal::JointTailRangeCoexpansion
                | TypedFactorR88RemainingSignal::JointTailSignedCojump
                | TypedFactorR88RemainingSignal::JointTailTerminalLocationCoshock
        )
    }) {
        let (signed, range, location) = joint_tail_value_maps(
            ctx,
            physical_groups
                .as_ref()
                .expect("joint cache requires shared physical groups"),
        )?;
        if requested.contains(&TypedFactorR88RemainingSignal::JointTailSignedCojump) {
            values.insert(TypedFactorR88RemainingSignal::JointTailSignedCojump, signed);
        }
        if requested.contains(&TypedFactorR88RemainingSignal::JointTailRangeCoexpansion) {
            values.insert(
                TypedFactorR88RemainingSignal::JointTailRangeCoexpansion,
                range,
            );
        }
        if requested.contains(&TypedFactorR88RemainingSignal::JointTailTerminalLocationCoshock) {
            values.insert(
                TypedFactorR88RemainingSignal::JointTailTerminalLocationCoshock,
                location,
            );
        }
    }
    for signal in requested {
        let by_code = match signal {
            TypedFactorR88RemainingSignal::UcdPeerStockReturnDispersion1 => {
                ucd_peer_stock_return_dispersion_values(ctx)?
            }
            TypedFactorR88RemainingSignal::SngPeerReturnDispersion1 => {
                sng_peer_return_dispersion_values(ctx)?
            }
            TypedFactorR88RemainingSignal::CsnPqlChurnNeighborGap => {
                csn_pql_churn_neighbor_gap_values(
                    ctx,
                    physical_groups
                        .as_ref()
                        .expect("CSN cache requires shared physical groups"),
                )
            }
            TypedFactorR88RemainingSignal::IsgmStockvolTradeQuoteClockCenterGap => {
                isgm_stockvol_trade_quote_clock_center_gap_values(
                    ctx,
                    physical_groups
                        .as_ref()
                        .expect("ISGM cache requires shared physical groups"),
                )?
            }
            TypedFactorR88RemainingSignal::ItrStockShockSameBinDirectionalAgreement => {
                itr_stock_shock_same_bin_directional_agreement_values(
                    ctx,
                    physical_groups
                        .as_ref()
                        .expect("ITR cache requires shared physical groups"),
                )?
            }
            TypedFactorR88RemainingSignal::QgeoMicroLastNextReturnSignAlignment => {
                qgeo_micro_last_next_return_sign_alignment_values(
                    ctx,
                    physical_groups
                        .as_ref()
                        .expect("QGEO cache requires shared physical groups"),
                )
            }
            _ => continue,
        };
        values.insert(signal, by_code);
    }
    Ok(PreparedR88RemainingValues { values })
}

/// The pure research hook a future dispatcher would call.  It intentionally
/// has no PyO3 / registry integration; the `Result` boundary preserves source
/// duplicate and mapping failures for that adapter to surface.
pub fn compute_r88_remaining_signal(
    ctx: &TypedFactorR88RemainingContext,
    raw_panel_code: &str,
    signal: &str,
) -> Result<f64, TypedFactorR88RemainingError> {
    let Some(parsed) = TypedFactorR88RemainingSignal::parse(signal) else {
        return Err(TypedFactorR88RemainingError::UnknownSignal(
            signal.to_string(),
        ));
    };
    match parsed {
        TypedFactorR88RemainingSignal::HybridCurrentRangeVsHistTwapCurve => {
            hybrid_current_range_vs_hist_twap_curve(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::HybridCurrentFlowVsHistOvernightResponse => {
            hybrid_current_flow_vs_hist_overnight_response(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::JointTailRangeCoexpansion => {
            joint_tail_range_coexpansion(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::JointTailSignedCojump => {
            joint_tail_signed_cojump(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::JointTailTerminalLocationCoshock => {
            joint_tail_terminal_location_coshock(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::ItrStockShockSameBinDirectionalAgreement => {
            itr_stock_shock_same_bin_directional_agreement(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::UcdPeerStockReturnDispersion1 => {
            ucd_peer_stock_return_dispersion1(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::IsgmStockvolTradeQuoteClockCenterGap => {
            isgm_stockvol_trade_quote_clock_center_gap(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::QgeoMicroLastNextReturnSignAlignment => Ok(
            qgeo_micro_last_next_return_sign_alignment(ctx, raw_panel_code),
        ),
        TypedFactorR88RemainingSignal::SngPeerReturnDispersion1 => {
            sng_peer_return_dispersion1(ctx, raw_panel_code)
        }
        TypedFactorR88RemainingSignal::CsnPqlChurnNeighborGap => {
            Ok(csn_pql_churn_neighbor_gap(ctx, raw_panel_code))
        }
    }
}
