//! Strict-prior daily kernels for the seven R88 research factors.
//!
//! The research-only typed-factor dispatcher constructs
//! [`TypedFactorR88DailyContext`] from its daily sources and preserves each
//! concrete `signal` / `family` pair from the R88 research profile.  This
//! module owns only the strict-prior formula contract: it has no live-profile,
//! scheduler, database, or output-path dependency.  The dispatcher requests
//! the source columns documented beside each signal below.
//!
//! Every helper filters `trade_date < score_date` before it derives a source
//! anchor, joins data, or computes a rank.  Thus a score-day daily row cannot
//! leak into a T1430 factor value even if it is supplied by an upstream caller.

use chrono::NaiveDate;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const EPS: f64 = 1e-12;
const WINDOW: usize = 60;
const MIN_JOINT_OBSERVATIONS: usize = 45;
const MIN_CONDITIONAL_OBSERVATIONS: usize = 10;
const MIN_TAIL_OBSERVATIONS: usize = 8;
const LOWER_COPULA_TAIL: f64 = 0.25;
const UPPER_COPULA_TAIL: f64 = 0.75;
const LOW_RETURN_RANK: f64 = 0.20;
const HIGH_AMOUNT_RANK: f64 = 0.80;

/// Raw `market_cbond.daily_price` fields used by the seven remaining R88
/// kernels.  A dispatcher may set fields unused by a requested signal to NaN;
/// the kernel does not turn a missing required source field into a zero.
#[derive(Clone, Debug)]
pub struct TypedFactorR88DailyPriceRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    pub prev_close_price: f64,
    pub close_price: f64,
    pub amount: f64,
    pub deal: f64,
}

/// Raw `market_cbond.daily_base` fields used by the two bond/stock signals.
#[derive(Clone, Debug)]
pub struct TypedFactorR88DailyBaseRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    pub stk_prev_close_price: f64,
    pub stk_close_price: f64,
}

/// Raw `market_cbond.daily_twap` fields used by the completed-session TWAP
/// slope.  The signal uses `twap_1300_1330` and `twap_1400_1430` only.
#[derive(Clone, Debug)]
pub struct TypedFactorR88DailyTwapRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    pub twap_1300_1330: f64,
    pub twap_1400_1430: f64,
}

/// Complete strict-prior daily material for one score day.
#[derive(Clone, Debug)]
pub struct TypedFactorR88DailyContext {
    pub score_date: NaiveDate,
    pub price_rows: Vec<TypedFactorR88DailyPriceRow>,
    pub base_rows: Vec<TypedFactorR88DailyBaseRow>,
    pub twap_rows: Vec<TypedFactorR88DailyTwapRow>,
}

impl TypedFactorR88DailyContext {
    pub fn new(score_date: NaiveDate) -> Self {
        Self {
            score_date,
            price_rows: Vec::new(),
            base_rows: Vec::new(),
            twap_rows: Vec::new(),
        }
    }
}

/// Exact research signal names handled by this module.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TypedFactorR88DailySignal {
    BsabUpsideBeta60,
    BsabDownsideBeta60,
    BsctUpperTailDependence60,
    OsaTerminalAmountStreak60,
    DrrcReturnTradeSizeRankSpearman60,
    DrrqReturnAmountOppositeTailExcess60,
    DtwmSessionAfternoonLateLogSlope,
}

impl TypedFactorR88DailySignal {
    pub fn parse(signal: &str) -> Option<Self> {
        match signal {
            "bsab_upside_beta60" => Some(Self::BsabUpsideBeta60),
            "bsab_downside_beta60" => Some(Self::BsabDownsideBeta60),
            "bsct_upper_tail_dependence60" => Some(Self::BsctUpperTailDependence60),
            "osa_terminal_amount_streak60" => Some(Self::OsaTerminalAmountStreak60),
            "drrc_return_trade_size_rank_spearman60" => {
                Some(Self::DrrcReturnTradeSizeRankSpearman60)
            }
            "drrq_return_amount_opposite_tail_excess60" => {
                Some(Self::DrrqReturnAmountOppositeTailExcess60)
            }
            "dtwm_session_afternoon_late_log_slope" => Some(Self::DtwmSessionAfternoonLateLogSlope),
            _ => None,
        }
    }

    pub fn signal(self) -> &'static str {
        match self {
            Self::BsabUpsideBeta60 => "bsab_upside_beta60",
            Self::BsabDownsideBeta60 => "bsab_downside_beta60",
            Self::BsctUpperTailDependence60 => "bsct_upper_tail_dependence60",
            Self::OsaTerminalAmountStreak60 => "osa_terminal_amount_streak60",
            Self::DrrcReturnTradeSizeRankSpearman60 => "drrc_return_trade_size_rank_spearman60",
            Self::DrrqReturnAmountOppositeTailExcess60 => {
                "drrq_return_amount_opposite_tail_excess60"
            }
            Self::DtwmSessionAfternoonLateLogSlope => "dtwm_session_afternoon_late_log_slope",
        }
    }

    /// Research `params.family` that must remain paired with this signal at
    /// typed-kernel integration time.
    pub fn family(self) -> &'static str {
        match self {
            Self::BsabUpsideBeta60 | Self::BsabDownsideBeta60 => "prior_asymmetric_equity_beta",
            Self::BsctUpperTailDependence60 => "prior_bond_stock_copula_tail_dependence",
            Self::OsaTerminalAmountStreak60 => "prior_observable_market_seasoning",
            Self::DrrcReturnTradeSizeRankSpearman60 => "prior_relative_return_flow_rank_coupling",
            Self::DrrqReturnAmountOppositeTailExcess60 => {
                "prior_relative_return_flow_tail_contradiction"
            }
            Self::DtwmSessionAfternoonLateLogSlope => "prior_session_rotation_microstructure",
        }
    }

    /// Source columns to request for this exact signal.  This documentation is
    /// intentionally colocated with the pure implementation; it does not
    /// create a runtime dependency on a profile or dispatcher.
    pub fn source_contract(self) -> &'static str {
        match self {
            Self::BsabUpsideBeta60 | Self::BsabDownsideBeta60 | Self::BsctUpperTailDependence60 => {
                "daily_price(exchange_code,prev_close_price,close_price); daily_base(exchange_code,stk_prev_close_price,stk_close_price)"
            }
            Self::OsaTerminalAmountStreak60 => {
                "daily_price(exchange_code,close_price,amount)"
            }
            Self::DrrcReturnTradeSizeRankSpearman60 => {
                "daily_price(exchange_code,prev_close_price,close_price,amount,deal)"
            }
            Self::DrrqReturnAmountOppositeTailExcess60 => {
                "daily_price(exchange_code,prev_close_price,close_price,amount)"
            }
            Self::DtwmSessionAfternoonLateLogSlope => {
                "daily_price(exchange_code,close_price); daily_twap(exchange_code,twap_1300_1330,twap_1400_1430)"
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypedFactorR88DailyError {
    DuplicateStrictPriorDate {
        source: &'static str,
        code: String,
        trade_date: NaiveDate,
    },
    UnknownSignal(String),
}

impl fmt::Display for TypedFactorR88DailyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStrictPriorDate {
                source,
                code,
                trade_date,
            } => write!(
                f,
                "typed_factor R88 {source} has duplicate strict-prior row for {code} on {trade_date}"
            ),
            Self::UnknownSignal(signal) => write!(f, "unknown typed_factor R88 daily signal: {signal}"),
        }
    }
}

impl Error for TypedFactorR88DailyError {}

/// Immutable score-day cache for the R88 daily signals.  The public per-code
/// functions remain the direct formula reference; the typed dispatcher builds
/// this once and then reads each requested output from the prepared maps.
#[derive(Clone, Debug, Default)]
pub struct PreparedR88DailyValues {
    values: BTreeMap<TypedFactorR88DailySignal, BTreeMap<String, f64>>,
}

impl PreparedR88DailyValues {
    /// Return a prepared value for a requested signal.  A code absent from the
    /// prepared source universe has the same fail-closed `NaN` result as the
    /// direct formula.
    pub fn lookup(&self, signal: TypedFactorR88DailySignal, raw_panel_code: &str) -> Option<f64> {
        self.values.get(&signal).map(|by_code| {
            let code = canonical_market_code(raw_panel_code, "");
            by_code.get(&code).copied().unwrap_or(f64::NAN)
        })
    }
}

#[derive(Clone, Debug)]
struct PriceRow {
    trade_date: NaiveDate,
    code: String,
    prev_close_price: f64,
    close_price: f64,
    amount: f64,
    deal: f64,
}

#[derive(Clone, Debug)]
struct BaseRow {
    trade_date: NaiveDate,
    code: String,
    stk_prev_close_price: f64,
    stk_close_price: f64,
}

#[derive(Clone, Debug)]
struct TwapRow {
    trade_date: NaiveDate,
    code: String,
    twap_1300_1330: f64,
    twap_1400_1430: f64,
}

#[derive(Clone, Debug)]
struct CrossAssetRow {
    trade_date: NaiveDate,
    code: String,
    prev_close_price: f64,
    close_price: f64,
    stk_prev_close_price: f64,
    stk_close_price: f64,
}

#[derive(Clone, Debug)]
struct RankedRow {
    trade_date: NaiveDate,
    code: String,
    return_rank: f64,
    amount_rank: f64,
    trade_size_rank: f64,
}

fn canonical_exchange(value: &str) -> String {
    match value.trim().to_ascii_uppercase().as_str() {
        "XSHG" | "SHSE" => "SH".to_string(),
        "XSHE" | "SZSE" => "SZ".to_string(),
        "BSE" | "BJSE" => "BJ".to_string(),
        other => other.to_string(),
    }
}

fn valid_market_exchange(value: &str) -> bool {
    matches!(value, "SH" | "SZ" | "BJ")
}

/// Exact Python-family daily-code normalization: a code needs either a valid
/// suffix or a valid supplied exchange.  It never guesses a market suffix.
fn canonical_market_code(value: &str, exchange: &str) -> String {
    let mut text = value.trim().to_ascii_uppercase();
    if text.is_empty() || text == "NAN" {
        return String::new();
    }
    if text.ends_with(".0") {
        text.truncate(text.len() - 2);
    }
    if let Some((bare, suffix)) = text.rsplit_once('.') {
        let suffix = canonical_exchange(suffix);
        if !bare.is_empty() && valid_market_exchange(&suffix) {
            return format!("{bare}.{suffix}");
        }
    }
    let suffix = canonical_exchange(exchange);
    if valid_market_exchange(&suffix) {
        format!("{text}.{suffix}")
    } else {
        String::new()
    }
}

fn strict_price_rows(
    ctx: &TypedFactorR88DailyContext,
) -> Result<Vec<PriceRow>, TypedFactorR88DailyError> {
    let mut rows: Vec<_> = ctx
        .price_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .filter_map(|row| {
            let code = canonical_market_code(&row.code, &row.exchange_code);
            (!code.is_empty()).then_some(PriceRow {
                trade_date: row.trade_date,
                code,
                prev_close_price: row.prev_close_price,
                close_price: row.close_price,
                amount: row.amount,
                deal: row.deal,
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
            return Err(TypedFactorR88DailyError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_price",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

fn strict_base_rows(
    ctx: &TypedFactorR88DailyContext,
) -> Result<Vec<BaseRow>, TypedFactorR88DailyError> {
    let mut rows: Vec<_> = ctx
        .base_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .filter_map(|row| {
            let code = canonical_market_code(&row.code, &row.exchange_code);
            (!code.is_empty()).then_some(BaseRow {
                trade_date: row.trade_date,
                code,
                stk_prev_close_price: row.stk_prev_close_price,
                stk_close_price: row.stk_close_price,
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
            return Err(TypedFactorR88DailyError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_base",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

fn strict_twap_rows(
    ctx: &TypedFactorR88DailyContext,
) -> Result<Vec<TwapRow>, TypedFactorR88DailyError> {
    let mut rows: Vec<_> = ctx
        .twap_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .filter_map(|row| {
            let code = canonical_market_code(&row.code, &row.exchange_code);
            (!code.is_empty()).then_some(TwapRow {
                trade_date: row.trade_date,
                code,
                twap_1300_1330: row.twap_1300_1330,
                twap_1400_1430: row.twap_1400_1430,
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
            return Err(TypedFactorR88DailyError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_twap",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

fn finite_log_return(close: f64, previous_close: f64) -> f64 {
    if close.is_finite() && previous_close.is_finite() && close > EPS && previous_close > EPS {
        let value = (close / previous_close).ln();
        value.is_finite().then_some(value).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

fn python_rank_input_log_ratio(numerator: f64, denominator: f64) -> f64 {
    // The Python rank families set the result after `Series.gt(EPS)`, then use
    // NumPy log.  Infinity is therefore rankable here and is rejected only by
    // the later `np.isfinite` joint-observation gate.
    if numerator > EPS && denominator > EPS {
        (numerator / denominator).ln()
    } else {
        f64::NAN
    }
}

fn python_rank_input_log(value: f64) -> f64 {
    if value > EPS {
        value.ln()
    } else {
        f64::NAN
    }
}

fn prepared_cross_asset_rows(
    ctx: &TypedFactorR88DailyContext,
) -> Result<Option<(Vec<CrossAssetRow>, NaiveDate)>, TypedFactorR88DailyError> {
    let price = strict_price_rows(ctx)?;
    let base = strict_base_rows(ctx)?;
    Ok(prepared_cross_asset_rows_from_rows(&price, &base))
}

/// Build the strict-prior joined panel once.  The row sort and join order are
/// intentionally identical to the direct-formula path above.
fn prepared_cross_asset_rows_from_rows(
    price: &[PriceRow],
    base: &[BaseRow],
) -> Option<(Vec<CrossAssetRow>, NaiveDate)> {
    let Some(price_anchor) = price.iter().map(|row| row.trade_date).max() else {
        return None;
    };
    let Some(base_anchor) = base.iter().map(|row| row.trade_date).max() else {
        return None;
    };
    if price_anchor != base_anchor {
        return None;
    }
    let base_by_key: BTreeMap<(NaiveDate, String), &BaseRow> = base
        .iter()
        .map(|row| ((row.trade_date, row.code.clone()), row))
        .collect();
    let mut joined = Vec::new();
    for price_row in price {
        let Some(base_row) = base_by_key.get(&(price_row.trade_date, price_row.code.clone()))
        else {
            continue;
        };
        joined.push(CrossAssetRow {
            trade_date: price_row.trade_date,
            code: price_row.code.clone(),
            prev_close_price: price_row.prev_close_price,
            close_price: price_row.close_price,
            stk_prev_close_price: base_row.stk_prev_close_price,
            stk_close_price: base_row.stk_close_price,
        });
    }
    joined.sort_by(|left, right| {
        left.code
            .cmp(&right.code)
            .then_with(|| left.trade_date.cmp(&right.trade_date))
    });
    Some((joined, price_anchor))
}

fn target_cross_asset_path_from_rows(
    joined: &[CrossAssetRow],
    anchor: NaiveDate,
    raw_panel_code: &str,
) -> Option<Vec<CrossAssetRow>> {
    let code = canonical_market_code(raw_panel_code, "");
    if code.is_empty() {
        return None;
    }
    let path: Vec<_> = joined
        .iter()
        .filter(|row| row.code == code)
        .cloned()
        .collect();
    if path.last().map(|row| row.trade_date) != Some(anchor) {
        return None;
    }
    let first = path.len().saturating_sub(WINDOW);
    Some(path[first..].to_vec())
}

fn cross_asset_paths_by_code(joined: Vec<CrossAssetRow>) -> BTreeMap<String, Vec<CrossAssetRow>> {
    let mut paths = BTreeMap::new();
    for row in joined {
        paths
            .entry(row.code.clone())
            .or_insert_with(Vec::new)
            .push(row);
    }
    paths
}

fn target_cross_asset_path_from_group(
    path: &[CrossAssetRow],
    anchor: NaiveDate,
) -> Option<&[CrossAssetRow]> {
    if path.last().map(|row| row.trade_date) != Some(anchor) {
        return None;
    }
    let first = path.len().saturating_sub(WINDOW);
    Some(&path[first..])
}

fn target_cross_asset_path(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<Option<Vec<CrossAssetRow>>, TypedFactorR88DailyError> {
    let Some((joined, anchor)) = prepared_cross_asset_rows(ctx)? else {
        return Ok(None);
    };
    Ok(target_cross_asset_path_from_rows(
        &joined,
        anchor,
        raw_panel_code,
    ))
}

fn ordinary_least_squares_slope(left: &[f64], right: &[f64], min_observations: usize) -> f64 {
    if left.len() != right.len() || left.len() < min_observations {
        return f64::NAN;
    }
    if left.iter().any(|value| !value.is_finite()) || right.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut denominator = 0.0;
    let mut numerator = 0.0;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let centered_left = *left_value - left_mean;
        denominator += centered_left * centered_left;
        numerator += centered_left * (*right_value - right_mean);
    }
    if !denominator.is_finite() || denominator <= EPS {
        return f64::NAN;
    }
    let value = numerator / denominator;
    value.is_finite().then_some(value).unwrap_or(f64::NAN)
}

/// OLS beta of bond log return on negative-stock-return observations in the
/// latest up-to-60 strict-prior *joined* price/base rows.
pub fn bsab_downside_beta60(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88DailyError> {
    asymmetric_beta(ctx, raw_panel_code, false)
}

/// OLS beta of bond log return on positive-stock-return observations in the
/// latest up-to-60 strict-prior *joined* price/base rows.
pub fn bsab_upside_beta60(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88DailyError> {
    asymmetric_beta(ctx, raw_panel_code, true)
}

fn asymmetric_beta(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
    positive_stock_leg: bool,
) -> Result<f64, TypedFactorR88DailyError> {
    let Some(path) = target_cross_asset_path(ctx, raw_panel_code)? else {
        return Ok(f64::NAN);
    };
    Ok(asymmetric_beta_from_path(&path, positive_stock_leg))
}

fn asymmetric_beta_from_path(path: &[CrossAssetRow], positive_stock_leg: bool) -> f64 {
    let returns: Vec<(f64, f64)> = path
        .iter()
        .map(|row| {
            (
                finite_log_return(row.close_price, row.prev_close_price),
                finite_log_return(row.stk_close_price, row.stk_prev_close_price),
            )
        })
        .collect();
    let Some((terminal_bond, terminal_stock)) = returns.last().copied() else {
        return f64::NAN;
    };
    // The Python implementation treats this latest strict-prior joint state
    // as an anchor; a malformed terminal state cannot be compacted away.
    if !terminal_bond.is_finite() || !terminal_stock.is_finite() {
        return f64::NAN;
    }
    let finite: Vec<_> = returns
        .into_iter()
        .filter(|(bond, stock)| bond.is_finite() && stock.is_finite())
        .collect();
    if finite.len() < MIN_JOINT_OBSERVATIONS {
        return f64::NAN;
    }
    let selected: Vec<_> = finite
        .into_iter()
        .filter(|(_, stock)| {
            if positive_stock_leg {
                *stock > 0.0
            } else {
                *stock < 0.0
            }
        })
        .collect();
    if selected.len() < MIN_CONDITIONAL_OBSERVATIONS {
        return f64::NAN;
    }
    let stock: Vec<f64> = selected.iter().map(|(_, stock)| *stock).collect();
    let bond: Vec<f64> = selected.iter().map(|(bond, _)| *bond).collect();
    ordinary_least_squares_slope(&stock, &bond, MIN_CONDITIONAL_OBSERVATIONS)
}

fn average_rank(values: &[f64]) -> Vec<f64> {
    let mut ranks = vec![f64::NAN; values.len()];
    let mut ordered: Vec<_> = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| (!value.is_nan()).then_some((value, index)))
        .collect();
    ordered.sort_by(|(left, _), (right, _)| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let mut start = 0;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && ordered[end].0 == ordered[start].0 {
            end += 1;
        }
        // One-based average rank: (start + 1 + end) / 2.
        let rank = (start + 1 + end) as f64 * 0.5;
        for &(_, index) in &ordered[start..end] {
            ranks[index] = rank;
        }
        start = end;
    }
    ranks
}

/// Empirical P(bond return rank >= q75 | stock return rank >= q75) over the
/// latest up-to-60 strict-prior joint observations.  The complete family gate
/// still requires eight observations in *both* stock tails, matching Python
/// even though this concrete output uses only the upper tail.
pub fn bsct_upper_tail_dependence60(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88DailyError> {
    let Some(path) = target_cross_asset_path(ctx, raw_panel_code)? else {
        return Ok(f64::NAN);
    };
    Ok(bsct_upper_tail_dependence60_from_path(&path))
}

fn bsct_upper_tail_dependence60_from_path(path: &[CrossAssetRow]) -> f64 {
    let returns: Vec<(f64, f64)> = path
        .iter()
        .map(|row| {
            (
                finite_log_return(row.close_price, row.prev_close_price),
                finite_log_return(row.stk_close_price, row.stk_prev_close_price),
            )
        })
        .collect();
    let Some((terminal_bond, terminal_stock)) = returns.last().copied() else {
        return f64::NAN;
    };
    if !terminal_bond.is_finite() || !terminal_stock.is_finite() {
        return f64::NAN;
    }
    let finite: Vec<_> = returns
        .into_iter()
        .filter(|(bond, stock)| bond.is_finite() && stock.is_finite())
        .collect();
    if finite.len() < MIN_JOINT_OBSERVATIONS {
        return f64::NAN;
    }
    let bond: Vec<f64> = finite.iter().map(|(value, _)| *value).collect();
    let stock: Vec<f64> = finite.iter().map(|(_, value)| *value).collect();
    let bond_min = bond.iter().copied().fold(f64::INFINITY, f64::min);
    let bond_max = bond.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let stock_min = stock.iter().copied().fold(f64::INFINITY, f64::min);
    let stock_max = stock.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if bond_max - bond_min <= EPS || stock_max - stock_min <= EPS {
        return f64::NAN;
    }
    // SciPy `rankdata(method="average") / (n + 1)`.
    let denominator = bond.len() as f64 + 1.0;
    let bond_rank: Vec<f64> = average_rank(&bond)
        .into_iter()
        .map(|rank| rank / denominator)
        .collect();
    let stock_rank: Vec<f64> = average_rank(&stock)
        .into_iter()
        .map(|rank| rank / denominator)
        .collect();
    let upper_stock: Vec<usize> = stock_rank
        .iter()
        .enumerate()
        .filter_map(|(index, rank)| (*rank >= UPPER_COPULA_TAIL).then_some(index))
        .collect();
    let lower_stock_count = stock_rank
        .iter()
        .filter(|rank| **rank <= LOWER_COPULA_TAIL)
        .count();
    if upper_stock.len() < MIN_TAIL_OBSERVATIONS || lower_stock_count < MIN_TAIL_OBSERVATIONS {
        return f64::NAN;
    }
    let upper = upper_stock
        .iter()
        .filter(|index| bond_rank[**index] >= UPPER_COPULA_TAIL)
        .count() as f64
        / upper_stock.len() as f64;
    upper.is_finite().then_some(upper).unwrap_or(f64::NAN)
}

/// Number of consecutive latest strict-prior source sessions (capped at 60)
/// with positive amount.  The terminal session must have a valid close; an
/// observed zero amount is a formula-defined `0.0`, not an imputed missing.
pub fn osa_terminal_amount_streak60(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88DailyError> {
    let price = strict_price_rows(ctx)?;
    Ok(osa_terminal_amount_streak60_from_price(
        &price,
        raw_panel_code,
    ))
}

fn osa_terminal_amount_streak60_from_price(price: &[PriceRow], raw_panel_code: &str) -> f64 {
    let (calendar, by_code) = osa_index(price);
    let code = canonical_market_code(raw_panel_code, "");
    if code.is_empty() {
        return f64::NAN;
    }
    osa_terminal_amount_streak60_from_index(&calendar, by_code.get(&code))
}

fn osa_index<'a>(
    price: &'a [PriceRow],
) -> (
    Vec<NaiveDate>,
    BTreeMap<String, BTreeMap<NaiveDate, &'a PriceRow>>,
) {
    let dates: BTreeSet<_> = price.iter().map(|row| row.trade_date).collect();
    let mut by_code = BTreeMap::new();
    for row in price {
        by_code
            .entry(row.code.clone())
            .or_insert_with(BTreeMap::new)
            .insert(row.trade_date, row);
    }
    (dates.into_iter().collect(), by_code)
}

fn osa_terminal_amount_streak60_from_index(
    calendar: &[NaiveDate],
    by_date: Option<&BTreeMap<NaiveDate, &PriceRow>>,
) -> f64 {
    let Some(anchor) = calendar.last().copied() else {
        return f64::NAN;
    };
    let Some(by_date) = by_date else {
        return f64::NAN;
    };
    if by_date.last_key_value().map(|(date, _)| *date) != Some(anchor) {
        return f64::NAN;
    }
    let first = calendar.len().saturating_sub(WINDOW);
    let recent = &calendar[first..];
    let terminal_observed = by_date
        .get(recent.last().expect("nonempty anchor calendar"))
        .map(|row| row.close_price.is_finite() && row.close_price > EPS)
        .unwrap_or(false);
    if !terminal_observed {
        return f64::NAN;
    }
    let mut streak = 0usize;
    for date in recent.iter().rev() {
        let active = by_date
            .get(date)
            .map(|row| row.amount.is_finite() && row.amount > EPS)
            .unwrap_or(false);
        if !active {
            break;
        }
        streak += 1;
    }
    streak as f64
}

fn ranked_price_rows(
    ctx: &TypedFactorR88DailyContext,
) -> Result<(Vec<RankedRow>, Vec<NaiveDate>), TypedFactorR88DailyError> {
    let price = strict_price_rows(ctx)?;
    Ok(ranked_price_rows_from_rows(&price))
}

/// Preserve the exact daily/date/code iteration order used by the direct
/// rank formulas while allowing one score-day rank build to serve both
/// requested rank signals.
fn ranked_price_rows_from_rows(price: &[PriceRow]) -> (Vec<RankedRow>, Vec<NaiveDate>) {
    let mut by_date: BTreeMap<NaiveDate, Vec<&PriceRow>> = BTreeMap::new();
    for row in price {
        by_date.entry(row.trade_date).or_default().push(row);
    }
    let mut output = Vec::new();
    for (trade_date, rows) in &by_date {
        let returns: Vec<_> = rows
            .iter()
            .map(|row| python_rank_input_log_ratio(row.close_price, row.prev_close_price))
            .collect();
        let amounts: Vec<_> = rows
            .iter()
            .map(|row| python_rank_input_log(row.amount))
            .collect();
        let trade_sizes: Vec<_> = rows
            .iter()
            .map(|row| python_rank_input_log_ratio(row.amount, row.deal))
            .collect();
        let return_rank = average_rank(&returns);
        let amount_rank = average_rank(&amounts);
        let trade_size_rank = average_rank(&trade_sizes);
        let return_denominator = returns.iter().filter(|value| !value.is_nan()).count() as f64;
        let amount_denominator = amounts.iter().filter(|value| !value.is_nan()).count() as f64;
        let trade_size_denominator =
            trade_sizes.iter().filter(|value| !value.is_nan()).count() as f64;
        for index in 0..rows.len() {
            output.push(RankedRow {
                trade_date: *trade_date,
                code: rows[index].code.clone(),
                return_rank: if return_denominator > 0.0 {
                    return_rank[index] / return_denominator
                } else {
                    f64::NAN
                },
                amount_rank: if amount_denominator > 0.0 {
                    amount_rank[index] / amount_denominator
                } else {
                    f64::NAN
                },
                trade_size_rank: if trade_size_denominator > 0.0 {
                    trade_size_rank[index] / trade_size_denominator
                } else {
                    f64::NAN
                },
            });
        }
    }
    let calendar: Vec<_> = by_date.keys().copied().collect();
    (output, calendar)
}

fn target_rank_path(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<Option<Vec<RankedRow>>, TypedFactorR88DailyError> {
    let (ranked, calendar) = ranked_price_rows(ctx)?;
    Ok(target_rank_path_from_ranked(
        &ranked,
        &calendar,
        raw_panel_code,
    ))
}

fn target_rank_path_from_ranked(
    ranked: &[RankedRow],
    calendar: &[NaiveDate],
    raw_panel_code: &str,
) -> Option<Vec<RankedRow>> {
    let Some(anchor) = calendar.last().copied() else {
        return None;
    };
    let code = canonical_market_code(raw_panel_code, "");
    if code.is_empty() {
        return None;
    }
    let mut path: Vec<_> = ranked
        .iter()
        .filter(|row| row.code == code)
        .cloned()
        .collect();
    path.sort_by_key(|row| row.trade_date);
    if path.last().map(|row| row.trade_date) != Some(anchor) {
        return None;
    }
    let anchor_position = calendar.len() - 1;
    let first_position = anchor_position.saturating_sub(WINDOW - 1);
    let positions: BTreeMap<_, _> = calendar
        .iter()
        .copied()
        .enumerate()
        .map(|(position, date)| (date, position))
        .collect();
    path.retain(|row| {
        positions
            .get(&row.trade_date)
            .copied()
            .unwrap_or(usize::MAX)
            >= first_position
    });
    Some(path)
}

fn ranked_paths_by_code(ranked: Vec<RankedRow>) -> BTreeMap<String, Vec<RankedRow>> {
    let mut paths = BTreeMap::new();
    for row in ranked {
        paths
            .entry(row.code.clone())
            .or_insert_with(Vec::new)
            .push(row);
    }
    paths
}

fn target_rank_path_from_group<'a>(
    path: &'a [RankedRow],
    calendar: &[NaiveDate],
) -> Option<&'a [RankedRow]> {
    let anchor = calendar.last().copied()?;
    if path.last().map(|row| row.trade_date) != Some(anchor) {
        return None;
    }
    let first_position = calendar.len().saturating_sub(WINDOW);
    let first_date = calendar[first_position];
    let first = path
        .iter()
        .position(|row| row.trade_date >= first_date)
        .unwrap_or(path.len());
    Some(&path[first..])
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < MIN_JOINT_OBSERVATIONS {
        return f64::NAN;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut numerator = 0.0;
    let mut left_sum_squares = 0.0;
    let mut right_sum_squares = 0.0;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let centered_left = *left_value - left_mean;
        let centered_right = *right_value - right_mean;
        numerator += centered_left * centered_right;
        left_sum_squares += centered_left * centered_left;
        right_sum_squares += centered_right * centered_right;
    }
    let denominator = (left_sum_squares * right_sum_squares).sqrt();
    if !denominator.is_finite() || denominator <= EPS {
        return f64::NAN;
    }
    let value = numerator / denominator;
    value.is_finite().then_some(value).unwrap_or(f64::NAN)
}

/// Pearson correlation of daily full-market percentile ranks of bond return
/// and `amount / deal`, over the last up-to-60 strict-prior market sessions.
pub fn drrc_return_trade_size_rank_spearman60(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88DailyError> {
    let Some(path) = target_rank_path(ctx, raw_panel_code)? else {
        return Ok(f64::NAN);
    };
    Ok(drrc_return_trade_size_rank_spearman60_from_path(&path))
}

fn drrc_return_trade_size_rank_spearman60_from_path(path: &[RankedRow]) -> f64 {
    let Some(terminal) = path.last() else {
        return f64::NAN;
    };
    if !terminal.return_rank.is_finite() || !terminal.trade_size_rank.is_finite() {
        return f64::NAN;
    }
    let pairs: Vec<_> = path
        .iter()
        .filter_map(|row| {
            (row.return_rank.is_finite() && row.trade_size_rank.is_finite())
                .then_some((row.return_rank, row.trade_size_rank))
        })
        .collect();
    if pairs.len() < MIN_JOINT_OBSERVATIONS {
        return f64::NAN;
    }
    let left: Vec<_> = pairs.iter().map(|(left, _)| *left).collect();
    let right: Vec<_> = pairs.iter().map(|(_, right)| *right).collect();
    pearson(&left, &right)
}

/// Mean strict-prior occurrence rate of a bottom-return-rank / top-amount-rank
/// event, less its explicit independence baseline (`0.20 * (1 - 0.80)`).
pub fn drrq_return_amount_opposite_tail_excess60(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88DailyError> {
    let Some(path) = target_rank_path(ctx, raw_panel_code)? else {
        return Ok(f64::NAN);
    };
    Ok(drrq_return_amount_opposite_tail_excess60_from_path(&path))
}

fn drrq_return_amount_opposite_tail_excess60_from_path(path: &[RankedRow]) -> f64 {
    let Some(terminal) = path.last() else {
        return f64::NAN;
    };
    if !terminal.return_rank.is_finite() || !terminal.amount_rank.is_finite() {
        return f64::NAN;
    }
    let pairs: Vec<_> = path
        .iter()
        .filter_map(|row| {
            (row.return_rank.is_finite() && row.amount_rank.is_finite())
                .then_some((row.return_rank, row.amount_rank))
        })
        .collect();
    if pairs.len() < MIN_JOINT_OBSERVATIONS {
        return f64::NAN;
    }
    let count = pairs
        .iter()
        .filter(|(return_rank, amount_rank)| {
            *return_rank <= LOW_RETURN_RANK && *amount_rank >= HIGH_AMOUNT_RANK
        })
        .count();
    let event_rate = count as f64 / pairs.len() as f64;
    let value = event_rate - LOW_RETURN_RANK * (1.0 - HIGH_AMOUNT_RANK);
    value.is_finite().then_some(value).unwrap_or(f64::NAN)
}

/// Completed-session afternoon log slope:
/// `log(twap[14:00,14:30] / twap[13:00,13:30])` on the latest strict-prior
/// valid-price / TWAP joined session.
pub fn dtwm_session_afternoon_late_log_slope(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorR88DailyError> {
    let price = strict_price_rows(ctx)?;
    let twap = strict_twap_rows(ctx)?;
    Ok(dtwm_session_afternoon_late_log_slope_from_rows(
        &price,
        &twap,
        raw_panel_code,
    ))
}

fn dtwm_session_afternoon_late_log_slope_from_rows(
    price: &[PriceRow],
    twap: &[TwapRow],
    raw_panel_code: &str,
) -> f64 {
    let (anchor, paths) = dtwm_paths(price, twap);
    let code = canonical_market_code(raw_panel_code, "");
    if code.is_empty() {
        return f64::NAN;
    }
    dtwm_session_afternoon_late_log_slope_from_path(anchor, paths.get(&code).map(Vec::as_slice))
}

fn dtwm_paths<'a>(
    price: &'a [PriceRow],
    twap: &'a [TwapRow],
) -> (
    Option<NaiveDate>,
    BTreeMap<String, Vec<(NaiveDate, &'a TwapRow)>>,
) {
    // Python first discards invalid completed price rows before it builds the
    // market session calendar and price/TWAP join.
    let valid_price: Vec<_> = price
        .iter()
        .filter(|row| row.close_price.is_finite() && row.close_price > EPS)
        .collect();
    let anchor = valid_price.iter().map(|row| row.trade_date).max();
    let twap_by_key: BTreeMap<(NaiveDate, String), &TwapRow> = twap
        .iter()
        .map(|row| ((row.trade_date, row.code.clone()), row))
        .collect();
    let mut paths = BTreeMap::new();
    for price_row in valid_price {
        if let Some(twap_row) = twap_by_key.get(&(price_row.trade_date, price_row.code.clone())) {
            paths
                .entry(price_row.code.clone())
                .or_insert_with(Vec::new)
                .push((price_row.trade_date, *twap_row));
        }
    }
    for path in paths.values_mut() {
        path.sort_by_key(|(trade_date, _)| *trade_date);
    }
    (anchor, paths)
}

fn dtwm_session_afternoon_late_log_slope_from_path(
    anchor: Option<NaiveDate>,
    path: Option<&[(NaiveDate, &TwapRow)]>,
) -> f64 {
    let Some(anchor) = anchor else {
        return f64::NAN;
    };
    let Some(path) = path else {
        return f64::NAN;
    };
    let Some((terminal_date, terminal_twap)) = path.last().copied() else {
        return f64::NAN;
    };
    if terminal_date != anchor {
        return f64::NAN;
    }
    let numerator = terminal_twap.twap_1400_1430;
    let denominator = terminal_twap.twap_1300_1330;
    if !numerator.is_finite() || !denominator.is_finite() || numerator <= EPS || denominator <= EPS
    {
        return f64::NAN;
    }
    let value = (numerator / denominator).ln();
    value.is_finite().then_some(value).unwrap_or(f64::NAN)
}

/// Prepare all requested daily R88 output maps for one score-day dispatch.
/// Strict-prior rows are canonicalised, sorted, duplicate-checked, joined and
/// ranked only once per relevant source family; each output is then an O(1)
/// lookup from [`PreparedR88DailyValues`].  The public direct functions remain
/// the reference implementation for parity tests and isolated callers.
pub fn prepare_r88_daily_values<I, J>(
    ctx: &TypedFactorR88DailyContext,
    requested_signals: I,
    raw_panel_codes: J,
) -> Result<PreparedR88DailyValues, TypedFactorR88DailyError>
where
    I: IntoIterator<Item = TypedFactorR88DailySignal>,
    J: IntoIterator<Item = String>,
{
    let requested: BTreeSet<_> = requested_signals.into_iter().collect();
    if requested.is_empty() {
        return Ok(PreparedR88DailyValues::default());
    }
    let codes: BTreeSet<_> = raw_panel_codes
        .into_iter()
        .map(|raw| canonical_market_code(&raw, ""))
        .filter(|code| !code.is_empty())
        .collect();
    // Every concrete R88 daily formula consumes strict-prior daily price;
    // build it once even when a requested panel code later fails canonical
    // validation so duplicate-source errors remain fail-closed.
    let price = strict_price_rows(ctx)?;
    let needs_cross_asset = requested.iter().any(|signal| {
        matches!(
            signal,
            TypedFactorR88DailySignal::BsabUpsideBeta60
                | TypedFactorR88DailySignal::BsabDownsideBeta60
                | TypedFactorR88DailySignal::BsctUpperTailDependence60
        )
    });
    let cross_asset = if needs_cross_asset {
        let base = strict_base_rows(ctx)?;
        prepared_cross_asset_rows_from_rows(&price, &base)
            .map(|(joined, anchor)| (anchor, cross_asset_paths_by_code(joined)))
    } else {
        None
    };
    let needs_rank = requested.iter().any(|signal| {
        matches!(
            signal,
            TypedFactorR88DailySignal::DrrcReturnTradeSizeRankSpearman60
                | TypedFactorR88DailySignal::DrrqReturnAmountOppositeTailExcess60
        )
    });
    let ranked = needs_rank.then(|| {
        let (rows, calendar) = ranked_price_rows_from_rows(&price);
        (calendar, ranked_paths_by_code(rows))
    });
    let osa = requested
        .contains(&TypedFactorR88DailySignal::OsaTerminalAmountStreak60)
        .then(|| osa_index(&price));
    let twap = if requested.contains(&TypedFactorR88DailySignal::DtwmSessionAfternoonLateLogSlope) {
        Some(strict_twap_rows(ctx)?)
    } else {
        None
    };
    let dtwm = twap.as_ref().map(|twap| dtwm_paths(&price, twap));

    let mut values = BTreeMap::new();
    for signal in requested {
        let mut by_code = BTreeMap::new();
        for code in &codes {
            let value = match signal {
                TypedFactorR88DailySignal::BsabUpsideBeta60 => cross_asset
                    .as_ref()
                    .and_then(|(anchor, paths)| {
                        paths
                            .get(code)
                            .and_then(|path| target_cross_asset_path_from_group(path, *anchor))
                    })
                    .map(|path| asymmetric_beta_from_path(path, true))
                    .unwrap_or(f64::NAN),
                TypedFactorR88DailySignal::BsabDownsideBeta60 => cross_asset
                    .as_ref()
                    .and_then(|(anchor, paths)| {
                        paths
                            .get(code)
                            .and_then(|path| target_cross_asset_path_from_group(path, *anchor))
                    })
                    .map(|path| asymmetric_beta_from_path(path, false))
                    .unwrap_or(f64::NAN),
                TypedFactorR88DailySignal::BsctUpperTailDependence60 => cross_asset
                    .as_ref()
                    .and_then(|(anchor, paths)| {
                        paths
                            .get(code)
                            .and_then(|path| target_cross_asset_path_from_group(path, *anchor))
                    })
                    .map(bsct_upper_tail_dependence60_from_path)
                    .unwrap_or(f64::NAN),
                TypedFactorR88DailySignal::OsaTerminalAmountStreak60 => osa
                    .as_ref()
                    .map(|(calendar, paths)| {
                        osa_terminal_amount_streak60_from_index(calendar, paths.get(code))
                    })
                    .unwrap_or(f64::NAN),
                TypedFactorR88DailySignal::DrrcReturnTradeSizeRankSpearman60 => ranked
                    .as_ref()
                    .and_then(|(calendar, paths)| {
                        paths
                            .get(code)
                            .and_then(|path| target_rank_path_from_group(path, calendar))
                    })
                    .map(drrc_return_trade_size_rank_spearman60_from_path)
                    .unwrap_or(f64::NAN),
                TypedFactorR88DailySignal::DrrqReturnAmountOppositeTailExcess60 => ranked
                    .as_ref()
                    .and_then(|(calendar, paths)| {
                        paths
                            .get(code)
                            .and_then(|path| target_rank_path_from_group(path, calendar))
                    })
                    .map(drrq_return_amount_opposite_tail_excess60_from_path)
                    .unwrap_or(f64::NAN),
                TypedFactorR88DailySignal::DtwmSessionAfternoonLateLogSlope => dtwm
                    .as_ref()
                    .map(|(anchor, paths)| {
                        dtwm_session_afternoon_late_log_slope_from_path(
                            *anchor,
                            paths.get(code).map(Vec::as_slice),
                        )
                    })
                    .unwrap_or(f64::NAN),
            };
            by_code.insert(code.clone(), value);
        }
        values.insert(signal, by_code);
    }
    Ok(PreparedR88DailyValues { values })
}

/// Direct reference hook for the integrated research typed-factor dispatcher
/// and isolated parity callers.  Prepared dispatches use
/// [`prepare_r88_daily_values`], while this function retains the exact
/// per-code formula boundary for verification.
pub fn compute_r88_daily_signal(
    ctx: &TypedFactorR88DailyContext,
    raw_panel_code: &str,
    signal: &str,
) -> Result<f64, TypedFactorR88DailyError> {
    let Some(signal) = TypedFactorR88DailySignal::parse(signal) else {
        return Err(TypedFactorR88DailyError::UnknownSignal(signal.to_string()));
    };
    match signal {
        TypedFactorR88DailySignal::BsabUpsideBeta60 => bsab_upside_beta60(ctx, raw_panel_code),
        TypedFactorR88DailySignal::BsabDownsideBeta60 => bsab_downside_beta60(ctx, raw_panel_code),
        TypedFactorR88DailySignal::BsctUpperTailDependence60 => {
            bsct_upper_tail_dependence60(ctx, raw_panel_code)
        }
        TypedFactorR88DailySignal::OsaTerminalAmountStreak60 => {
            osa_terminal_amount_streak60(ctx, raw_panel_code)
        }
        TypedFactorR88DailySignal::DrrcReturnTradeSizeRankSpearman60 => {
            drrc_return_trade_size_rank_spearman60(ctx, raw_panel_code)
        }
        TypedFactorR88DailySignal::DrrqReturnAmountOppositeTailExcess60 => {
            drrq_return_amount_opposite_tail_excess60(ctx, raw_panel_code)
        }
        TypedFactorR88DailySignal::DtwmSessionAfternoonLateLogSlope => {
            dtwm_session_afternoon_late_log_slope(ctx, raw_panel_code)
        }
    }
}
