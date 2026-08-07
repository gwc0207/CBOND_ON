//! Typed, pure-Rust kernels for daily-path factor contracts.
//!
//! The ordinary dispatcher includes this module through the typed-kernel
//! boundary. Exact parity is verified before a contract is admitted.
//!
//! ## Source, calendar, terminal, and missing-value contracts
//!
//! * All contexts hold already canonicalised market-code keys.  The eventual
//!   parser must keep `exchange_code` long enough to canonicalise the raw
//!   DataHub code exactly as Python does; this typed layer intentionally does
//!   not guess exchanges or merge aliases.
//! * Every source uses only `trade_date < score_date`.  Score-day and future
//!   rows are ignored, never used as a fallback.
//! * Duplicate strict-prior `(trade_date, code)` records are an explicit
//!   error.  Missing values are never replaced by zero.
//! * The two `dohw_*` signals use the global strict-prior `daily_price`
//!   calendar.  A code must reach that common terminal date; its 60-session
//!   window is measured in that global calendar, not by a sparse code-local
//!   row count.  Their terminal OHLC state must be finite after the Python
//!   formula is evaluated.
//! * `dret_drawup_drawdown_asym` follows the older catalogue's finite-only
//!   tail convention.  It has no global-anchor or terminal-observation gate:
//!   invalid returns are discarded before taking the last 20 finite values.
//! * `bstk_tail_cocrash_residual20` requires the independent `daily_price`
//!   anchor and a current `daily_base` row.  Its last 41 joined observations
//!   must occupy 41 adjacent global price-calendar sessions and all four
//!   supplied price fields must be finite and positive.

use chrono::NaiveDate;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const DAILY_PATHS_EPS: f64 = 1e-12;
pub const DOHW_WINDOW: usize = 60;
pub const DOHW_MIN_OBSERVATIONS: usize = 45;
pub const DOHW_MIN_DIRECTIONAL_OBSERVATIONS: usize = 8;
pub const DRET_WINDOW: usize = 20;
pub const DRET_MIN_OBSERVATIONS: usize = 15;
pub const BSTK_COMPLETE_WINDOW: usize = 41;

/// A strict-prior `market_cbond.daily_price` row for the OHLC/path kernels.
///
/// `act_prev_close_price` is intentionally separate from `prev_close_price`:
/// the catalogue's daily-return rule uses the former only when it is strictly
/// positive, otherwise it falls back to the latter.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyPathPriceRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub prev_close_price: f64,
    pub act_prev_close_price: f64,
    pub open_price: f64,
    pub high_price: f64,
    pub low_price: f64,
    pub close_price: f64,
}

impl TypedFactorDailyPathPriceRow {
    pub fn empty(trade_date: NaiveDate, code: impl Into<String>) -> Self {
        Self {
            trade_date,
            code: code.into(),
            prev_close_price: f64::NAN,
            act_prev_close_price: f64::NAN,
            open_price: f64::NAN,
            high_price: f64::NAN,
            low_price: f64::NAN,
            close_price: f64::NAN,
        }
    }
}

/// Daily-price context for `dohw_*` and `dret_drawup_drawdown_asym`.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyPathContext {
    pub score_date: NaiveDate,
    pub price_by_code: BTreeMap<String, Vec<TypedFactorDailyPathPriceRow>>,
}

impl TypedFactorDailyPathContext {
    pub fn new(score_date: NaiveDate) -> Self {
        Self {
            score_date,
            price_by_code: BTreeMap::new(),
        }
    }
}

/// The independent daily-price calendar is used by the tracking-error factor
/// only as a source/session anchor.  Its `close_price` field is validated by
/// the Python source loader but is not part of the formula itself, so the
/// typed prototype retains only the two fields it consumes.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyPriceAnchorRow {
    pub trade_date: NaiveDate,
    pub code: String,
}

/// Strict-prior `market_cbond.daily_base` fields for the tracking-error
/// family.  These are historical bond and mapped-stock daily closes, not
/// current intraday prices.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyTrackingBaseRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub cb_prev_close_price: f64,
    pub cb_close_price: f64,
    pub stk_prev_close_price: f64,
    pub stk_close_price: f64,
}

/// Two-source context for `bstk_tail_cocrash_residual20`.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyTrackingContext {
    pub score_date: NaiveDate,
    pub price_anchor_by_code: BTreeMap<String, Vec<TypedFactorDailyPriceAnchorRow>>,
    pub base_by_code: BTreeMap<String, Vec<TypedFactorDailyTrackingBaseRow>>,
}

impl TypedFactorDailyTrackingContext {
    pub fn new(score_date: NaiveDate) -> Self {
        Self {
            score_date,
            price_anchor_by_code: BTreeMap::new(),
            base_by_code: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypedFactorDailyPathsError {
    DuplicateStrictPriorDate {
        source: &'static str,
        code: String,
        trade_date: NaiveDate,
    },
}

impl fmt::Display for TypedFactorDailyPathsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStrictPriorDate {
                source,
                code,
                trade_date,
            } => write!(
                formatter,
                "typed_factor daily-path source {source} has duplicate strict-prior row for {code} on {trade_date}"
            ),
        }
    }
}

impl Error for TypedFactorDailyPathsError {}

fn strict_rows<'a, T, F>(
    rows_by_code: &'a BTreeMap<String, Vec<T>>,
    score_date: NaiveDate,
    code: &str,
    source: &'static str,
    date_of: F,
) -> Result<Vec<&'a T>, TypedFactorDailyPathsError>
where
    F: Fn(&T) -> NaiveDate,
{
    let mut rows: Vec<&T> = rows_by_code
        .get(code)
        .into_iter()
        .flatten()
        .filter(|row| date_of(row) < score_date)
        .collect();
    rows.sort_by_key(|row| date_of(row));
    for pair in rows.windows(2) {
        let trade_date = date_of(pair[0]);
        if trade_date == date_of(pair[1]) {
            return Err(TypedFactorDailyPathsError::DuplicateStrictPriorDate {
                source,
                code: code.to_string(),
                trade_date,
            });
        }
    }
    Ok(rows)
}

fn strict_path_histories<'a>(
    ctx: &'a TypedFactorDailyPathContext,
) -> Result<BTreeMap<String, Vec<&'a TypedFactorDailyPathPriceRow>>, TypedFactorDailyPathsError> {
    let mut histories = BTreeMap::new();
    for code in ctx.price_by_code.keys() {
        histories.insert(
            code.clone(),
            strict_rows(
                &ctx.price_by_code,
                ctx.score_date,
                code,
                "market_cbond.daily_price",
                |row: &TypedFactorDailyPathPriceRow| row.trade_date,
            )?,
        );
    }
    Ok(histories)
}

fn global_calendar<T, F>(histories: &BTreeMap<String, Vec<&T>>, date_of: F) -> Vec<NaiveDate>
where
    F: Fn(&T) -> NaiveDate,
{
    let mut dates = BTreeSet::new();
    for rows in histories.values() {
        for row in rows {
            dates.insert(date_of(row));
        }
    }
    dates.into_iter().collect()
}

fn finite_or_nan(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}

fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() <= DAILY_PATHS_EPS {
        return f64::NAN;
    }
    finite_or_nan(numerator / denominator)
}

#[derive(Clone, Copy, Debug)]
struct OhlcMetrics {
    wick_asymmetry: f64,
    intraday_return: f64,
    intraday_range: f64,
}

fn ohlc_metrics(row: &TypedFactorDailyPathPriceRow) -> OhlcMetrics {
    // Deliberately use comparison semantics rather than an up-front
    // `is_finite` guard.  Pandas' `gt/ge/le` lets +Inf through a positive
    // comparison and its downstream NumPy formula then determines whether
    // the final metric is finite.  In particular, an infinite prior close is
    // not used in the formula but still satisfies the Python validity gate.
    let width = row.high_price - row.low_price;
    let valid = row.open_price > DAILY_PATHS_EPS
        && row.high_price > DAILY_PATHS_EPS
        && row.low_price > DAILY_PATHS_EPS
        && row.close_price > DAILY_PATHS_EPS
        && row.prev_close_price > DAILY_PATHS_EPS
        && width > DAILY_PATHS_EPS
        && row.high_price >= row.open_price
        && row.high_price >= row.close_price
        && row.low_price <= row.open_price
        && row.low_price <= row.close_price;
    if !valid {
        return OhlcMetrics {
            wick_asymmetry: f64::NAN,
            intraday_return: f64::NAN,
            intraday_range: f64::NAN,
        };
    }
    let top = row.open_price.max(row.close_price);
    let bottom = row.open_price.min(row.close_price);
    OhlcMetrics {
        wick_asymmetry: (row.high_price - top - (bottom - row.low_price)) / width,
        intraday_return: (row.close_price / row.open_price).ln(),
        intraday_range: (row.high_price / row.low_price).ln(),
    }
}

fn ohlc_recent<'a>(
    histories: &'a BTreeMap<String, Vec<&'a TypedFactorDailyPathPriceRow>>,
    code: &str,
) -> Vec<&'a TypedFactorDailyPathPriceRow> {
    let calendar = global_calendar(histories, |row: &TypedFactorDailyPathPriceRow| {
        row.trade_date
    });
    let Some(anchor) = calendar.last().copied() else {
        return Vec::new();
    };
    let Some(rows) = histories.get(code) else {
        return Vec::new();
    };
    if rows.last().map(|row| row.trade_date) != Some(anchor) {
        return Vec::new();
    }
    let positions: BTreeMap<NaiveDate, usize> = calendar
        .iter()
        .copied()
        .enumerate()
        .map(|(position, date)| (date, position))
        .collect();
    let anchor_position = positions[&anchor];
    let first_position = anchor_position
        .saturating_add(1)
        .saturating_sub(DOHW_WINDOW);
    rows.iter()
        .copied()
        .filter(|row| positions[&row.trade_date] >= first_position)
        .collect()
}

/// `dohw_mean_wick_asymmetry60`: latest up-to-60 global strict-prior
/// sessions' finite wick-asymmetry mean, requiring a finite terminal wick and
/// at least 45 finite values.
pub fn dohw_mean_wick_asymmetry60(
    ctx: &TypedFactorDailyPathContext,
    code: &str,
) -> Result<f64, TypedFactorDailyPathsError> {
    let histories = strict_path_histories(ctx)?;
    let recent = ohlc_recent(&histories, code);
    let Some(terminal) = recent.last() else {
        return Ok(f64::NAN);
    };
    if !ohlc_metrics(terminal).wick_asymmetry.is_finite() {
        return Ok(f64::NAN);
    }
    let values: Vec<f64> = recent
        .iter()
        .map(|row| ohlc_metrics(row).wick_asymmetry)
        .filter(|value| value.is_finite())
        .collect();
    if values.len() < DOHW_MIN_OBSERVATIONS {
        return Ok(f64::NAN);
    }
    Ok(finite_or_nan(
        values.iter().sum::<f64>() / values.len() as f64,
    ))
}

/// `dohw_intraday_sign_range_asymmetry60`: latest up-to-60 global
/// strict-prior sessions' mean `log(high / low)` on positive completed
/// intraday log-return days less its mean on negative days.  It requires the
/// two terminal formulas, 45 finite paired observations, and eight points in
/// each direction.
pub fn dohw_intraday_sign_range_asymmetry60(
    ctx: &TypedFactorDailyPathContext,
    code: &str,
) -> Result<f64, TypedFactorDailyPathsError> {
    let histories = strict_path_histories(ctx)?;
    let recent = ohlc_recent(&histories, code);
    let Some(terminal) = recent.last() else {
        return Ok(f64::NAN);
    };
    let terminal_metrics = ohlc_metrics(terminal);
    if !terminal_metrics.intraday_return.is_finite() || !terminal_metrics.intraday_range.is_finite()
    {
        return Ok(f64::NAN);
    }

    let metrics: Vec<OhlcMetrics> = recent.iter().map(|row| ohlc_metrics(row)).collect();
    let paired: Vec<OhlcMetrics> = metrics
        .into_iter()
        .filter(|metric| metric.intraday_return.is_finite() && metric.intraday_range.is_finite())
        .collect();
    if paired.len() < DOHW_MIN_OBSERVATIONS {
        return Ok(f64::NAN);
    }
    let positive: Vec<f64> = paired
        .iter()
        .filter(|metric| metric.intraday_return > 0.0)
        .map(|metric| metric.intraday_range)
        .collect();
    let negative: Vec<f64> = paired
        .iter()
        .filter(|metric| metric.intraday_return < 0.0)
        .map(|metric| metric.intraday_range)
        .collect();
    if positive.len() < DOHW_MIN_DIRECTIONAL_OBSERVATIONS
        || negative.len() < DOHW_MIN_DIRECTIONAL_OBSERVATIONS
    {
        return Ok(f64::NAN);
    }
    let positive_mean = positive.iter().sum::<f64>() / positive.len() as f64;
    let negative_mean = negative.iter().sum::<f64>() / negative.len() as f64;
    Ok(finite_or_nan(positive_mean - negative_mean))
}

fn catalogue_daily_return(row: &TypedFactorDailyPathPriceRow) -> f64 {
    let previous = if row.act_prev_close_price > 0.0 {
        row.act_prev_close_price
    } else {
        row.prev_close_price
    };
    if previous > 0.0 && row.close_price > 0.0 {
        row.close_price / previous - 1.0
    } else {
        f64::NAN
    }
}

fn drawup_drawdown_asymmetry(returns: &[f64]) -> f64 {
    if returns.len() < 5 || returns.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    let mut wealth = 1.0_f64;
    // NumPy's `maximum.accumulate(wealth)` / `minimum.accumulate(wealth)`
    // begin at the *first completed return*, not at an artificial initial
    // wealth of one.  That distinction matters when the first return is a
    // loss or gain.
    let mut running_high = f64::NAN;
    let mut running_low = f64::NAN;
    let mut drawdown = f64::INFINITY;
    let mut drawup = f64::NEG_INFINITY;
    for (index, return_value) in returns.iter().enumerate() {
        wealth *= 1.0 + *return_value;
        // A finite daily return can still overflow/underflow a long
        // cumulative product.  NumPy propagates the subsequent undefined
        // high/low ratio; fail closed rather than silently repairing it.
        if !wealth.is_finite() || wealth <= 0.0 {
            return f64::NAN;
        }
        if index == 0 {
            running_high = wealth;
            running_low = wealth;
        } else {
            running_high = running_high.max(wealth);
            running_low = running_low.min(wealth);
        }
        drawdown = drawdown.min(wealth / running_high - 1.0);
        drawup = drawup.max(wealth / running_low - 1.0);
    }
    safe_div(drawup, drawdown.abs())
}

/// `dret_drawup_drawdown_asym`: use the last 20 finite historical completed
/// daily returns (at least 15), then divide cumulative draw-up by the absolute
/// cumulative drawdown.  It exactly keeps the Python catalogue's finite-only
/// chronology: an invalid latest source row is discarded rather than making
/// the signal stale or terminally unavailable.
pub fn dret_drawup_drawdown_asym(
    ctx: &TypedFactorDailyPathContext,
    code: &str,
) -> Result<f64, TypedFactorDailyPathsError> {
    let histories = strict_path_histories(ctx)?;
    let Some(rows) = histories.get(code) else {
        return Ok(f64::NAN);
    };
    let finite_returns: Vec<f64> = rows
        .iter()
        .map(|row| catalogue_daily_return(row))
        .filter(|value| value.is_finite())
        .collect();
    if finite_returns.len() < DRET_MIN_OBSERVATIONS {
        return Ok(f64::NAN);
    }
    let first = finite_returns.len().saturating_sub(DRET_WINDOW);
    Ok(finite_or_nan(drawup_drawdown_asymmetry(
        &finite_returns[first..],
    )))
}

fn strict_anchor_histories<'a>(
    ctx: &'a TypedFactorDailyTrackingContext,
) -> Result<BTreeMap<String, Vec<&'a TypedFactorDailyPriceAnchorRow>>, TypedFactorDailyPathsError> {
    let mut histories = BTreeMap::new();
    for code in ctx.price_anchor_by_code.keys() {
        histories.insert(
            code.clone(),
            strict_rows(
                &ctx.price_anchor_by_code,
                ctx.score_date,
                code,
                "market_cbond.daily_price",
                |row: &TypedFactorDailyPriceAnchorRow| row.trade_date,
            )?,
        );
    }
    Ok(histories)
}

fn strict_tracking_histories<'a>(
    ctx: &'a TypedFactorDailyTrackingContext,
) -> Result<BTreeMap<String, Vec<&'a TypedFactorDailyTrackingBaseRow>>, TypedFactorDailyPathsError>
{
    let mut histories = BTreeMap::new();
    for code in ctx.base_by_code.keys() {
        histories.insert(
            code.clone(),
            strict_rows(
                &ctx.base_by_code,
                ctx.score_date,
                code,
                "market_cbond.daily_base",
                |row: &TypedFactorDailyTrackingBaseRow| row.trade_date,
            )?,
        );
    }
    Ok(histories)
}

fn safe_beta(stock_returns: &[f64], bond_returns: &[f64]) -> f64 {
    if stock_returns.len() < 5
        || stock_returns.len() != bond_returns.len()
        || stock_returns.iter().any(|value| !value.is_finite())
        || bond_returns.iter().any(|value| !value.is_finite())
    {
        return f64::NAN;
    }
    let stock_mean = stock_returns.iter().sum::<f64>() / stock_returns.len() as f64;
    let bond_mean = bond_returns.iter().sum::<f64>() / bond_returns.len() as f64;
    let mut denominator = 0.0;
    let mut numerator = 0.0;
    for (&stock, &bond) in stock_returns.iter().zip(bond_returns.iter()) {
        let centered_stock = stock - stock_mean;
        denominator += centered_stock * centered_stock;
        numerator += centered_stock * (bond - bond_mean);
    }
    if !denominator.is_finite() || denominator <= DAILY_PATHS_EPS {
        return f64::NAN;
    }
    finite_or_nan(numerator / denominator)
}

/// NumPy's `quantile(..., 0.2)` default `method="linear"` for a complete
/// finite slice.  The tracking factor already establishes finiteness before
/// calling it, but this helper remains fail-closed for direct use.
fn linear_quantile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty()
        || !quantile.is_finite()
        || !(0.0..=1.0).contains(&quantile)
        || values.iter().any(|value| !value.is_finite())
    {
        return f64::NAN;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if ordered.len() == 1 {
        return ordered[0];
    }
    let position = quantile * (ordered.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return ordered[lower];
    }
    let fraction = position - lower as f64;
    finite_or_nan(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)
}

/// `bstk_tail_cocrash_residual20`: residual bond return on the stock's lower
/// 20% tail days after fitting the historical 20-day stock-beta.  The price
/// source establishes the shared market calendar; base history is date-key
/// inner joined to it and cannot bridge a missing session.
pub fn bstk_tail_cocrash_residual20(
    ctx: &TypedFactorDailyTrackingContext,
    code: &str,
) -> Result<f64, TypedFactorDailyPathsError> {
    let price_histories = strict_anchor_histories(ctx)?;
    let base_histories = strict_tracking_histories(ctx)?;
    let calendar = global_calendar(&price_histories, |row: &TypedFactorDailyPriceAnchorRow| {
        row.trade_date
    });
    let Some(anchor) = calendar.last().copied() else {
        return Ok(f64::NAN);
    };
    let Some(price_rows) = price_histories.get(code) else {
        return Ok(f64::NAN);
    };
    if price_rows.last().map(|row| row.trade_date) != Some(anchor) {
        return Ok(f64::NAN);
    }
    let Some(base_rows) = base_histories.get(code) else {
        return Ok(f64::NAN);
    };
    let price_dates: BTreeSet<NaiveDate> = price_rows.iter().map(|row| row.trade_date).collect();
    let joined: Vec<&TypedFactorDailyTrackingBaseRow> = base_rows
        .iter()
        .copied()
        .filter(|row| price_dates.contains(&row.trade_date))
        .collect();
    if joined.last().map(|row| row.trade_date) != Some(anchor)
        || joined.len() < BSTK_COMPLETE_WINDOW
    {
        return Ok(f64::NAN);
    }
    let calendar_positions: BTreeMap<NaiveDate, usize> = calendar
        .iter()
        .copied()
        .enumerate()
        .map(|(position, date)| (date, position))
        .collect();
    let tail_start = joined.len() - BSTK_COMPLETE_WINDOW;
    let tail = &joined[tail_start..];
    let positions: Vec<usize> = tail
        .iter()
        .map(|row| calendar_positions[&row.trade_date])
        .collect();
    let expected_start = positions[BSTK_COMPLETE_WINDOW - 1] + 1 - BSTK_COMPLETE_WINDOW;
    if positions
        .iter()
        .enumerate()
        .any(|(offset, position)| *position != expected_start + offset)
    {
        return Ok(f64::NAN);
    }
    if tail.iter().any(|row| {
        !row.cb_prev_close_price.is_finite()
            || !row.cb_close_price.is_finite()
            || !row.stk_prev_close_price.is_finite()
            || !row.stk_close_price.is_finite()
            || row.cb_prev_close_price <= DAILY_PATHS_EPS
            || row.cb_close_price <= DAILY_PATHS_EPS
            || row.stk_prev_close_price <= DAILY_PATHS_EPS
            || row.stk_close_price <= DAILY_PATHS_EPS
    }) {
        return Ok(f64::NAN);
    }

    let bond_returns: Vec<f64> = tail
        .iter()
        .map(|row| row.cb_close_price / row.cb_prev_close_price - 1.0)
        .collect();
    let stock_returns: Vec<f64> = tail
        .iter()
        .map(|row| row.stk_close_price / row.stk_prev_close_price - 1.0)
        .collect();
    let beta20 = safe_beta(&stock_returns[20..40], &bond_returns[20..40]);
    if !beta20.is_finite() {
        return Ok(f64::NAN);
    }
    let residual: Vec<f64> = bond_returns[21..]
        .iter()
        .zip(stock_returns[21..].iter())
        .map(|(&bond, &stock)| bond - beta20 * stock)
        .collect();
    let tail_stock = &stock_returns[21..];
    let threshold = linear_quantile(tail_stock, 0.2);
    if !threshold.is_finite() {
        return Ok(f64::NAN);
    }
    let selected: Vec<f64> = residual
        .iter()
        .zip(tail_stock.iter())
        .filter_map(|(&value, &stock)| (stock <= threshold).then_some(value))
        .collect();
    if selected.len() < 4 {
        return Ok(f64::NAN);
    }
    Ok(finite_or_nan(
        selected.iter().sum::<f64>() / selected.len() as f64,
    ))
}
