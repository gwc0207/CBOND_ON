//! Strict-prior daily kernels for typed factor contracts.
//!
//! The ordinary `compute_factor_frame` dispatcher reaches this module through
//! `typed_factor_kernels`; it is not a separate runtime route.  The context
//! expects already-normalised market codes and enforces the strict-prior
//! (`trade_date < score_date`) cut itself.

use chrono::NaiveDate;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub const TYPED_FACTOR_EPS: f64 = 1e-12;

/// Strict-prior daily-base fields needed by the P1 live-23 kernels.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyBaseRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub debt_puredebt_ratio: f64,
    pub puredebt_prem_ratio: f64,
    pub bond_prem_ratio: f64,
    pub redemption_prem_ratio: f64,
}

impl TypedFactorDailyBaseRow {
    pub fn empty(trade_date: NaiveDate, code: impl Into<String>) -> Self {
        Self {
            trade_date,
            code: code.into(),
            debt_puredebt_ratio: f64::NAN,
            puredebt_prem_ratio: f64::NAN,
            bond_prem_ratio: f64::NAN,
            redemption_prem_ratio: f64::NAN,
        }
    }
}

/// Strict-prior daily-price fields needed by the P1 live-23 kernels.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyPriceRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub prev_close_price: f64,
    pub close_price: f64,
    pub high_price: f64,
    pub low_price: f64,
    pub volume: f64,
    pub amount: f64,
    pub deal: f64,
}

impl TypedFactorDailyPriceRow {
    pub fn empty(trade_date: NaiveDate, code: impl Into<String>) -> Self {
        Self {
            trade_date,
            code: code.into(),
            prev_close_price: f64::NAN,
            close_price: f64::NAN,
            high_price: f64::NAN,
            low_price: f64::NAN,
            volume: f64::NAN,
            amount: f64::NAN,
            deal: f64::NAN,
        }
    }
}

/// Strict-prior daily-TWAP fields needed by the P1 live-23 kernels.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyTwapRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub twap_0930_0935: f64,
    pub twap_0935_1000: f64,
}

impl TypedFactorDailyTwapRow {
    pub fn empty(trade_date: NaiveDate, code: impl Into<String>) -> Self {
        Self {
            trade_date,
            code: code.into(),
            twap_0930_0935: f64::NAN,
            twap_0935_1000: f64::NAN,
        }
    }
}

/// Daily source material for one factor score day.
///
/// The eventual `lib.rs` adapter should populate these maps from the existing
/// `AuxData.daily_sources` parser after applying the same code normalisation
/// as the Python research kernels.  Rows may include the score day; every
/// kernel below excludes it explicitly.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyContext {
    pub score_date: NaiveDate,
    pub base_by_code: BTreeMap<String, Vec<TypedFactorDailyBaseRow>>,
    pub price_by_code: BTreeMap<String, Vec<TypedFactorDailyPriceRow>>,
    pub twap_by_code: BTreeMap<String, Vec<TypedFactorDailyTwapRow>>,
}

impl TypedFactorDailyContext {
    pub fn new(score_date: NaiveDate) -> Self {
        Self {
            score_date,
            base_by_code: BTreeMap::new(),
            price_by_code: BTreeMap::new(),
            twap_by_code: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypedFactorDailyError {
    DuplicateStrictPriorDate {
        source: &'static str,
        code: String,
        trade_date: NaiveDate,
    },
    UnknownP1Signal(String),
}

impl fmt::Display for TypedFactorDailyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStrictPriorDate {
                source,
                code,
                trade_date,
            } => write!(
                f,
                "typed_factor daily source {source} has duplicate strict-prior row for {code} on {trade_date}"
            ),
            Self::UnknownP1Signal(signal) => write!(f, "unknown typed_factor P1 daily signal: {signal}"),
        }
    }
}

impl Error for TypedFactorDailyError {}

fn strict_rows<'a, T, F>(
    rows_by_code: &'a BTreeMap<String, Vec<T>>,
    score_date: NaiveDate,
    code: &str,
    source: &'static str,
    date_of: F,
) -> Result<Vec<&'a T>, TypedFactorDailyError>
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
            return Err(TypedFactorDailyError::DuplicateStrictPriorDate {
                source,
                code: code.to_string(),
                trade_date,
            });
        }
    }
    Ok(rows)
}

fn base_history<'a>(
    ctx: &'a TypedFactorDailyContext,
    code: &str,
) -> Result<Vec<&'a TypedFactorDailyBaseRow>, TypedFactorDailyError> {
    strict_rows(
        &ctx.base_by_code,
        ctx.score_date,
        code,
        "market_cbond.daily_base",
        |row| row.trade_date,
    )
}

fn price_history<'a>(
    ctx: &'a TypedFactorDailyContext,
    code: &str,
) -> Result<Vec<&'a TypedFactorDailyPriceRow>, TypedFactorDailyError> {
    strict_rows(
        &ctx.price_by_code,
        ctx.score_date,
        code,
        "market_cbond.daily_price",
        |row| row.trade_date,
    )
}

fn twap_history<'a>(
    ctx: &'a TypedFactorDailyContext,
    code: &str,
) -> Result<Vec<&'a TypedFactorDailyTwapRow>, TypedFactorDailyError> {
    strict_rows(
        &ctx.twap_by_code,
        ctx.score_date,
        code,
        "market_cbond.daily_twap",
        |row| row.trade_date,
    )
}

fn finite_or_nan(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}

fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() <= TYPED_FACTOR_EPS {
        return f64::NAN;
    }
    finite_or_nan(numerator / denominator)
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    let value = safe_div(numerator, denominator);
    if value.is_finite() {
        finite_or_nan(value - 1.0)
    } else {
        f64::NAN
    }
}

fn last_not_nan<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    values
        .into_iter()
        .filter(|value| !value.is_nan())
        .last()
        .unwrap_or(f64::NAN)
}

fn last_finite(values: &[f64]) -> f64 {
    values
        .last()
        .copied()
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NAN)
}

fn finite_tail(values: impl IntoIterator<Item = f64>, count: usize, min_count: usize) -> Vec<f64> {
    let finite: Vec<f64> = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < min_count {
        Vec::new()
    } else {
        let first = finite.len().saturating_sub(count);
        finite[first..].to_vec()
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    finite_or_nan(values.iter().sum::<f64>() / values.len() as f64)
}

/// NumPy/Pandas-equivalent sample standard deviation (`ddof=1`) for finite tails.
fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let avg = mean(values);
    if !avg.is_finite() {
        return f64::NAN;
    }
    let squared_error = values
        .iter()
        .map(|value| {
            let centered = *value - avg;
            centered * centered
        })
        .sum::<f64>();
    finite_or_nan((squared_error / (values.len() - 1) as f64).sqrt())
}

fn trailing_sample_std(
    values: impl IntoIterator<Item = f64>,
    count: usize,
    min_count: usize,
) -> f64 {
    let tail = finite_tail(values, count, min_count);
    sample_std(&tail)
}

fn trailing_mean(values: impl IntoIterator<Item = f64>, count: usize, min_count: usize) -> f64 {
    let tail = finite_tail(values, count, min_count);
    mean(&tail)
}

fn trailing_z_last(values: &[f64], count: usize, min_count: usize) -> f64 {
    let latest = last_finite(values);
    let tail = finite_tail(values.iter().copied(), count, min_count);
    if !latest.is_finite() || tail.len() < 2 {
        return f64::NAN;
    }
    safe_div(latest - mean(&tail), sample_std(&tail))
}

fn trailing_pearson(left: &[f64], right: &[f64], count: usize, min_count: usize) -> f64 {
    let mut pairs: Vec<(f64, f64)> = left
        .iter()
        .copied()
        .zip(right.iter().copied())
        // `DataFrame.dropna()` retains +/-Inf; NumPy then produces a non-finite
        // correlation.  Preserve that result instead of silently filtering it.
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .collect();
    let first = pairs.len().saturating_sub(count);
    pairs = pairs.split_off(first);
    if pairs.len() < min_count || pairs.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return f64::NAN;
    }
    let left_values: Vec<f64> = pairs.iter().map(|(a, _)| *a).collect();
    let right_values: Vec<f64> = pairs.iter().map(|(_, b)| *b).collect();
    let left_mean = mean(&left_values);
    let right_mean = mean(&right_values);
    let left_ss = left_values
        .iter()
        .map(|value| {
            let centered = *value - left_mean;
            centered * centered
        })
        .sum::<f64>();
    let right_ss = right_values
        .iter()
        .map(|value| {
            let centered = *value - right_mean;
            centered * centered
        })
        .sum::<f64>();
    let population_left_std = (left_ss / left_values.len() as f64).sqrt();
    let population_right_std = (right_ss / right_values.len() as f64).sqrt();
    if !population_left_std.is_finite()
        || !population_right_std.is_finite()
        || population_left_std <= TYPED_FACTOR_EPS
        || population_right_std <= TYPED_FACTOR_EPS
    {
        return f64::NAN;
    }
    let covariance_numerator = left_values
        .iter()
        .zip(right_values.iter())
        .map(|(a, b)| (*a - left_mean) * (*b - right_mean))
        .sum::<f64>();
    finite_or_nan(covariance_numerator / (left_ss * right_ss).sqrt())
}

/// `base_debt_premium_floor_gap`: latest non-null debt pure-debt ratio less
/// latest non-null pure-debt premium ratio, over strict-prior daily base data.
pub fn base_debt_premium_floor_gap(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyError> {
    let rows = base_history(ctx, code)?;
    let debt = last_not_nan(rows.iter().map(|row| row.debt_puredebt_ratio));
    let premium = last_not_nan(rows.iter().map(|row| row.puredebt_prem_ratio));
    Ok(finite_or_nan(debt - premium))
}

/// `dredemption_bondpremium_interaction`: terminal strict-prior bond premium
/// times terminal strict-prior redemption premium.
pub fn dredemption_bondpremium_interaction(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyError> {
    let rows = base_history(ctx, code)?;
    let bond_premium: Vec<f64> = rows.iter().map(|row| row.bond_prem_ratio).collect();
    let redemption_premium: Vec<f64> = rows.iter().map(|row| row.redemption_prem_ratio).collect();
    Ok(finite_or_nan(
        last_finite(&bond_premium) * last_finite(&redemption_premium),
    ))
}

/// `dredemption_premium_z20`: terminal redemption premium z-score using the
/// last 20 finite strict-prior values and sample (`ddof=1`) volatility.
pub fn dredemption_premium_z20(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyError> {
    let rows = base_history(ctx, code)?;
    let values: Vec<f64> = rows.iter().map(|row| row.redemption_prem_ratio).collect();
    Ok(trailing_z_last(&values, 20, 12))
}

/// `dret_volatility_20`: sample standard deviation of last 20 finite
/// `close_price / prev_close_price - 1` values (at least 12 observations).
pub fn dret_volatility_20(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyError> {
    let rows = price_history(ctx, code)?;
    Ok(trailing_sample_std(
        rows.iter()
            .map(|row| ratio(row.close_price, row.prev_close_price)),
        20,
        12,
    ))
}

/// `dliq_volume_return_corr20`: Pearson correlation of `log(volume)` and
/// completed daily return across the latest 20 complete pairs.
pub fn dliq_volume_return_corr20(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyError> {
    let rows = price_history(ctx, code)?;
    let log_volume: Vec<f64> = rows
        .iter()
        .map(|row| {
            if row.volume.is_finite() && row.volume > 0.0 {
                row.volume.ln()
            } else {
                f64::NAN
            }
        })
        .collect();
    let returns: Vec<f64> = rows
        .iter()
        .map(|row| ratio(row.close_price, row.prev_close_price))
        .collect();
    Ok(trailing_pearson(&log_volume, &returns, 20, 12))
}

/// `dtwap_morning_slope20`: mean of the last 20 finite completed-session
/// `twap_0935_1000 / twap_0930_0935 - 1` values (at least 12 observations).
pub fn dtwap_morning_slope20(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyError> {
    let rows = twap_history(ctx, code)?;
    Ok(trailing_mean(
        rows.iter()
            .map(|row| ratio(row.twap_0935_1000, row.twap_0930_0935)),
        20,
        12,
    ))
}

fn global_strict_price_anchor(
    ctx: &TypedFactorDailyContext,
) -> Result<Option<NaiveDate>, TypedFactorDailyError> {
    let mut anchor: Option<NaiveDate> = None;
    for code in ctx.price_by_code.keys() {
        let rows = price_history(ctx, code)?;
        if let Some(last) = rows.last() {
            anchor = Some(anchor.map_or(last.trade_date, |current| current.max(last.trade_date)));
        }
    }
    Ok(anchor)
}

fn anchored_price_history<'a>(
    ctx: &'a TypedFactorDailyContext,
    code: &str,
) -> Result<Vec<&'a TypedFactorDailyPriceRow>, TypedFactorDailyError> {
    let rows = price_history(ctx, code)?;
    let anchor = global_strict_price_anchor(ctx)?;
    if rows.last().map(|row| row.trade_date) != anchor {
        return Ok(Vec::new());
    }
    Ok(rows)
}

/// `drt_rebound_from_low20`: use the latest 20 complete rows only if this
/// bond reaches the global strict-prior daily-price anchor.  This preserves the
/// source kernel's stale-history fail-closed rule.
pub fn drt_rebound_from_low20(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyError> {
    let rows = anchored_price_history(ctx, code)?;
    // `_complete_tail` in the Python reference has a deliberately stricter
    // terminal-observation contract than its subsequent `dropna()` call: if
    // the latest raw strict-prior observation is incomplete, the whole signal
    // is unavailable.  Do this check before filtering older incomplete rows;
    // otherwise a stale tail could silently produce a value.
    if let Some(latest) = rows.last() {
        if latest.close_price.is_nan()
            || latest.high_price.is_nan()
            || latest.low_price.is_nan()
            || latest.amount.is_nan()
        {
            return Ok(f64::NAN);
        }
    }
    let complete: Vec<&TypedFactorDailyPriceRow> = rows
        .into_iter()
        // Python `dropna()` keeps infinities.  We do the same; `safe_div` and
        // final sanitisation turn a non-finite final result into NaN.
        .filter(|row| {
            !row.close_price.is_nan()
                && !row.high_price.is_nan()
                && !row.low_price.is_nan()
                && !row.amount.is_nan()
        })
        .collect();
    if complete.len() < 12 {
        return Ok(f64::NAN);
    }
    let first = complete.len().saturating_sub(20);
    let tail = &complete[first..];
    let mut trough_index = 0usize;
    for index in 1..tail.len() {
        if tail[index].low_price < tail[trough_index].low_price {
            trough_index = index;
        }
    }
    Ok(safe_div(
        tail[tail.len() - 1].close_price - tail[trough_index].low_price,
        tail[trough_index].low_price,
    ))
}

/// Dispatch only the first P1 subset.  The factor-string/`params.signal`
/// routing remains outside this inactive module until the full 23-factor
/// parity suite approves integration.
pub fn compute_p1_daily_signal(
    ctx: &TypedFactorDailyContext,
    code: &str,
    signal: &str,
) -> Result<f64, TypedFactorDailyError> {
    match signal {
        "base_debt_premium_floor_gap" => base_debt_premium_floor_gap(ctx, code),
        "dredemption_bondpremium_interaction" => dredemption_bondpremium_interaction(ctx, code),
        "dredemption_premium_z20" => dredemption_premium_z20(ctx, code),
        "drt_rebound_from_low20" => drt_rebound_from_low20(ctx, code),
        "dret_volatility_20" => dret_volatility_20(ctx, code),
        "dliq_volume_return_corr20" => dliq_volume_return_corr20(ctx, code),
        "dtwap_morning_slope20" => dtwap_morning_slope20(ctx, code),
        other => Err(TypedFactorDailyError::UnknownP1Signal(other.to_string())),
    }
}
