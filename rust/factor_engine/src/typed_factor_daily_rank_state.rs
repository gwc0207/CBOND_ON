//! Strict-prior daily rank/state kernels for typed factor contracts.
//!
//! The ordinary Rust dispatcher reaches this pure-Rust implementation through
//! the typed-kernel boundary. The caller supplies exchange-normalised codes
//! and may include score-day rows: every function excludes those rows before
//! validating the complete strict-prior universe.

use chrono::NaiveDate;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const EPS: f64 = 1e-12;
const WINDOW: usize = 60;
const MIN_JOINT_OBSERVATIONS: usize = 45;
const MIN_DIRECTIONAL_OBSERVATIONS: usize = 8;

/// Fields from `market_cbond.daily_price` required by the staged signals.
#[derive(Clone, Debug)]
pub struct TypedFactorRankStatePriceRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub prev_close_price: f64,
    pub act_prev_close_price: f64,
    pub close_price: f64,
    pub amount: f64,
}

/// Fields from `market_cbond.daily_base` required by the staged signals.
#[derive(Clone, Debug)]
pub struct TypedFactorRankStateBaseRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub remain_size: f64,
    pub current_yield: f64,
}

/// Complete strict-prior source material for one score day.
#[derive(Clone, Debug)]
pub struct TypedFactorRankStateContext {
    pub score_date: NaiveDate,
    pub price_rows: Vec<TypedFactorRankStatePriceRow>,
    pub base_rows: Vec<TypedFactorRankStateBaseRow>,
}

impl TypedFactorRankStateContext {
    pub fn new(score_date: NaiveDate) -> Self {
        Self {
            score_date,
            price_rows: Vec::new(),
            base_rows: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypedFactorRankStateError {
    DuplicateStrictPriorDate {
        source: &'static str,
        code: String,
        trade_date: NaiveDate,
    },
}

impl fmt::Display for TypedFactorRankStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStrictPriorDate {
                source,
                code,
                trade_date,
            } => write!(
                f,
                "typed_factor rank/state source {source} has duplicate strict-prior row for {code} on {trade_date}"
            ),
        }
    }
}

impl Error for TypedFactorRankStateError {}

#[derive(Clone, Debug)]
struct JoinedRow {
    trade_date: NaiveDate,
    code: String,
    prev_close_price: f64,
    act_prev_close_price: f64,
    close_price: f64,
    amount: f64,
    remain_size: f64,
    current_yield: f64,
}

fn strict_price_rows(
    ctx: &TypedFactorRankStateContext,
) -> Result<Vec<&TypedFactorRankStatePriceRow>, TypedFactorRankStateError> {
    let mut rows: Vec<_> = ctx
        .price_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .collect();
    rows.sort_by(|left, right| {
        (left.trade_date, left.code.as_str()).cmp(&(right.trade_date, right.code.as_str()))
    });
    for pair in rows.windows(2) {
        if pair[0].trade_date == pair[1].trade_date && pair[0].code == pair[1].code {
            return Err(TypedFactorRankStateError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_price",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

fn strict_base_rows(
    ctx: &TypedFactorRankStateContext,
) -> Result<Vec<&TypedFactorRankStateBaseRow>, TypedFactorRankStateError> {
    let mut rows: Vec<_> = ctx
        .base_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .collect();
    rows.sort_by(|left, right| {
        (left.trade_date, left.code.as_str()).cmp(&(right.trade_date, right.code.as_str()))
    });
    for pair in rows.windows(2) {
        if pair[0].trade_date == pair[1].trade_date && pair[0].code == pair[1].code {
            return Err(TypedFactorRankStateError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_base",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

/// Match the source kernels' two-source anchor and one-to-one inner join.
fn joined_history(
    ctx: &TypedFactorRankStateContext,
) -> Result<(Vec<JoinedRow>, Option<NaiveDate>), TypedFactorRankStateError> {
    let price = strict_price_rows(ctx)?;
    let base = strict_base_rows(ctx)?;
    let Some(price_anchor) = price.last().map(|row| row.trade_date) else {
        return Ok((Vec::new(), None));
    };
    let Some(base_anchor) = base.last().map(|row| row.trade_date) else {
        return Ok((Vec::new(), None));
    };
    if price_anchor != base_anchor {
        return Ok((Vec::new(), None));
    }

    let base_by_key: BTreeMap<(NaiveDate, &str), &TypedFactorRankStateBaseRow> = base
        .iter()
        .map(|row| ((row.trade_date, row.code.as_str()), *row))
        .collect();
    let mut joined = Vec::new();
    for price_row in price {
        let Some(base_row) = base_by_key.get(&(price_row.trade_date, price_row.code.as_str()))
        else {
            continue;
        };
        joined.push(JoinedRow {
            trade_date: price_row.trade_date,
            code: price_row.code.clone(),
            prev_close_price: price_row.prev_close_price,
            act_prev_close_price: price_row.act_prev_close_price,
            close_price: price_row.close_price,
            amount: price_row.amount,
            remain_size: base_row.remain_size,
            current_yield: base_row.current_yield,
        });
    }
    joined.sort_by(|left, right| {
        (left.trade_date, left.code.as_str()).cmp(&(right.trade_date, right.code.as_str()))
    });
    Ok((joined, Some(price_anchor)))
}

fn average_pct_rank(values: &[f64]) -> Vec<f64> {
    let mut result = vec![f64::NAN; values.len()];
    // pandas rank excludes NaN but ranks +/-Infinity, so only exclude NaN.
    let mut ordered: Vec<(f64, usize)> = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| (!value.is_nan()).then_some((value, index)))
        .collect();
    ordered.sort_by(|(left, _), (right, _)| left.partial_cmp(right).unwrap_or(Ordering::Equal));
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
        let rank = ((start + 1 + end) as f64) * 0.5 / denominator;
        for &(_, original_index) in &ordered[start..end] {
            result[original_index] = rank;
        }
        start = end;
    }
    result
}

fn log_ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator > EPS && denominator > EPS {
        let value = (numerator / denominator).ln();
        if value.is_finite() || value.is_infinite() {
            value
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    }
}

fn correlation(left: &[f64], right: &[f64], min_observations: usize) -> f64 {
    if left.len() != right.len() {
        return f64::NAN;
    }
    let pairs: Vec<(f64, f64)> = left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    if pairs.len() < min_observations {
        return f64::NAN;
    }
    let left_mean = pairs.iter().map(|(x, _)| x).sum::<f64>() / pairs.len() as f64;
    let right_mean = pairs.iter().map(|(_, y)| y).sum::<f64>() / pairs.len() as f64;
    let mut numerator = 0.0;
    let mut left_ss = 0.0;
    let mut right_ss = 0.0;
    for (x, y) in pairs {
        let left_centered = x - left_mean;
        let right_centered = y - right_mean;
        numerator += left_centered * right_centered;
        left_ss += left_centered * left_centered;
        right_ss += right_centered * right_centered;
    }
    let denominator = (left_ss * right_ss).sqrt();
    if !denominator.is_finite() || denominator <= EPS {
        return f64::NAN;
    }
    let value = numerator / denominator;
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}

/// `prcn_return_capacity_rank_corr60` with the Python source calendar rule.
///
/// Daily percentile ranks are formed after the price/base inner join.  The
/// trailing 60 sessions are measured in the complete joined-market calendar,
/// not in a compressed per-bond history.
pub fn prcn_return_capacity_rank_corr60(
    ctx: &TypedFactorRankStateContext,
    code: &str,
) -> Result<f64, TypedFactorRankStateError> {
    let (mut joined, anchor) = joined_history(ctx)?;
    let Some(anchor) = anchor else {
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

    let mut by_date: BTreeMap<NaiveDate, Vec<usize>> = BTreeMap::new();
    for (index, row) in joined.iter().enumerate() {
        by_date.entry(row.trade_date).or_default().push(index);
    }
    let mut return_ranks = vec![f64::NAN; joined.len()];
    let mut capacity_ranks = vec![f64::NAN; joined.len()];
    for indices in by_date.values() {
        let returns: Vec<f64> = indices
            .iter()
            .map(|index| log_ratio(joined[*index].close_price, joined[*index].prev_close_price))
            .collect();
        let capacities: Vec<f64> = indices
            .iter()
            .map(|index| log_ratio(joined[*index].amount, joined[*index].remain_size))
            .collect();
        for ((index, return_rank), capacity_rank) in indices
            .iter()
            .zip(average_pct_rank(&returns))
            .zip(average_pct_rank(&capacities))
        {
            return_ranks[*index] = return_rank;
            capacity_ranks[*index] = capacity_rank;
        }
    }

    // `joined` is date/code ordered, while the source groups code/date.
    // Rebuild only the requested path in date order before applying the
    // global-calendar cutoff used by the Python implementation.
    let mut path: Vec<(NaiveDate, f64, f64)> = joined
        .drain(..)
        .enumerate()
        .filter_map(|(index, row)| {
            (row.code == code).then_some((
                row.trade_date,
                return_ranks[index],
                capacity_ranks[index],
            ))
        })
        .collect();
    path.sort_by_key(|(date, _, _)| *date);
    if path.last().map(|(date, _, _)| *date) != Some(anchor) {
        return Ok(f64::NAN);
    }
    let first_position = anchor_position.saturating_sub(WINDOW - 1);
    let recent: Vec<_> = path
        .into_iter()
        .filter(|(date, _, _)| positions.get(date).copied().unwrap_or(usize::MAX) >= first_position)
        .collect();
    let Some((_, terminal_left, terminal_right)) = recent.last().copied() else {
        return Ok(f64::NAN);
    };
    if !terminal_left.is_finite() || !terminal_right.is_finite() {
        return Ok(f64::NAN);
    }
    let left: Vec<f64> = recent.iter().map(|(_, value, _)| *value).collect();
    let right: Vec<f64> = recent.iter().map(|(_, _, value)| *value).collect();
    Ok(correlation(&left, &right, MIN_JOINT_OBSERVATIONS))
}

fn slope(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < MIN_DIRECTIONAL_OBSERVATIONS {
        return f64::NAN;
    }
    if left.iter().any(|value| !value.is_finite()) || right.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let centered_left: Vec<f64> = left.iter().map(|value| *value - left_mean).collect();
    let denominator = centered_left.iter().map(|value| value * value).sum::<f64>();
    if !denominator.is_finite() || denominator <= EPS {
        return f64::NAN;
    }
    let numerator = centered_left
        .iter()
        .zip(right.iter())
        .map(|(x, y)| *x * (*y - right_mean))
        .sum::<f64>();
    let value = numerator / denominator;
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}

/// `ydpt_yield_fall_return_beta60`: OLS slope of completed adjusted-price log
/// return on negative current-yield changes across the terminal 60 changes.
pub fn ydpt_yield_fall_return_beta60(
    ctx: &TypedFactorRankStateContext,
    code: &str,
) -> Result<f64, TypedFactorRankStateError> {
    let (joined, anchor) = joined_history(ctx)?;
    let Some(anchor) = anchor else {
        return Ok(f64::NAN);
    };
    let mut path: Vec<_> = joined.into_iter().filter(|row| row.code == code).collect();
    path.sort_by_key(|row| row.trade_date);
    if path.last().map(|row| row.trade_date) != Some(anchor) || path.len() < WINDOW + 1 {
        return Ok(f64::NAN);
    }
    let first = path.len() - (WINDOW + 1);
    let tail = &path[first..];
    if tail.iter().any(|row| {
        !row.act_prev_close_price.is_finite()
            || !row.close_price.is_finite()
            || !row.current_yield.is_finite()
    }) {
        return Ok(f64::NAN);
    }
    let returns: Vec<f64> = tail
        .iter()
        .map(|row| log_ratio(row.close_price, row.act_prev_close_price))
        .collect();
    if returns.iter().any(|value| !value.is_finite()) {
        return Ok(f64::NAN);
    }
    let yield_changes: Vec<f64> = tail
        .windows(2)
        .map(|window| window[1].current_yield - window[0].current_yield)
        .collect();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (change, response) in yield_changes.into_iter().zip(returns.into_iter().skip(1)) {
        if change < -EPS {
            x.push(change);
            y.push(response);
        }
    }
    Ok(slope(&x, &y))
}
