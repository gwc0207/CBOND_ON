//! Strict-prior daily information and transition kernels for typed contracts.
//!
//! This module is a typed, pure-Rust implementation for exact declared signal
//! contracts and is called internally by the ordinary factor dispatcher.
//!
//! The context contract mirrors the Python daily-price kernels after their
//! parser has canonicalized codes: `price_by_code` is keyed by canonical market
//! code and contains the raw daily source rows.  Every function applies
//! `trade_date < score_date`, validates duplicate strict-prior `(date, code)`
//! records fail-closed, and derives its source-session calendar from the union
//! of all strict-prior daily-price rows.

use crate::typed_factor_daily::{TypedFactorDailyContext, TypedFactorDailyPriceRow};
use crate::typed_factor_math::{
    normalized_mutual_information_3state, normalized_transition_entropy,
};
use chrono::NaiveDate;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const TYPED_FACTOR_DAILY_INFORMATION_WINDOW: usize = 60;
pub const TYPED_FACTOR_DAILY_INFORMATION_MIN_OBSERVATIONS: usize = 45;
pub const TYPED_FACTOR_DAILY_INFORMATION_MIN_ADJACENT_PAIRS: usize = 40;
pub const TYPED_FACTOR_DAILY_INFORMATION_EPS: f64 = 1e-12;
const JEFFREYS_PSEUDOCOUNT: f64 = 0.5;

/// Explicit error surface for the isolated daily-information prototype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypedFactorDailyInformationError {
    DuplicateStrictPriorDate { code: String, trade_date: NaiveDate },
    UnknownSignal(String),
}

impl fmt::Display for TypedFactorDailyInformationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStrictPriorDate { code, trade_date } => write!(
                f,
                "typed_factor daily_price has duplicate strict-prior row for {code} on {trade_date}"
            ),
            Self::UnknownSignal(signal) => {
                write!(f, "unknown typed_factor daily-information signal: {signal}")
            }
        }
    }
}

impl Error for TypedFactorDailyInformationError {}

#[derive(Clone, Debug)]
struct PriceWindow {
    sessions: Vec<NaiveDate>,
    /// The source-row count (rather than reindexed length) preserves the
    /// return/topology kernel's `len(returns) >= 45` guard.
    observed_rows: usize,
    prev_close: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    amount: Vec<f64>,
    deal: Vec<f64>,
}

fn strict_price_rows<'a>(
    ctx: &'a TypedFactorDailyContext,
    code: &str,
) -> Result<Vec<&'a TypedFactorDailyPriceRow>, TypedFactorDailyInformationError> {
    let mut rows: Vec<&TypedFactorDailyPriceRow> = ctx
        .price_by_code
        .get(code)
        .map(|source| {
            source
                .iter()
                .filter(|row| row.trade_date < ctx.score_date)
                .collect()
        })
        .unwrap_or_default();
    rows.sort_by_key(|row| row.trade_date);
    for pair in rows.windows(2) {
        if pair[0].trade_date == pair[1].trade_date {
            return Err(TypedFactorDailyInformationError::DuplicateStrictPriorDate {
                code: code.to_string(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

/// Validate every strict-prior key before selecting an output code, matching
/// Python `_strict_history`: a duplicate anywhere in the source rejects the
/// complete calculation, not merely the requested security.
fn strict_calendar(
    ctx: &TypedFactorDailyContext,
) -> Result<Vec<NaiveDate>, TypedFactorDailyInformationError> {
    let mut calendar = BTreeSet::new();
    for code in ctx.price_by_code.keys() {
        for row in strict_price_rows(ctx, code)? {
            calendar.insert(row.trade_date);
        }
    }
    Ok(calendar.into_iter().collect())
}

/// Build the exact latest-up-to-60 source-session window for an anchored code.
///
/// `None` represents the Python groups-dictionary stale-tail guard: the code
/// does not reach the global strict-prior source anchor, so every output is
/// unavailable rather than computed from stale history.
fn anchored_price_window(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<Option<PriceWindow>, TypedFactorDailyInformationError> {
    let calendar = strict_calendar(ctx)?;
    let Some(anchor) = calendar.last().copied() else {
        return Ok(None);
    };
    let rows = strict_price_rows(ctx, code)?;
    if rows.last().map(|row| row.trade_date) != Some(anchor) {
        return Ok(None);
    }

    let first = calendar
        .len()
        .saturating_sub(TYPED_FACTOR_DAILY_INFORMATION_WINDOW);
    let sessions = calendar[first..].to_vec();
    let rows_by_date: BTreeMap<NaiveDate, &TypedFactorDailyPriceRow> = rows
        .iter()
        .copied()
        .map(|row| (row.trade_date, row))
        .collect();
    let observed_rows = rows
        .iter()
        .filter(|row| row.trade_date >= sessions[0])
        .count();

    let mut prev_close = Vec::with_capacity(sessions.len());
    let mut close = Vec::with_capacity(sessions.len());
    let mut volume = Vec::with_capacity(sessions.len());
    let mut amount = Vec::with_capacity(sessions.len());
    let mut deal = Vec::with_capacity(sessions.len());
    for session in &sessions {
        if let Some(row) = rows_by_date.get(session) {
            prev_close.push(row.prev_close_price);
            close.push(row.close_price);
            volume.push(row.volume);
            amount.push(row.amount);
            deal.push(row.deal);
        } else {
            prev_close.push(f64::NAN);
            close.push(f64::NAN);
            volume.push(f64::NAN);
            amount.push(f64::NAN);
            deal.push(f64::NAN);
        }
    }
    Ok(Some(PriceWindow {
        sessions,
        observed_rows,
        prev_close,
        close,
        volume,
        amount,
        deal,
    }))
}

#[inline]
fn log_positive(value: f64) -> f64 {
    if value.is_finite() && value > TYPED_FACTOR_DAILY_INFORMATION_EPS {
        value.ln()
    } else {
        f64::NAN
    }
}

#[inline]
fn log_return(previous_close: f64, close: f64) -> f64 {
    if previous_close.is_finite()
        && close.is_finite()
        && previous_close > TYPED_FACTOR_DAILY_INFORMATION_EPS
        && close > TYPED_FACTOR_DAILY_INFORMATION_EPS
    {
        (close / previous_close).ln()
    } else {
        f64::NAN
    }
}

/// Difference adjacent entries only.  A missing value is not carried across.
fn adjacent_change(values: &[f64]) -> Vec<f64> {
    let mut output = vec![f64::NAN; values.len()];
    for index in 1..values.len() {
        if values[index].is_finite() && values[index - 1].is_finite() {
            output[index] = values[index] - values[index - 1];
        }
    }
    output
}

fn lcc_amount_trade_size_input(window: &PriceWindow) -> (Vec<f64>, Vec<f64>) {
    let log_amount: Vec<f64> = window.amount.iter().copied().map(log_positive).collect();
    let log_deal: Vec<f64> = window.deal.iter().copied().map(log_positive).collect();
    let trade_size: Vec<f64> = log_amount
        .iter()
        .zip(log_deal.iter())
        .map(|(amount, deal)| {
            if amount.is_finite() && deal.is_finite() {
                *amount - *deal
            } else {
                f64::NAN
            }
        })
        .collect();
    (adjacent_change(&log_amount), adjacent_change(&trade_size))
}

fn lcc_volume_deal_input(window: &PriceWindow) -> (Vec<f64>, Vec<f64>) {
    let log_volume: Vec<f64> = window.volume.iter().copied().map(log_positive).collect();
    let log_deal: Vec<f64> = window.deal.iter().copied().map(log_positive).collect();
    (adjacent_change(&log_volume), adjacent_change(&log_deal))
}

/// `lcc_amount_trade_size_information60`: normalized 3x3 Jeffreys mutual
/// information between `sign(delta log(amount))` and
/// `sign(delta log(amount/deal))` on the last up-to-60 strict-prior global
/// source sessions.  Sign-zero uses the Python composition kernel's `1e-12`
/// tolerance.
pub fn lcc_amount_trade_size_information60(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyInformationError> {
    let Some(window) = anchored_price_window(ctx, code)? else {
        return Ok(f64::NAN);
    };
    let (amount_change, trade_size_change) = lcc_amount_trade_size_input(&window);
    Ok(normalized_mutual_information_3state(
        &amount_change,
        &trade_size_change,
        TYPED_FACTOR_DAILY_INFORMATION_EPS,
        TYPED_FACTOR_DAILY_INFORMATION_MIN_OBSERVATIONS,
        JEFFREYS_PSEUDOCOUNT,
    ))
}

/// `lcc_volume_deal_information60`: normalized 3x3 Jeffreys mutual
/// information between `sign(delta log(volume))` and `sign(delta log(deal))`
/// on the same strict-prior global-session window.
pub fn lcc_volume_deal_information60(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyInformationError> {
    let Some(window) = anchored_price_window(ctx, code)? else {
        return Ok(f64::NAN);
    };
    let (volume_change, deal_change) = lcc_volume_deal_input(&window);
    Ok(normalized_mutual_information_3state(
        &volume_change,
        &deal_change,
        TYPED_FACTOR_DAILY_INFORMATION_EPS,
        TYPED_FACTOR_DAILY_INFORMATION_MIN_OBSERVATIONS,
        JEFFREYS_PSEUDOCOUNT,
    ))
}

/// Return/log-liquidity arrays for the return-topology family.
///
/// Unlike the LCC family, the Python topology kernel keeps only the security's
/// raw rows and then rejects a state when successive rows are not adjacent in
/// the union source calendar.  The reindexed representation here produces the
/// same state mask while retaining `observed_rows` for its separate 45-row
/// availability guard.
fn topology_state_arrays(window: &PriceWindow, liquidity: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let returns: Vec<f64> = window
        .prev_close
        .iter()
        .zip(window.close.iter())
        .map(|(previous, close)| log_return(*previous, *close))
        .collect();
    let changes = adjacent_change(liquidity);
    (returns[1..].to_vec(), changes[1..].to_vec())
}

fn joint_return_amount_states(returns: &[f64], amount_change: &[f64]) -> Vec<i16> {
    returns
        .iter()
        .zip(amount_change.iter())
        .map(|(ret, amount)| {
            if !ret.is_finite() || !amount.is_finite() {
                return -1;
            }
            // The topology reference uses `np.sign` (exact zero, rather than
            // the LCC kernel's epsilon-zero convention).
            let return_state = if *ret < 0.0 {
                0_i16
            } else if *ret > 0.0 {
                2_i16
            } else {
                1_i16
            };
            let amount_state = if *amount < 0.0 {
                0_i16
            } else if *amount > 0.0 {
                2_i16
            } else {
                1_i16
            };
            return_state * 3 + amount_state
        })
        .collect()
}

fn topology_available(window: &PriceWindow) -> bool {
    // `sessions` is intentionally carried in the typed window to make the
    // source-calendar contract explicit.  Reindexing it is what prevents any
    // synthetic cross-gap state from becoming valid.
    !window.sessions.is_empty()
        && window.observed_rows >= TYPED_FACTOR_DAILY_INFORMATION_MIN_OBSERVATIONS
}

/// `rlmi_return_deal_sign_mutual_information60`: normalized 3x3 Jeffreys MI
/// between completed daily return-sign and `delta log(deal)` sign.  The latest
/// state must be a complete, calendar-adjacent state and at least 40 states
/// must be valid.
pub fn rlmi_return_deal_sign_mutual_information60(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyInformationError> {
    let Some(window) = anchored_price_window(ctx, code)? else {
        return Ok(f64::NAN);
    };
    if !topology_available(&window) {
        return Ok(f64::NAN);
    }
    let log_deal: Vec<f64> = window.deal.iter().copied().map(log_positive).collect();
    let (returns, deal_change) = topology_state_arrays(&window, &log_deal);
    Ok(normalized_mutual_information_3state(
        &returns,
        &deal_change,
        0.0,
        TYPED_FACTOR_DAILY_INFORMATION_MIN_ADJACENT_PAIRS,
        JEFFREYS_PSEUDOCOUNT,
    ))
}

/// `rjst_amount_joint_transition_entropy60`: unsmoothed entropy / `log(81)`
/// of adjacent transitions across the nine `(return-sign, amount-change-sign)`
/// states.  Both the latest state transition and at least 40 transitions must
/// be valid; any source-calendar gap therefore fails the terminal contract.
pub fn rjst_amount_joint_transition_entropy60(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<f64, TypedFactorDailyInformationError> {
    let Some(window) = anchored_price_window(ctx, code)? else {
        return Ok(f64::NAN);
    };
    if !topology_available(&window) {
        return Ok(f64::NAN);
    }
    let log_amount: Vec<f64> = window.amount.iter().copied().map(log_positive).collect();
    let (returns, amount_change) = topology_state_arrays(&window, &log_amount);
    let states = joint_return_amount_states(&returns, &amount_change);
    let transition_valid: Vec<bool> = states
        .windows(2)
        .map(|pair| pair[0] >= 0 && pair[1] >= 0)
        .collect();
    Ok(normalized_transition_entropy(
        &states,
        &transition_valid,
        9,
        TYPED_FACTOR_DAILY_INFORMATION_MIN_ADJACENT_PAIRS,
        0.0,
        true,
        true,
    ))
}

/// Return the non-zero 9x9 RJST transition counts after all Rust-side window,
/// source-calendar, terminal-state, and minimum-observation guards. The
/// opt-in PyO3 kernel uses these exact counts with NumPy only for the final
/// entropy reduction, matching the Python reference accumulation primitive
/// bit-for-bit without delegating factor construction back to Python.
pub fn rjst_amount_joint_transition_nonzero_counts(
    ctx: &TypedFactorDailyContext,
    code: &str,
) -> Result<Option<Vec<f64>>, TypedFactorDailyInformationError> {
    let Some(window) = anchored_price_window(ctx, code)? else {
        return Ok(None);
    };
    if !topology_available(&window) {
        return Ok(None);
    }
    let log_amount: Vec<f64> = window.amount.iter().copied().map(log_positive).collect();
    let (returns, amount_change) = topology_state_arrays(&window, &log_amount);
    let states = joint_return_amount_states(&returns, &amount_change);
    let transition_valid: Vec<bool> = states
        .windows(2)
        .map(|pair| pair[0] >= 0 && pair[1] >= 0)
        .collect();
    if states
        .last()
        .copied()
        .filter(|state| *state >= 0 && *state < 9)
        .is_none()
        || !transition_valid.last().copied().unwrap_or(false)
    {
        return Ok(None);
    }

    let mut counts = [0.0_f64; 81];
    let mut valid_count = 0usize;
    for (index, is_valid) in transition_valid.iter().copied().enumerate() {
        if !is_valid {
            continue;
        }
        let from = states[index];
        let to = states[index + 1];
        if !(0..9).contains(&from) || !(0..9).contains(&to) {
            return Ok(None);
        }
        counts[(from as usize) * 9 + to as usize] += 1.0;
        valid_count += 1;
    }
    if valid_count < TYPED_FACTOR_DAILY_INFORMATION_MIN_ADJACENT_PAIRS {
        return Ok(None);
    }
    Ok(Some(
        counts.into_iter().filter(|count| *count > 0.0).collect(),
    ))
}

/// Dispatch only this inactive four-signal information/transition subset.
pub fn compute_daily_information_signal(
    ctx: &TypedFactorDailyContext,
    code: &str,
    signal: &str,
) -> Result<f64, TypedFactorDailyInformationError> {
    match signal {
        "lcc_amount_trade_size_information60" => lcc_amount_trade_size_information60(ctx, code),
        "lcc_volume_deal_information60" => lcc_volume_deal_information60(ctx, code),
        "rjst_amount_joint_transition_entropy60" => {
            rjst_amount_joint_transition_entropy60(ctx, code)
        }
        "rlmi_return_deal_sign_mutual_information60" => {
            rlmi_return_deal_sign_mutual_information60(ctx, code)
        }
        other => Err(TypedFactorDailyInformationError::UnknownSignal(
            other.to_string(),
        )),
    }
}
