//! Exact Rust formulas for the research-only R88 intraday contract gap.
//!
//! This module intentionally contains no Python bindings or factor-dispatch
//! wiring.  The caller owns the physical score-day proof: it must pass rows
//! whose exchange timestamp belongs to the requested score day.  The pure
//! functions below then apply the remaining hard PIT rule themselves: a row
//! is visible only through local `14:29:00` (with the same microsecond-floor
//! comparison used by pandas `Series.dt.time`).
//!
//! The nine outputs are deliberately kept in their Python-family semantics:
//!
//! * expansion metrics keep all physically-visible rows, as the Python
//!   expansion catalogue does;
//! * execution-discreteness metrics additionally restrict to the two
//!   continuous trading sessions and reject all invalid event paths;
//! * `book_quote_dislocation` preserves its catalogue-local tolerant
//!   `nanmean` behavior instead of inheriting the stricter best-book gate.
//!
//! That separation matters: a malformed quote volume invalidates
//! `exp_stick_quote_update_rate`, but it must not turn independent price-path
//! outputs into missing values.

use std::cmp::Ordering;

const EPS: f64 = 1e-12;
const PRICE_REL_TOL: f64 = 1e-8;
const NS_PER_SECOND: i64 = 1_000_000_000;
const NS_PER_MICROSECOND: i64 = 1_000;
const NS_PER_MINUTE: i64 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: i64 = 60 * NS_PER_MINUTE;
const NS_PER_DAY: i64 = 24 * NS_PER_HOUR;

const MORNING_START: i64 = clock_ns(9, 30, 0, 0);
const MORNING_END: i64 = clock_ns(11, 30, 0, 0);
const AFTERNOON_START: i64 = clock_ns(13, 0, 0, 0);
const STRICT_CUTOFF: i64 = clock_ns(14, 29, 0, 0);

const EXECDISC_MIN_ROWS: usize = 12;
const EXECDISC_MIN_TRADE_EVENTS: usize = 8;
const EXECDISC_MIN_NONZERO_UPDATES: usize = 6;
const EXECDISC_MIN_DIRECTION_RUNS: usize = 3;

/// Build a local wall-clock nanosecond value for one score-day snapshot.
pub const fn clock_ns(hour: i64, minute: i64, second: i64, nanosecond: i64) -> i64 {
    hour * NS_PER_HOUR + minute * NS_PER_MINUTE + second * NS_PER_SECOND + nanosecond
}

/// Fields directly consumed by at least one R88 formula in this module.
///
/// Dispatcher integration must enforce the Python family-level schema before
/// parsing rows.  In particular, `book_quote_dislocation` retains the
/// `factor_mining_intraday_catalog_v1` full-panel schema even though its
/// terminal formula only reads `last`, `ask_price1`, and `bid_price1`.
pub const R88_DIRECT_REQUIRED_PANEL_COLUMNS: &[&str] = &[
    "trade_time",
    "last",
    "volume",
    "amount",
    "num_trades",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
];

pub const R88_INTRADAY_SIGNALS: &[&str] = &[
    "exp_rotation_segment_return_dispersion",
    "exp_exec_amount_concentration_impact",
    "exp_noise_median_mean_abs_return_ratio",
    "exp_noise_variance_ratio_2",
    "exp_noise_variance_ratio_5",
    "exp_stick_quote_update_rate",
    "execdisc_direction_reversal_rate",
    "execdisc_step_multiplicity_entropy",
    "book_quote_dislocation",
];

/// One physical T-day snapshot for a single bond.
///
/// `time_ns` is Asia/Shanghai local time since midnight.  The dispatcher must
/// already have excluded a relabelled prior-day row before constructing this
/// type.  `seq` resolves equal exchange timestamps exactly as Python's stable
/// `sort_values([trade_time, seq], kind="mergesort")` path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct R88IntradayRow {
    pub time_ns: i64,
    pub seq: i64,
    pub last: f64,
    pub volume: f64,
    pub amount: f64,
    pub num_trades: f64,
    pub ask_price1: f64,
    pub bid_price1: f64,
    pub ask_volume1: f64,
    pub bid_volume1: f64,
}

/// The exact nine research-only R88 metrics for one `(score_day, code)` path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct R88IntradayMetrics {
    pub exp_rotation_segment_return_dispersion: f64,
    pub exp_exec_amount_concentration_impact: f64,
    pub exp_noise_median_mean_abs_return_ratio: f64,
    pub exp_noise_variance_ratio_2: f64,
    pub exp_noise_variance_ratio_5: f64,
    pub exp_stick_quote_update_rate: f64,
    pub execdisc_direction_reversal_rate: f64,
    pub execdisc_step_multiplicity_entropy: f64,
    pub book_quote_dislocation: f64,
}

impl R88IntradayMetrics {
    fn nan() -> Self {
        Self {
            exp_rotation_segment_return_dispersion: f64::NAN,
            exp_exec_amount_concentration_impact: f64::NAN,
            exp_noise_median_mean_abs_return_ratio: f64::NAN,
            exp_noise_variance_ratio_2: f64::NAN,
            exp_noise_variance_ratio_5: f64::NAN,
            exp_stick_quote_update_rate: f64::NAN,
            execdisc_direction_reversal_rate: f64::NAN,
            execdisc_step_multiplicity_entropy: f64::NAN,
            book_quote_dislocation: f64::NAN,
        }
    }

    fn finite_or_nan(mut self) -> Self {
        for value in [
            &mut self.exp_rotation_segment_return_dispersion,
            &mut self.exp_exec_amount_concentration_impact,
            &mut self.exp_noise_median_mean_abs_return_ratio,
            &mut self.exp_noise_variance_ratio_2,
            &mut self.exp_noise_variance_ratio_5,
            &mut self.exp_stick_quote_update_rate,
            &mut self.execdisc_direction_reversal_rate,
            &mut self.execdisc_step_multiplicity_entropy,
            &mut self.book_quote_dislocation,
        ] {
            if !value.is_finite() {
                *value = f64::NAN;
            }
        }
        self
    }

    /// Read an exact named signal without exposing a second public dispatch.
    pub fn value(&self, signal: &str) -> f64 {
        match signal {
            "exp_rotation_segment_return_dispersion" => self.exp_rotation_segment_return_dispersion,
            "exp_exec_amount_concentration_impact" => self.exp_exec_amount_concentration_impact,
            "exp_noise_median_mean_abs_return_ratio" => self.exp_noise_median_mean_abs_return_ratio,
            "exp_noise_variance_ratio_2" => self.exp_noise_variance_ratio_2,
            "exp_noise_variance_ratio_5" => self.exp_noise_variance_ratio_5,
            "exp_stick_quote_update_rate" => self.exp_stick_quote_update_rate,
            "execdisc_direction_reversal_rate" => self.execdisc_direction_reversal_rate,
            "execdisc_step_multiplicity_entropy" => self.execdisc_step_multiplicity_entropy,
            "book_quote_dislocation" => self.book_quote_dislocation,
            _ => f64::NAN,
        }
    }
}

/// `true` only for the nine factor signal names owned by this module.
pub fn is_r88_intraday_signal(signal: &str) -> bool {
    R88_INTRADAY_SIGNALS.contains(&signal)
}

/// Return all nine formula values for one physical T-day path.
pub fn r88_intraday_metrics(rows: &[R88IntradayRow]) -> R88IntradayMetrics {
    let expansion_rows = ordered_pit_rows(rows);
    let execution_rows = ordered_continuous_rows(rows);

    let mut out = R88IntradayMetrics::nan();
    out.exp_rotation_segment_return_dispersion =
        rotation_segment_return_dispersion(&expansion_rows);
    out.exp_exec_amount_concentration_impact = amount_concentration_impact(&expansion_rows);
    out.exp_noise_median_mean_abs_return_ratio = median_mean_abs_return_ratio(&expansion_rows);
    out.exp_noise_variance_ratio_2 = variance_ratio(&expansion_rows, 2);
    out.exp_noise_variance_ratio_5 = variance_ratio(&expansion_rows, 5);
    out.exp_stick_quote_update_rate = quote_update_rate(&expansion_rows);
    out.book_quote_dislocation = quote_dislocation(&expansion_rows);

    let execution = execution_discreteness_metrics(&execution_rows);
    out.execdisc_direction_reversal_rate = execution.direction_reversal_rate;
    out.execdisc_step_multiplicity_entropy = execution.step_multiplicity_entropy;
    out.finite_or_nan()
}

/// Read one R88 intraday signal. Unknown names fail closed as `NaN`.
pub fn r88_intraday_signal(rows: &[R88IntradayRow], signal: &str) -> f64 {
    r88_intraday_metrics(rows).value(signal)
}

fn python_clock_ns(time_ns: i64) -> i64 {
    // pandas Timestamp.dt.time has microsecond rather than nanosecond
    // comparison semantics.  Floor in Euclidean arithmetic so a negative
    // value can never be rounded into the visible score-day interval.
    time_ns.div_euclid(NS_PER_MICROSECOND) * NS_PER_MICROSECOND
}

fn pit_visible(time_ns: i64) -> bool {
    let value = python_clock_ns(time_ns);
    (0..NS_PER_DAY).contains(&value) && value <= STRICT_CUTOFF
}

fn session_label(time_ns: i64) -> Option<u8> {
    let value = python_clock_ns(time_ns);
    if (MORNING_START..=MORNING_END).contains(&value) {
        Some(0)
    } else if (AFTERNOON_START..=STRICT_CUTOFF).contains(&value) {
        Some(1)
    } else {
        None
    }
}

fn ordered_rows(
    rows: &[R88IntradayRow],
    predicate: impl Fn(&R88IntradayRow) -> bool,
) -> Vec<R88IntradayRow> {
    let mut kept: Vec<(usize, R88IntradayRow)> = rows
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, row)| predicate(row))
        .collect();
    kept.sort_by(|left, right| {
        left.1
            .time_ns
            .cmp(&right.1.time_ns)
            .then_with(|| left.1.seq.cmp(&right.1.seq))
            .then_with(|| left.0.cmp(&right.0))
    });
    kept.into_iter().map(|(_, row)| row).collect()
}

fn ordered_pit_rows(rows: &[R88IntradayRow]) -> Vec<R88IntradayRow> {
    ordered_rows(rows, |row| pit_visible(row.time_ns))
}

fn ordered_continuous_rows(rows: &[R88IntradayRow]) -> Vec<R88IntradayRow> {
    ordered_rows(rows, |row| session_label(row.time_ns).is_some())
}

fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() <= EPS {
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

fn prices(rows: &[R88IntradayRow]) -> Option<Vec<f64>> {
    if rows.len() < 2 {
        return None;
    }
    let values: Vec<f64> = rows.iter().map(|row| row.last).collect();
    if values.iter().all(|value| value.is_finite() && *value > 0.0) {
        Some(values)
    } else {
        None
    }
}

fn returns(prices: &[f64]) -> Option<Vec<f64>> {
    if prices.len() < 3 {
        return None;
    }
    let values: Vec<f64> = prices
        .windows(2)
        .map(|pair| pair[1] / pair[0] - 1.0)
        .collect();
    if values.iter().all(|value| value.is_finite()) {
        Some(values)
    } else {
        None
    }
}

fn counter_increments(
    rows: &[R88IntradayRow],
    field: impl Fn(&R88IntradayRow) -> f64,
) -> Option<Vec<f64>> {
    if rows.len() < 3 {
        return None;
    }
    let values: Vec<f64> = rows.iter().map(field).collect();
    if !values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        return None;
    }
    let increments: Vec<f64> = values.windows(2).map(|pair| pair[1] - pair[0]).collect();
    if increments
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        Some(increments)
    } else {
        None
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() || !values.iter().all(|value| value.is_finite()) {
        return f64::NAN;
    }
    let value = values.iter().sum::<f64>() / values.len() as f64;
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}

fn std_population(values: &[f64]) -> f64 {
    let center = mean(values);
    if !center.is_finite() {
        return f64::NAN;
    }
    let variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    if variance.is_finite() && variance >= 0.0 {
        variance.sqrt()
    } else {
        f64::NAN
    }
}

fn sample_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let center = mean(values);
    if !center.is_finite() {
        return f64::NAN;
    }
    let variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    if variance.is_finite() {
        variance
    } else {
        f64::NAN
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() || !values.iter().all(|value| value.is_finite()) {
        return f64::NAN;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn rotation_segment_return_dispersion(rows: &[R88IntradayRow]) -> f64 {
    let Some(path) = prices(rows) else {
        return f64::NAN;
    };
    if returns(&path).is_none() {
        return f64::NAN;
    }

    let mut sessions = [Vec::new(), Vec::new(), Vec::new()];
    for (row, price) in rows.iter().zip(path.iter().copied()) {
        let clock = python_clock_ns(row.time_ns);
        let bucket = if clock < clock_ns(10, 30, 0, 0) {
            0
        } else if clock < clock_ns(13, 30, 0, 0) {
            1
        } else {
            2
        };
        sessions[bucket].push(price);
    }
    if sessions.iter().any(|values| values.len() < 2) {
        return f64::NAN;
    }
    let mut session_returns = [0.0; 3];
    for (index, values) in sessions.iter().enumerate() {
        let first = values[0];
        let last = values[values.len() - 1];
        let value = safe_div(last - first, first);
        if !value.is_finite() {
            return f64::NAN;
        }
        session_returns[index] = value;
    }
    std_population(&session_returns)
}

fn amount_concentration_impact(rows: &[R88IntradayRow]) -> f64 {
    let Some(path) = prices(rows) else {
        return f64::NAN;
    };
    let Some(path_returns) = returns(&path) else {
        return f64::NAN;
    };
    let Some(amount) = counter_increments(rows, |row| row.amount) else {
        return f64::NAN;
    };
    let total = amount.iter().sum::<f64>();
    if !total.is_finite() || total <= EPS {
        return f64::NAN;
    }
    let value = amount
        .iter()
        .zip(path_returns.iter())
        .map(|(weight, ret)| weight / total * ret.abs())
        .sum::<f64>();
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}

fn variance_ratio(rows: &[R88IntradayRow], horizon: usize) -> f64 {
    let Some(path) = prices(rows) else {
        return f64::NAN;
    };
    let Some(path_returns) = returns(&path) else {
        return f64::NAN;
    };
    if horizon <= 1 {
        return f64::NAN;
    }
    let count = path_returns.len() / horizon * horizon;
    if count < horizon * 3 {
        return f64::NAN;
    }
    let fine = &path_returns[..count];
    let fine_variance = sample_variance(fine);
    if !fine_variance.is_finite() || fine_variance <= EPS {
        return f64::NAN;
    }
    let coarse: Vec<f64> = fine
        .chunks_exact(horizon)
        .map(|chunk| chunk.iter().sum::<f64>())
        .collect();
    if coarse.len() < 3 {
        return f64::NAN;
    }
    safe_div(sample_variance(&coarse), horizon as f64 * fine_variance)
}

fn median_mean_abs_return_ratio(rows: &[R88IntradayRow]) -> f64 {
    let Some(path) = prices(rows) else {
        return f64::NAN;
    };
    let Some(path_returns) = returns(&path) else {
        return f64::NAN;
    };
    if path_returns.len() < 6 {
        return f64::NAN;
    }
    let absolute: Vec<f64> = path_returns.iter().map(|value| value.abs()).collect();
    safe_div(median(absolute.clone()), mean(&absolute))
}

fn quote_update_rate(rows: &[R88IntradayRow]) -> f64 {
    let Some(path) = prices(rows) else {
        return f64::NAN;
    };
    let ask: Vec<f64> = rows.iter().map(|row| row.ask_price1).collect();
    let bid: Vec<f64> = rows.iter().map(|row| row.bid_price1).collect();
    let ask_volume: Vec<f64> = rows.iter().map(|row| row.ask_volume1).collect();
    let bid_volume: Vec<f64> = rows.iter().map(|row| row.bid_volume1).collect();
    if ask.len() < 2
        || !ask.iter().all(|value| value.is_finite() && *value > 0.0)
        || !bid.iter().all(|value| value.is_finite() && *value > 0.0)
        || !ask_volume
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        || !bid_volume
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        || ask.iter().zip(bid.iter()).any(|(ask, bid)| ask < bid)
    {
        return f64::NAN;
    }
    let mid: Vec<f64> = ask
        .iter()
        .zip(bid.iter())
        .map(|(ask, bid)| (ask + bid) / 2.0)
        .collect();
    let mid_tolerance = EPS.max(mid[0].abs() * 1e-6);
    // `path` is intentionally read above: Python's family validates the
    // price path even though this selected output is quote-only.
    let _ = path;
    let changed = mid
        .windows(2)
        .filter(|pair| (pair[1] - pair[0]).abs() > mid_tolerance)
        .count();
    changed as f64 / (mid.len() - 1) as f64
}

fn quote_dislocation(rows: &[R88IntradayRow]) -> f64 {
    // This is the narrower `book_quote_dislocation` formula from
    // `_feature_row_intraday`.  Do not apply quote-volume/book-ladder gates:
    // the Python catalogue does not apply them to this output.
    if rows.len() < 5 {
        return f64::NAN;
    }
    let Some(path) = prices(rows) else {
        return f64::NAN;
    };
    let mut has_finite_mid = false;
    let mut values = Vec::with_capacity(rows.len());
    for (row, last) in rows.iter().zip(path.iter().copied()) {
        let mid = (row.ask_price1 + row.bid_price1) / 2.0;
        if mid.is_finite() {
            has_finite_mid = true;
        }
        let value = (last - mid).abs() / mid;
        if !value.is_nan() {
            values.push(value);
        }
    }
    if !has_finite_mid || values.is_empty() {
        return f64::NAN;
    }
    let value = values.iter().sum::<f64>() / values.len() as f64;
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}

#[derive(Clone, Copy, Debug)]
struct ExecutionDiscretenessMetrics {
    direction_reversal_rate: f64,
    step_multiplicity_entropy: f64,
}

impl ExecutionDiscretenessMetrics {
    fn nan() -> Self {
        Self {
            direction_reversal_rate: f64::NAN,
            step_multiplicity_entropy: f64::NAN,
        }
    }
}

fn execution_discreteness_metrics(rows: &[R88IntradayRow]) -> ExecutionDiscretenessMetrics {
    if rows.len() < EXECDISC_MIN_ROWS {
        return ExecutionDiscretenessMetrics::nan();
    }
    let mut sessions = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(session) = session_label(row.time_ns) else {
            return ExecutionDiscretenessMetrics::nan();
        };
        if !row.last.is_finite()
            || !row.num_trades.is_finite()
            || row.last <= 0.0
            || row.num_trades < 0.0
        {
            return ExecutionDiscretenessMetrics::nan();
        }
        sessions.push(session);
    }
    if rows
        .windows(2)
        .any(|pair| pair[0].time_ns >= pair[1].time_ns)
    {
        // Python checks `duplicated` and strict monotonicity before it
        // constructs trade events.  `seq` cannot rescue duplicate clocks.
        return ExecutionDiscretenessMetrics::nan();
    }
    for session in [0_u8, 1_u8] {
        let mut previous: Option<i64> = None;
        for (row, row_session) in rows.iter().zip(sessions.iter().copied()) {
            if row_session == session {
                if let Some(last_time) = previous {
                    let gap = row.time_ns - last_time;
                    if gap <= 0 || gap > 20 * NS_PER_MINUTE {
                        return ExecutionDiscretenessMetrics::nan();
                    }
                }
                previous = Some(row.time_ns);
            }
        }
        if previous.is_none() {
            return ExecutionDiscretenessMetrics::nan();
        }
    }

    let mut event_delta = Vec::new();
    let mut event_moving = Vec::new();
    let mut event_sessions = Vec::new();
    for index in 1..rows.len() {
        let prior = rows[index - 1];
        let current = rows[index];
        let increment = current.num_trades - prior.num_trades;
        if !increment.is_finite() || increment < 0.0 {
            return ExecutionDiscretenessMetrics::nan();
        }
        if sessions[index] != sessions[index - 1] || increment <= 0.0 {
            continue;
        }
        let delta = current.last - prior.last;
        let scale = current.last.max(prior.last);
        let moving = delta.abs() > PRICE_REL_TOL * scale;
        if !delta.is_finite() || !scale.is_finite() {
            return ExecutionDiscretenessMetrics::nan();
        }
        event_delta.push(delta);
        event_moving.push(moving);
        event_sessions.push(sessions[index]);
    }
    if event_delta.len() < EXECDISC_MIN_TRADE_EVENTS {
        return ExecutionDiscretenessMetrics::nan();
    }
    let mut nonzero_delta = Vec::new();
    let mut nonzero_sessions = Vec::new();
    for ((delta, moving), session) in event_delta
        .iter()
        .copied()
        .zip(event_moving.iter().copied())
        .zip(event_sessions.iter().copied())
    {
        if moving {
            nonzero_delta.push(delta);
            nonzero_sessions.push(session);
        }
    }
    if nonzero_delta.len() < EXECDISC_MIN_NONZERO_UPDATES {
        return ExecutionDiscretenessMetrics::nan();
    }

    let step_multiplicity_entropy = step_multiplicity_entropy(&nonzero_delta);
    let direction_reversal_rate = direction_reversal_rate(&nonzero_delta, &nonzero_sessions);
    ExecutionDiscretenessMetrics {
        direction_reversal_rate,
        step_multiplicity_entropy,
    }
}

fn normalized_entropy(counts: &[usize]) -> f64 {
    if counts.is_empty() {
        return f64::NAN;
    }
    let total: usize = counts.iter().sum();
    if total == 0 {
        return f64::NAN;
    }
    let positive: Vec<usize> = counts.iter().copied().filter(|count| *count > 0).collect();
    if positive.len() <= 1 {
        return 0.0;
    }
    let total = total as f64;
    let value = -positive
        .iter()
        .map(|count| {
            let probability = *count as f64 / total;
            probability * probability.ln()
        })
        .sum::<f64>()
        / (positive.len() as f64).ln();
    if value.is_finite() && value >= -EPS {
        value
    } else {
        f64::NAN
    }
}

fn round_ties_even(value: f64) -> Option<i64> {
    if !value.is_finite() || value <= 0.0 || value > i64::MAX as f64 {
        return None;
    }
    // NumPy's `np.rint` is IEEE round-to-nearest, ties-to-even.  Keep the
    // native primitive rather than compare a binary fraction to literal 0.5;
    // the latter can diverge after the preceding scale division.
    let rounded = value.round_ties_even();
    if rounded > i64::MAX as f64 {
        return None;
    }
    Some(rounded as i64)
}

fn step_multiplicity_entropy(nonzero_delta: &[f64]) -> f64 {
    let steps: Vec<f64> = nonzero_delta.iter().map(|value| value.abs()).collect();
    if steps.len() < EXECDISC_MIN_NONZERO_UPDATES
        || !steps.iter().all(|value| value.is_finite() && *value > EPS)
    {
        return f64::NAN;
    }
    let lower_count = steps
        .len()
        .min(((steps.len() as f64 * 0.25).ceil() as usize).max(3));
    let mut sorted = steps.clone();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let small_scale = median(sorted[..lower_count].to_vec());
    if !small_scale.is_finite() || small_scale <= EPS {
        return f64::NAN;
    }
    let mut multiples = Vec::with_capacity(steps.len());
    for step in steps {
        let Some(rounded) = round_ties_even(step / small_scale) else {
            return f64::NAN;
        };
        multiples.push(rounded.max(1));
    }
    multiples.sort_unstable();
    let mut counts = Vec::new();
    let mut current_count = 0usize;
    let mut prior: Option<i64> = None;
    for value in multiples {
        if prior.is_some_and(|old| old != value) {
            counts.push(current_count);
            current_count = 0;
        }
        current_count += 1;
        prior = Some(value);
    }
    if current_count > 0 {
        counts.push(current_count);
    }
    normalized_entropy(&counts)
}

fn run_lengths(values: &[i8]) -> Vec<usize> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut length = 1usize;
    for pair in values.windows(2) {
        if pair[1] == pair[0] {
            length += 1;
        } else {
            out.push(length);
            length = 1;
        }
    }
    out.push(length);
    out
}

fn direction_reversal_rate(nonzero_delta: &[f64], sessions: &[u8]) -> f64 {
    if nonzero_delta.len() < EXECDISC_MIN_NONZERO_UPDATES
        || nonzero_delta.len() != sessions.len()
        || !nonzero_delta
            .iter()
            .all(|value| value.is_finite() && value.abs() > EPS)
    {
        return f64::NAN;
    }
    let mut total = 0usize;
    let mut transitions = 0usize;
    let mut reversals = 0usize;
    let mut all_runs = Vec::new();
    for session in [0_u8, 1_u8] {
        let local: Vec<i8> = nonzero_delta
            .iter()
            .copied()
            .zip(sessions.iter().copied())
            .filter_map(|(delta, signal_session)| {
                (signal_session == session).then_some(if delta > 0.0 { 1 } else { -1 })
            })
            .collect();
        if local.is_empty() {
            continue;
        }
        total += local.len();
        if local.len() > 1 {
            transitions += local.len() - 1;
            reversals += local.windows(2).filter(|pair| pair[1] != pair[0]).count();
        }
        all_runs.extend(run_lengths(&local));
    }
    if total < EXECDISC_MIN_NONZERO_UPDATES
        || transitions == 0
        || all_runs.len() < EXECDISC_MIN_DIRECTION_RUNS
    {
        return f64::NAN;
    }
    reversals as f64 / transitions as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(time_ns: i64, seq: i64, last: f64, amount: f64, trades: f64) -> R88IntradayRow {
        R88IntradayRow {
            time_ns,
            seq,
            last,
            volume: amount / 10.0,
            amount,
            num_trades: trades,
            ask_price1: last + 0.1,
            bid_price1: last - 0.1,
            ask_volume1: 10.0,
            bid_volume1: 11.0,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(actual.is_finite(), "expected finite value, got {actual:?}");
        assert!(
            (actual - expected).abs() <= 1e-12,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn strict_cutoff_uses_pandas_microsecond_semantics() {
        assert!(pit_visible(clock_ns(14, 29, 0, 999)));
        assert!(!pit_visible(clock_ns(14, 29, 0, 1_000)));
        assert!(!pit_visible(clock_ns(14, 29, 30, 0)));
        assert!(!pit_visible(clock_ns(14, 30, 0, 0)));
    }

    #[test]
    fn expansion_formulae_match_closed_form_references_and_ignore_late_rows() {
        let clocks = [
            clock_ns(9, 30, 0, 0),
            clock_ns(9, 31, 0, 0),
            clock_ns(9, 32, 0, 0),
            clock_ns(9, 33, 0, 0),
            clock_ns(9, 34, 0, 0),
            clock_ns(9, 35, 0, 0),
            clock_ns(10, 30, 0, 0),
            clock_ns(10, 31, 0, 0),
            clock_ns(10, 32, 0, 0),
            clock_ns(10, 33, 0, 0),
            clock_ns(10, 34, 0, 0),
            clock_ns(10, 35, 0, 0),
            clock_ns(13, 30, 0, 0),
            clock_ns(13, 31, 0, 0),
            clock_ns(13, 32, 0, 0),
            clock_ns(13, 33, 0, 0),
            clock_ns(13, 34, 0, 0),
            clock_ns(13, 35, 0, 0),
        ];
        let prices = [
            100.0, 100.5, 100.2, 100.8, 100.6, 101.0, 102.0, 101.5, 102.5, 103.0, 103.5, 104.0,
            105.0, 104.5, 105.5, 106.0, 107.0, 108.0,
        ];
        let mut rows: Vec<_> = clocks
            .iter()
            .copied()
            .zip(prices)
            .enumerate()
            .map(|(index, (time, price))| {
                let mut out = row(
                    time,
                    (99 - index) as i64,
                    price,
                    (index * (index + 3)) as f64,
                    index as f64,
                );
                out.ask_price1 = price + if index % 3 == 0 { 0.2 } else { 0.1 };
                out.bid_price1 = price - if index % 4 == 0 { 0.2 } else { 0.1 };
                out
            })
            .collect();
        let baseline = r88_intraday_metrics(&rows);
        assert!(baseline.exp_rotation_segment_return_dispersion.is_finite());
        assert!(baseline.exp_exec_amount_concentration_impact.is_finite());
        assert!(baseline.exp_noise_median_mean_abs_return_ratio.is_finite());
        assert!(baseline.exp_noise_variance_ratio_2.is_finite());
        assert!(baseline.exp_noise_variance_ratio_5.is_finite());
        assert!(baseline.exp_stick_quote_update_rate.is_finite());
        assert!(baseline.book_quote_dislocation.is_finite());

        let early = (101.0 - 100.0) / 100.0;
        let middle = (104.0 - 102.0) / 102.0;
        let late = (108.0 - 105.0) / 105.0;
        let expected_rotation = std_population(&[early, middle, late]);
        assert_close(
            baseline.exp_rotation_segment_return_dispersion,
            expected_rotation,
        );

        let returns: Vec<f64> = prices
            .windows(2)
            .map(|pair| pair[1] / pair[0] - 1.0)
            .collect();
        let increments: Vec<f64> = (1..prices.len())
            .map(|index| (index * (index + 3)) as f64 - ((index - 1) * (index + 2)) as f64)
            .collect();
        let total = increments.iter().sum::<f64>();
        let expected_impact = increments
            .iter()
            .zip(returns.iter())
            .map(|(amount, ret)| amount / total * ret.abs())
            .sum::<f64>();
        assert_close(
            baseline.exp_exec_amount_concentration_impact,
            expected_impact,
        );

        let absolute: Vec<f64> = returns.iter().map(|value| value.abs()).collect();
        assert_close(
            baseline.exp_noise_median_mean_abs_return_ratio,
            median(absolute.clone()) / mean(&absolute),
        );
        let vr = |horizon: usize| {
            let count = returns.len() / horizon * horizon;
            let fine = &returns[..count];
            let coarse: Vec<f64> = fine
                .chunks_exact(horizon)
                .map(|chunk| chunk.iter().sum::<f64>())
                .collect();
            sample_variance(&coarse) / (horizon as f64 * sample_variance(fine))
        };
        assert_close(baseline.exp_noise_variance_ratio_2, vr(2));
        assert_close(baseline.exp_noise_variance_ratio_5, vr(5));

        rows.push(row(clock_ns(14, 29, 30, 0), 999, 9_999.0, 1e12, 1e12));
        rows.push(row(clock_ns(14, 30, 0, 0), 1_000, 1.0, 1.0, 1.0));
        let with_late = r88_intraday_metrics(&rows);
        assert_eq!(
            with_late.exp_rotation_segment_return_dispersion.to_bits(),
            baseline.exp_rotation_segment_return_dispersion.to_bits()
        );
        assert_eq!(
            with_late.exp_exec_amount_concentration_impact.to_bits(),
            baseline.exp_exec_amount_concentration_impact.to_bits()
        );
        assert_eq!(
            with_late.exp_noise_variance_ratio_5.to_bits(),
            baseline.exp_noise_variance_ratio_5.to_bits()
        );
    }

    #[test]
    fn execution_discreteness_matches_multiplicity_and_direction_references() {
        let steps = [1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 3.0, -3.0, 1.0, -1.0, 2.0];
        let mut last = 100.0;
        let mut rows = vec![row(clock_ns(9, 30, 0, 0), 0, last, 0.0, 0.0)];
        for (index, step) in steps[..6].iter().copied().enumerate() {
            last += step;
            rows.push(row(
                clock_ns(9, 30, (index + 1) as i64, 0),
                (50 - index) as i64,
                last,
                (index + 1) as f64,
                (index + 1) as f64,
            ));
        }
        // The Python reference requires nonempty observations in both
        // continuous sessions. The noon bridge itself is not an event.
        rows.push(row(clock_ns(13, 0, 0, 0), 90, last, 7.0, 7.0));
        for (offset, step) in steps[6..].iter().copied().enumerate() {
            last += step;
            let index = offset + 7;
            rows.push(row(
                clock_ns(13, 0, (offset + 1) as i64, 0),
                (50 - index) as i64,
                last,
                (index + 1) as f64,
                (index + 1) as f64,
            ));
        }
        let metrics = r88_intraday_metrics(&rows);
        assert_close(metrics.execdisc_direction_reversal_rate, 1.0);
        let expected_entropy = normalized_entropy(&[6, 3, 2]);
        assert_close(metrics.execdisc_step_multiplicity_entropy, expected_entropy);

        rows.reverse();
        let reversed = r88_intraday_metrics(&rows);
        assert_eq!(
            reversed.execdisc_direction_reversal_rate.to_bits(),
            metrics.execdisc_direction_reversal_rate.to_bits()
        );
        assert_eq!(
            reversed.execdisc_step_multiplicity_entropy.to_bits(),
            metrics.execdisc_step_multiplicity_entropy.to_bits()
        );
    }

    #[test]
    fn execution_discreteness_rejects_duplicate_clock_and_counter_reset() {
        let mut rows = (0..12)
            .map(|index| {
                row(
                    clock_ns(9, 30, index, 0),
                    index,
                    100.0 + index as f64,
                    index as f64,
                    index as f64,
                )
            })
            .collect::<Vec<_>>();
        rows[7].time_ns = rows[6].time_ns;
        assert!(r88_intraday_metrics(&rows)
            .execdisc_direction_reversal_rate
            .is_nan());

        rows[7].time_ns = clock_ns(9, 30, 7, 0);
        rows[9].num_trades = 1.0;
        assert!(r88_intraday_metrics(&rows)
            .execdisc_step_multiplicity_entropy
            .is_nan());
    }

    #[test]
    fn family_local_failures_do_not_cross_contaminate_other_outputs() {
        let mut rows = (0..18)
            .map(|index| {
                let time = if index < 6 {
                    clock_ns(9, 30, index, 0)
                } else if index < 12 {
                    clock_ns(10, 30, index - 6, 0)
                } else {
                    clock_ns(13, 30, index - 12, 0)
                };
                row(
                    time,
                    index,
                    100.0 + index as f64 * 0.1,
                    index as f64 * 10.0,
                    index as f64,
                )
            })
            .collect::<Vec<_>>();
        rows[4].ask_volume1 = -1.0;
        let metrics = r88_intraday_metrics(&rows);
        assert!(metrics.exp_stick_quote_update_rate.is_nan());
        assert!(metrics.book_quote_dislocation.is_finite());
        assert!(metrics.exp_noise_median_mean_abs_return_ratio.is_finite());
        assert!(metrics.exp_rotation_segment_return_dispersion.is_finite());
    }
}
