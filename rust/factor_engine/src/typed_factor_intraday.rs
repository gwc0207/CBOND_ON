//! Pure-Rust kernels for typed intraday factor contracts.
//!
//! This module is called internally by the ordinary `compute_factor_frame`
//! dispatcher. Callers provide one physical score-day / one instrument at a
//! time; the functions retain exact time ordering, continuous-session,
//! counter-reset, invalid-book, and fail-closed contracts.
//!
//! `time_ns` is local time since midnight in nanoseconds.  The row arrays may
//! be unsorted: processing follows the Python ordering `(trade_time, seq)` with
//! input position as a stable tie breaker.  Session filtering first truncates
//! to microsecond precision, matching pandas `Series.dt.time`; consequently a
//! timestamp at `14:29:00.000000999` is still visible to the Python reference.
//! QED deliberately keeps the one possible 11:30 -> 13:00 adjacent interval
//! after filtering, because the reference Python QED kernel does not impose a
//! same-session interval gate.  LRD does impose that gate, matching its Python
//! `__session` logic.

use std::cmp::Ordering;

/// Python QED's `1e-8` at-quote classification tolerance.
pub const QED_AT_QUOTE_TOL: f64 = 1e-8;

const EPS: f64 = 1e-12;
const PRICE_TOL: f64 = 1e-12;
const MIN_ROWS: usize = 12;
const MIN_QUOTE_EVENTS: usize = 4;
const MIN_DIRECTION_EVENTS: usize = 4;
const MIN_REPRICE_EVENT_INTERVALS: usize = 3;
const MIN_MOVED_LEVELS: usize = 3;

const NS_PER_SECOND: i64 = 1_000_000_000;
const NS_PER_MICROSECOND: i64 = 1_000;
const NS_PER_MINUTE: i64 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: i64 = 60 * NS_PER_MINUTE;

const MORNING_START: i64 = clock_ns(9, 30, 0, 0);
const MORNING_END: i64 = clock_ns(11, 30, 0, 0);
const AFTERNOON_START: i64 = clock_ns(13, 0, 0, 0);
const CUTOFF: i64 = clock_ns(14, 29, 0, 0);

/// Build a local, same-day nanosecond clock value used by [`QuoteRow`] and
/// [`BookRow`].  Values outside one day are treated as nonphysical and are
/// excluded in the same way the Python timestamps are filtered out.
pub const fn clock_ns(hour: i64, minute: i64, second: i64, nanosecond: i64) -> i64 {
    hour * NS_PER_HOUR + minute * NS_PER_MINUTE + second * NS_PER_SECOND + nanosecond
}

/// One physical score-day quote snapshot for one bond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuoteRow {
    /// Local score-day time since midnight, in nanoseconds.
    pub time_ns: i64,
    /// Source snapshot sequence used after `time_ns` for deterministic order.
    pub seq: i64,
    pub last: f64,
    pub ask_price1: f64,
    pub bid_price1: f64,
    /// Cumulative number of trades; a negative increment fails closed.
    pub num_trades: f64,
}

/// One physical score-day L1--L5 order-book snapshot for one bond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BookRow {
    /// Local score-day time since midnight, in nanoseconds.
    pub time_ns: i64,
    /// Source snapshot sequence used after `time_ns` for deterministic order.
    pub seq: i64,
    pub ask_price: [f64; 5],
    pub bid_price: [f64; 5],
    pub ask_volume: [f64; 5],
    pub bid_volume: [f64; 5],
}

/// The three QED columns admitted to the live-23 pack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QedTypedFactorMetrics {
    pub prior_quote_location_dispersion: f64,
    pub prior_quote_tail_penetration: f64,
    pub prior_quote_lag2_agreement: f64,
}

impl QedTypedFactorMetrics {
    fn nan() -> Self {
        Self {
            prior_quote_location_dispersion: f64::NAN,
            prior_quote_tail_penetration: f64::NAN,
            prior_quote_lag2_agreement: f64::NAN,
        }
    }

    fn finite_or_nan(mut self) -> Self {
        if !self.prior_quote_location_dispersion.is_finite() {
            self.prior_quote_location_dispersion = f64::NAN;
        }
        if !self.prior_quote_tail_penetration.is_finite() {
            self.prior_quote_tail_penetration = f64::NAN;
        }
        if !self.prior_quote_lag2_agreement.is_finite() {
            self.prior_quote_lag2_agreement = f64::NAN;
        }
        self
    }
}

pub(crate) fn session_label(time_ns: i64) -> Option<u8> {
    // pandas exposes `Timestamp` values through `Series.dt.time` in
    // microsecond resolution.  Its comparison against `datetime.time(14, 29)`
    // therefore retains sub-microsecond timestamps through
    // `14:29:00.000000999`.  Use Euclidean division so nonphysical negative
    // clocks remain outside the session rather than being rounded toward zero.
    let python_time_ns = time_ns.div_euclid(NS_PER_MICROSECOND) * NS_PER_MICROSECOND;
    if (MORNING_START..=MORNING_END).contains(&python_time_ns) {
        Some(0)
    } else if (AFTERNOON_START..=CUTOFF).contains(&python_time_ns) {
        Some(1)
    } else {
        None
    }
}

fn ordering_key<T>(
    left: &(usize, T),
    right: &(usize, T),
    time: impl Fn(&T) -> i64,
    seq: impl Fn(&T) -> i64,
) -> Ordering {
    time(&left.1)
        .cmp(&time(&right.1))
        .then_with(|| seq(&left.1).cmp(&seq(&right.1)))
        .then_with(|| left.0.cmp(&right.0))
}

fn ordered_quote_rows(rows: &[QuoteRow]) -> Vec<QuoteRow> {
    let mut kept: Vec<(usize, QuoteRow)> = rows
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, row)| session_label(row.time_ns).is_some())
        .collect();
    kept.sort_by(|left, right| ordering_key(left, right, |row| row.time_ns, |row| row.seq));
    kept.into_iter().map(|(_, row)| row).collect()
}

fn ordered_book_rows(rows: &[BookRow]) -> Vec<(BookRow, u8)> {
    let mut kept: Vec<(usize, BookRow, u8)> = rows
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, row)| session_label(row.time_ns).map(|session| (index, row, session)))
        .collect();
    kept.sort_by(|left, right| {
        left.1
            .time_ns
            .cmp(&right.1.time_ns)
            .then_with(|| left.1.seq.cmp(&right.1.seq))
            .then_with(|| left.0.cmp(&right.0))
    });
    kept.into_iter()
        .map(|(_, row, session)| (row, session))
        .collect()
}

fn valid_quote_row(row: &QuoteRow) -> bool {
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

/// Pure implementation of the selected QED outputs.
///
/// This is deliberately per `(score_day, code)` group.  It mirrors
/// `factor_mining_quote_execution_dynamics_v1`: only the 09:30--11:30 and
/// 13:00--14:29 physical windows remain; `num_trades` must be monotone across
/// the filtered rows; locations use the strictly prior L1 book; and a failed
/// input contract returns NaN rather than a substitute value.
pub fn quote_execution_typed_factor(rows: &[QuoteRow]) -> QedTypedFactorMetrics {
    let rows = ordered_quote_rows(rows);
    if rows.len() < MIN_ROWS
        || rows
            .windows(2)
            .any(|pair| pair[0].time_ns == pair[1].time_ns)
        || rows.iter().any(|row| !valid_quote_row(row))
    {
        return QedTypedFactorMetrics::nan();
    }

    let mut locations = Vec::with_capacity(rows.len().saturating_sub(1));
    let mut weights = Vec::with_capacity(rows.len().saturating_sub(1));
    for pair in rows.windows(2) {
        let prior = pair[0];
        let current = pair[1];
        let increment = current.num_trades - prior.num_trades;
        if !increment.is_finite() || increment < 0.0 {
            return QedTypedFactorMetrics::nan();
        }
        let prior_spread = prior.ask_price1 - prior.bid_price1;
        if increment > 0.0 && prior_spread > EPS {
            let prior_midpoint = (prior.ask_price1 + prior.bid_price1) / 2.0;
            let location = (current.last - prior_midpoint) / (prior_spread / 2.0);
            if !location.is_finite() || !increment.is_finite() || increment <= EPS {
                return QedTypedFactorMetrics::nan();
            }
            locations.push(location);
            weights.push(increment);
        }
    }
    if locations.len() < MIN_QUOTE_EVENTS {
        return QedTypedFactorMetrics::nan();
    }

    // Python's _location_metrics can fail independently of direction metrics,
    // so retain that separation here instead of returning a blanket NaN.
    let total_weight: f64 = weights.iter().sum();
    let (dispersion, tail_penetration) = if !total_weight.is_finite() || total_weight <= EPS {
        (f64::NAN, f64::NAN)
    } else {
        let center = locations
            .iter()
            .zip(weights.iter())
            .map(|(location, weight)| location * weight)
            .sum::<f64>()
            / total_weight;
        let variance = locations
            .iter()
            .zip(weights.iter())
            .map(|(location, weight)| weight * (location - center).powi(2))
            .sum::<f64>()
            / total_weight;
        if !variance.is_finite() || variance < 0.0 {
            (f64::NAN, f64::NAN)
        } else {
            let tail = locations
                .iter()
                .zip(weights.iter())
                .map(|(location, weight)| weight * (location.abs() - 1.0).max(0.0).ln_1p())
                .sum::<f64>()
                / total_weight;
            (variance.max(0.0).sqrt(), tail)
        }
    };

    let directions: Vec<f64> = locations
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
        .collect();
    let lag2_agreement = if directions.len() < MIN_DIRECTION_EVENTS {
        f64::NAN
    } else {
        directions
            .iter()
            .skip(2)
            .zip(directions.iter())
            .map(|(current, lagged)| current * lagged)
            .sum::<f64>()
            / (directions.len() - 2) as f64
    };

    QedTypedFactorMetrics {
        prior_quote_location_dispersion: dispersion,
        prior_quote_tail_penetration: tail_penetration,
        prior_quote_lag2_agreement: lag2_agreement,
    }
    .finite_or_nan()
}

fn valid_book_row(row: &BookRow) -> bool {
    let finite_prices = row
        .ask_price
        .iter()
        .chain(row.bid_price.iter())
        .all(|value| value.is_finite() && *value > 0.0);
    let finite_depths = row
        .ask_volume
        .iter()
        .chain(row.bid_volume.iter())
        .all(|value| value.is_finite() && *value >= 0.0);
    finite_prices
        && finite_depths
        && row.ask_price[0] >= row.bid_price[0]
        && row.ask_price.windows(2).all(|pair| pair[1] >= pair[0])
        && row.bid_price.windows(2).all(|pair| pair[1] <= pair[0])
}

fn all_ladder_log_moves_finite(rows: &[(BookRow, u8)]) -> bool {
    // Python `_ladder_directions` materializes the entire `np.log(prices[1:]
    // / prices[:-1])` matrix *before* it applies `same_session`.  Thus even a
    // lunch-spanning ratio that overflows or underflows invalidates the whole
    // instrument, rather than merely suppressing that one interval.
    rows.windows(2).all(|pair| {
        let (prior, _) = pair[0];
        let (current, _) = pair[1];
        prior
            .bid_price
            .iter()
            .zip(current.bid_price.iter())
            .chain(prior.ask_price.iter().zip(current.ask_price.iter()))
            .all(|(prior_price, current_price)| (*current_price / *prior_price).ln().is_finite())
    })
}

fn ladder_direction(prior: &[f64; 5], current: &[f64; 5], same_session: bool) -> f64 {
    if !same_session {
        return 0.0;
    }
    let mut positive = 0usize;
    let mut negative = 0usize;
    for level in 0..5 {
        let log_move = (current[level] / prior[level]).ln();
        if !log_move.is_finite() {
            return 0.0;
        }
        if log_move > PRICE_TOL {
            positive += 1;
        } else if log_move < -PRICE_TOL {
            negative += 1;
        }
    }
    if positive >= MIN_MOVED_LEVELS && positive > negative {
        1.0
    } else if negative >= MIN_MOVED_LEVELS && negative > positive {
        -1.0
    } else {
        0.0
    }
}

/// Pure implementation of `lrd_cross_side_reprice_symmetry`.
///
/// It retains the stricter LRD session boundary: an interval spanning the
/// midday break is never a repricing event even when both ladders moved.
/// Invalid L1--L5 prices *or depths* fail closed, because Python's `_book`
/// validates the complete book before computing this price-only output.
pub fn lrd_cross_side_reprice_symmetry(rows: &[BookRow]) -> f64 {
    let rows = ordered_book_rows(rows);
    if rows.len() < MIN_ROWS
        || rows.iter().any(|(row, _)| !valid_book_row(row))
        || !all_ladder_log_moves_finite(&rows)
    {
        return f64::NAN;
    }

    let mut joint_products = Vec::with_capacity(rows.len().saturating_sub(1));
    for pair in rows.windows(2) {
        let (prior, prior_session) = pair[0];
        let (current, current_session) = pair[1];
        let same_session = prior_session == current_session;
        let bid_direction = ladder_direction(&prior.bid_price, &current.bid_price, same_session);
        let ask_direction = ladder_direction(&prior.ask_price, &current.ask_price, same_session);
        if bid_direction != 0.0 && ask_direction != 0.0 {
            joint_products.push(bid_direction * ask_direction);
        }
    }
    if joint_products.len() < MIN_REPRICE_EVENT_INTERVALS {
        return f64::NAN;
    }
    let result = joint_products.iter().sum::<f64>() / joint_products.len() as f64;
    if result.is_finite() {
        result
    } else {
        f64::NAN
    }
}

/// `rdm_joint_reprice_depth_retention`: for intervals in which both ladders
/// make validated reprices, average `min(total_depth_t, total_depth_t-1) /
/// total_depth_t-1`.  It deliberately shares the complete-book, session, and
/// global log-move validation contract with the LRD signals.
pub fn rdm_joint_reprice_depth_retention(rows: &[BookRow]) -> f64 {
    let rows = ordered_book_rows(rows);
    if rows.len() < MIN_ROWS
        || rows.iter().any(|(row, _)| !valid_book_row(row))
        || !all_ladder_log_moves_finite(&rows)
    {
        return f64::NAN;
    }
    let mut retention = Vec::with_capacity(rows.len().saturating_sub(1));
    for pair in rows.windows(2) {
        let (prior, prior_session) = pair[0];
        let (current, current_session) = pair[1];
        let same_session = prior_session == current_session;
        let bid_direction = ladder_direction(&prior.bid_price, &current.bid_price, same_session);
        let ask_direction = ladder_direction(&prior.ask_price, &current.ask_price, same_session);
        if bid_direction == 0.0 || ask_direction == 0.0 {
            continue;
        }
        let previous_depth =
            prior.bid_volume.iter().sum::<f64>() + prior.ask_volume.iter().sum::<f64>();
        let current_depth =
            current.bid_volume.iter().sum::<f64>() + current.ask_volume.iter().sum::<f64>();
        if !previous_depth.is_finite()
            || !current_depth.is_finite()
            || previous_depth <= EPS
            || current_depth < 0.0
        {
            return f64::NAN;
        }
        let value = previous_depth.min(current_depth) / previous_depth;
        if !value.is_finite() {
            return f64::NAN;
        }
        retention.push(value);
    }
    if retention.len() < MIN_REPRICE_EVENT_INTERVALS {
        return f64::NAN;
    }
    let value = retention.iter().sum::<f64>() / retention.len() as f64;
    if value.is_finite() {
        value
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quote_row(time_ns: i64, seq: i64, last: f64, num_trades: f64) -> QuoteRow {
        QuoteRow {
            time_ns,
            seq,
            last,
            ask_price1: 101.0,
            bid_price1: 99.0,
            num_trades,
        }
    }

    fn book_row(time_ns: i64, seq: i64, bid1: f64, ask1: f64) -> BookRow {
        let mut bid_price = [0.0; 5];
        let mut ask_price = [0.0; 5];
        for level in 0..5 {
            bid_price[level] = bid1 - level as f64 * 0.1;
            ask_price[level] = ask1 + level as f64 * 0.1;
        }
        BookRow {
            time_ns,
            seq,
            ask_price,
            bid_price,
            ask_volume: [10.0; 5],
            bid_volume: [10.0; 5],
        }
    }

    fn flat_book_row(time_ns: i64, seq: i64, bid: f64, ask: f64) -> BookRow {
        BookRow {
            time_ns,
            seq,
            ask_price: [ask; 5],
            bid_price: [bid; 5],
            ask_volume: [10.0; 5],
            bid_volume: [10.0; 5],
        }
    }

    fn assert_close(left: f64, right: f64) {
        assert!(left.is_finite(), "left must be finite: {left:?}");
        assert!(right.is_finite(), "right must be finite: {right:?}");
        assert!(
            (left - right).abs() <= 1e-12,
            "left={left:.16e}, right={right:.16e}"
        );
    }

    #[test]
    fn qed_preserves_time_seq_order_and_at_quote_tolerance() {
        let locations = [
            1.0 - QED_AT_QUOTE_TOL,
            -1.0 + QED_AT_QUOTE_TOL,
            0.0,
            2.0,
            -2.0,
            0.5,
            1.5,
            -1.5,
            0.0,
            1.0,
            -1.0,
        ];
        let mut rows = vec![quote_row(clock_ns(9, 30, 0, 0), 9, 100.0, 0.0)];
        for (index, location) in locations.iter().copied().enumerate() {
            rows.push(quote_row(
                clock_ns(9, 30, (index + 1) as i64, 0),
                (50 - index) as i64,
                100.0 + location,
                (index + 1) as f64,
            ));
        }
        let ordered = quote_execution_typed_factor(&rows);
        rows.reverse();
        let reversed = quote_execution_typed_factor(&rows);
        assert_close(
            ordered.prior_quote_location_dispersion,
            reversed.prior_quote_location_dispersion,
        );
        assert_close(
            ordered.prior_quote_tail_penetration,
            reversed.prior_quote_tail_penetration,
        );
        assert_close(
            ordered.prior_quote_lag2_agreement,
            reversed.prior_quote_lag2_agreement,
        );
        assert_close(ordered.prior_quote_lag2_agreement, 1.0);

        let mean = locations.iter().sum::<f64>() / locations.len() as f64;
        let expected_sd = (locations
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / locations.len() as f64)
            .sqrt();
        let expected_tail = locations
            .iter()
            .map(|value| (value.abs() - 1.0).max(0.0).ln_1p())
            .sum::<f64>()
            / locations.len() as f64;
        assert_close(ordered.prior_quote_location_dispersion, expected_sd);
        assert_close(ordered.prior_quote_tail_penetration, expected_tail);
    }

    #[test]
    fn qed_filters_lunch_rows_but_matches_reference_cross_lunch_adjacency() {
        let mut rows = Vec::new();
        for index in 0..6 {
            rows.push(quote_row(
                clock_ns(9, 30, index, 0),
                index,
                100.0,
                index as f64,
            ));
        }
        // Python QED filters the 12:00 observation, then treats the first
        // afternoon snapshot as adjacent to the 11:30 snapshot.
        rows.push(quote_row(clock_ns(12, 0, 0, 0), 99, 10_000.0, 999.0));
        for index in 0..6 {
            rows.push(quote_row(
                clock_ns(13, 0, index, 0),
                100 + index,
                if index == 0 { 110.0 } else { 100.0 },
                (index + 6) as f64,
            ));
        }
        let result = quote_execution_typed_factor(&rows);
        assert!(result.prior_quote_location_dispersion.is_finite());
        assert!(result.prior_quote_tail_penetration > 0.0);
        // Only the lunch-crossing interval is directional here, so the
        // separately gated lag-2 output correctly remains missing.
        assert!(result.prior_quote_lag2_agreement.is_nan());
    }

    #[test]
    fn qed_counter_reset_fails_closed() {
        let mut rows = (0..12)
            .map(|index| quote_row(clock_ns(9, 30, index, 0), index, 101.0, index as f64))
            .collect::<Vec<_>>();
        rows[8].num_trades = 1.0;
        let result = quote_execution_typed_factor(&rows);
        assert!(result.prior_quote_location_dispersion.is_nan());
        assert!(result.prior_quote_tail_penetration.is_nan());
        assert!(result.prior_quote_lag2_agreement.is_nan());
    }

    #[test]
    fn lrd_sorts_by_time_seq_and_excludes_midday_crossing() {
        let mut rows = Vec::new();
        for index in 0..6 {
            rows.push(book_row(
                clock_ns(11, 20, index, 0),
                100 - index,
                90.0 + index as f64,
                110.0 + index as f64,
            ));
        }
        // The lunch-crossing interval has opposed bid/ask moves.  Python LRD
        // gives it direction zero because its session labels differ.
        for index in 0..6 {
            rows.push(book_row(
                clock_ns(13, 0, index, 0),
                200 - index,
                80.0 + index as f64,
                120.0 + index as f64,
            ));
        }
        let expected = lrd_cross_side_reprice_symmetry(&rows);
        rows.reverse();
        let reversed = lrd_cross_side_reprice_symmetry(&rows);
        assert_close(expected, 1.0);
        assert_close(reversed, expected);
    }

    #[test]
    fn lrd_invalid_depth_fails_closed_even_for_price_only_output() {
        let mut rows = (0..12)
            .map(|index| {
                book_row(
                    clock_ns(9, 30, index, 0),
                    index,
                    90.0 + index as f64,
                    110.0 + index as f64,
                )
            })
            .collect::<Vec<_>>();
        rows[3].ask_volume[2] = -0.1;
        assert!(lrd_cross_side_reprice_symmetry(&rows).is_nan());
        assert!(rdm_joint_reprice_depth_retention(&rows).is_nan());
    }

    #[test]
    fn lrd_cross_lunch_nonfinite_log_fails_closed_globally() {
        let mut rows = Vec::new();
        for index in 0..6 {
            let scale = 1.0 + 0.01 * index as f64;
            rows.push(flat_book_row(
                clock_ns(9, 30, index, 0),
                index,
                1.0e-308 * scale,
                2.0e-308 * scale,
            ));
        }
        for index in 0..6 {
            let scale = 1.0 + 0.01 * index as f64;
            rows.push(flat_book_row(
                clock_ns(13, 0, index, 0),
                100 + index,
                8.0e307 * scale,
                9.0e307 * scale,
            ));
        }

        // Every row is valid and the morning contains enough symmetric
        // reprices.  Python nonetheless returns NaN because the intervening
        // 11:30 -> 13:00 log-ratio is non-finite before its session mask.
        assert!(lrd_cross_side_reprice_symmetry(&rows).is_nan());
        assert!(rdm_joint_reprice_depth_retention(&rows).is_nan());
    }

    #[test]
    fn rdm_joint_depth_retention_uses_only_joint_reprice_intervals() {
        let mut rows = Vec::new();
        for index in 0..12 {
            let mut row = book_row(
                clock_ns(9, 30, index, 0),
                index,
                90.0 + index as f64,
                110.0 + index as f64,
            );
            row.ask_volume = [10.0; 5];
            row.bid_volume = [20.0 - index as f64; 5];
            rows.push(row);
        }
        // Each adjacent pair reprices both ladders.  Total depth
        // stays positive and declines by five across each side, so the literal
        // previous-depth denominator is testable independently of the LRD
        // direction sign.
        let expected = (1..12)
            .map(|index| (150.0 - 5.0 * index as f64) / (150.0 - 5.0 * (index - 1) as f64))
            .sum::<f64>()
            / 11.0;
        assert_close(rdm_joint_reprice_depth_retention(&rows), expected);
    }

    #[test]
    fn submicrosecond_cutoff_matches_pandas_dt_time_for_qed_and_lrd() {
        assert_eq!(session_label(clock_ns(14, 29, 0, 999)), Some(1));
        assert_eq!(session_label(clock_ns(14, 29, 0, 1_000)), None);
        assert_eq!(session_label(clock_ns(11, 30, 0, 999)), Some(0));
        assert_eq!(session_label(clock_ns(11, 30, 0, 1_000)), None);

        let mut quote_rows = (0..11)
            .map(|index| quote_row(clock_ns(9, 30, index, 0), index, 101.0, index as f64))
            .collect::<Vec<_>>();
        quote_rows.push(quote_row(clock_ns(14, 29, 0, 999), 99, 101.0, 11.0));
        let qed = quote_execution_typed_factor(&quote_rows);
        assert_close(qed.prior_quote_location_dispersion, 0.0);
        assert_close(qed.prior_quote_tail_penetration, 0.0);
        assert_close(qed.prior_quote_lag2_agreement, 1.0);

        let mut book_rows = (0..11)
            .map(|index| {
                book_row(
                    clock_ns(9, 30, index, 0),
                    index,
                    90.0 + index as f64,
                    110.0 + index as f64,
                )
            })
            .collect::<Vec<_>>();
        book_rows.push(book_row(clock_ns(14, 29, 0, 999), 99, 101.0, 121.0));
        assert_close(lrd_cross_side_reprice_symmetry(&book_rows), 1.0);
    }
}
