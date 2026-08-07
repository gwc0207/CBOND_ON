//! Numerical primitives for typed factor kernels.
//!
//! These pure, fail-closed helpers are shared by the typed implementations and
//! preserve their explicit missing-data, ranking, and state-count contracts.

use std::cmp::Ordering;

/// Return whether a value is a finite IEEE-754 number.
#[inline]
pub(crate) fn finite(value: f64) -> bool {
    value.is_finite()
}

/// Keep finite observations in their original order.
///
/// This intentionally does not turn missing values into zero or carry values
/// across a missing observation.
pub(crate) fn finite_values(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .copied()
        .filter(|value| finite(*value))
        .collect()
}

/// Pairwise-drop non-finite observations while preserving the original order.
///
/// A length mismatch is a data-contract failure and returns `None`, rather
/// than silently truncating to the shorter input.
pub(crate) fn pair_drop_finite(left: &[f64], right: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    if left.len() != right.len() {
        return None;
    }
    let mut kept_left = Vec::with_capacity(left.len());
    let mut kept_right = Vec::with_capacity(right.len());
    for (&x, &y) in left.iter().zip(right.iter()) {
        if finite(x) && finite(y) {
            kept_left.push(x);
            kept_right.push(y);
        }
    }
    Some((kept_left, kept_right))
}

/// Sample standard deviation (`ddof=1`) of an already-complete numeric slice.
///
/// Missing/non-finite input and fewer than two observations fail closed to
/// `NaN`.  Callers that intentionally use a finite-only tail must call
/// [`finite_values`] explicitly first.
pub(crate) fn sample_std_ddof1(values: &[f64]) -> f64 {
    if values.len() < 2 || values.iter().any(|value| !finite(*value)) {
        return f64::NAN;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sum_sq = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>();
    let variance = sum_sq / (n - 1.0);
    if finite(variance) && variance >= 0.0 {
        variance.sqrt()
    } else {
        f64::NAN
    }
}

/// Pearson correlation after strict pairwise finite-value dropping.
///
/// This matches the factor convention of dropping a pair only when either
/// member is missing, then applying the requested minimum valid-pair guard.
pub(crate) fn pearson_pair_drop(left: &[f64], right: &[f64], min_pairs: usize) -> f64 {
    let Some((left, right)) = pair_drop_finite(left, right) else {
        return f64::NAN;
    };
    if left.len() < min_pairs || left.is_empty() {
        return f64::NAN;
    }
    let n = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / n;
    let right_mean = right.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut left_sum_sq = 0.0;
    let mut right_sum_sq = 0.0;
    for (&x, &y) in left.iter().zip(right.iter()) {
        let centered_left = x - left_mean;
        let centered_right = y - right_mean;
        numerator += centered_left * centered_right;
        left_sum_sq += centered_left * centered_left;
        right_sum_sq += centered_right * centered_right;
    }
    let denominator = (left_sum_sq * right_sum_sq).sqrt();
    if !finite(denominator) || denominator <= 0.0 {
        return f64::NAN;
    }
    let value = numerator / denominator;
    if finite(value) {
        value
    } else {
        f64::NAN
    }
}

/// Reproduce `pandas.Series.rank(method="average", pct=True)` for f64 data.
///
/// `NaN` retains a `NaN` rank.  Finite values and infinities are rankable, as
/// in pandas' default numeric rank semantics; `-0.0` and `0.0` are one tie.
pub(crate) fn average_pct_rank(values: &[f64]) -> Vec<f64> {
    let mut output = vec![f64::NAN; values.len()];
    let mut ordered: Vec<(f64, usize)> = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| (!value.is_nan()).then_some((value, index)))
        .collect();
    if ordered.is_empty() {
        return output;
    }
    ordered.sort_by(|(left, _), (right, _)| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let denominator = ordered.len() as f64;
    let mut first = 0usize;
    while first < ordered.len() {
        let value = ordered[first].0;
        let mut end = first + 1;
        while end < ordered.len() && ordered[end].0 == value {
            end += 1;
        }
        // Pandas ranks are one-based.  `end` is the inclusive rank of the
        // last tied observation because it is the first exclusive index.
        let average_rank = ((first + 1 + end) as f64) * 0.5;
        let percentile_rank = average_rank / denominator;
        for &(_, original_index) in &ordered[first..end] {
            output[original_index] = percentile_rank;
        }
        first = end;
    }
    output
}

/// NumPy-style linear quantile for a complete finite slice.
///
/// This is the `method="linear"` interpolation convention used by the live-23
/// tail-cocrash factor.  Invalid quantiles, missing values, and empty input
/// fail closed to `NaN`.
pub(crate) fn linear_quantile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty()
        || !finite(quantile)
        || !(0.0..=1.0).contains(&quantile)
        || values.iter().any(|value| !finite(*value))
    {
        return f64::NAN;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if ordered.len() == 1 {
        return ordered[0];
    }
    let position = quantile * ((ordered.len() - 1) as f64);
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    if lower_index == upper_index {
        return ordered[lower_index];
    }
    let fraction = position - (lower_index as f64);
    let value = ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction;
    if finite(value) {
        value
    } else {
        f64::NAN
    }
}

/// Encode a finite value into the Python live-23 negative/zero/positive state.
///
/// `-1` represents a missing/non-finite state; `0`, `1`, and `2` represent
/// negative, approximately-zero, and positive values respectively.
pub(crate) fn sign3(value: f64, epsilon: f64) -> i8 {
    if !finite(value) || !finite(epsilon) || epsilon < 0.0 {
        return -1;
    }
    if value < -epsilon {
        0
    } else if value.abs() <= epsilon {
        1
    } else {
        2
    }
}

/// Match NumPy's contiguous `float64` reduction order for the small dense
/// vectors emitted by the typed kernels.
///
/// A serial Rust accumulator is mathematically equivalent to `np.sum`, but
/// NumPy uses pairwise accumulation for contiguous vectors of eight or more
/// elements.  The 3x3 information kernels therefore differed from their
/// Python reference by one to four ULPs even though their states and counts
/// were identical.  Preserve NumPy's reduction order so a Rust-first live
/// route retains the existing factor values bit-for-bit.
#[inline]
fn numpy_contiguous_pairwise_sum(values: &[f64]) -> f64 {
    if values.len() < 8 {
        return values.iter().copied().sum();
    }

    // NumPy's small-vector branch starts eight independent accumulators,
    // reduces them in a balanced tree, then appends any tail serially.  The
    // information term has exactly nine elements, so treating it as a simple
    // left-to-right sequence (or even as adjacent pairs) is not sufficient.
    const PAIRWISE_BLOCK: usize = 128;
    if values.len() <= PAIRWISE_BLOCK {
        let mut accumulators = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let block_end = values.len() - values.len() % 8;
        for index in (8..block_end).step_by(8) {
            for offset in 0..8 {
                accumulators[offset] += values[index + offset];
            }
        }
        let mut sum = ((accumulators[0] + accumulators[1]) + (accumulators[2] + accumulators[3]))
            + ((accumulators[4] + accumulators[5]) + (accumulators[6] + accumulators[7]));
        for value in &values[block_end..] {
            sum += *value;
        }
        return sum;
    }

    let split = values.len() / 2;
    numpy_contiguous_pairwise_sum(&values[..split])
        + numpy_contiguous_pairwise_sum(&values[split..])
}

/// Normalized 3x3 sign-state mutual information with a Jeffreys pseudocount.
///
/// The terminal pair must be valid, matching the strict live-23 historical
/// information factors.  Missing pairs are excluded rather than imputed.
pub(crate) fn normalized_mutual_information_3state(
    left: &[f64],
    right: &[f64],
    epsilon: f64,
    min_observations: usize,
    pseudocount: f64,
) -> f64 {
    if left.len() != right.len()
        || left.is_empty()
        || !finite(epsilon)
        || epsilon < 0.0
        || !finite(pseudocount)
        || pseudocount < 0.0
    {
        return f64::NAN;
    }
    let left_states: Vec<i8> = left.iter().map(|value| sign3(*value, epsilon)).collect();
    let right_states: Vec<i8> = right.iter().map(|value| sign3(*value, epsilon)).collect();
    if left_states.last() == Some(&-1) || right_states.last() == Some(&-1) {
        return f64::NAN;
    }

    let mut counts = [[pseudocount; 3]; 3];
    let mut valid_count = 0usize;
    for (&left_state, &right_state) in left_states.iter().zip(right_states.iter()) {
        if left_state >= 0 && right_state >= 0 {
            counts[left_state as usize][right_state as usize] += 1.0;
            valid_count += 1;
        }
    }
    if valid_count < min_observations {
        return f64::NAN;
    }
    let total = counts.iter().flatten().sum::<f64>();
    if !finite(total) || total <= 0.0 {
        return f64::NAN;
    }
    let mut left_marginal = [0.0; 3];
    let mut right_marginal = [0.0; 3];
    for left_state in 0..3 {
        for right_state in 0..3 {
            let probability = counts[left_state][right_state] / total;
            left_marginal[left_state] += probability;
            right_marginal[right_state] += probability;
        }
    }
    let mut information_terms = [0.0; 9];
    for left_state in 0..3 {
        for right_state in 0..3 {
            let probability = counts[left_state][right_state] / total;
            let denominator = left_marginal[left_state] * right_marginal[right_state];
            if probability > 0.0 && denominator > 0.0 {
                information_terms[left_state * 3 + right_state] =
                    probability * (probability / denominator).ln();
            }
        }
    }
    let value = numpy_contiguous_pairwise_sum(&information_terms) / 3.0_f64.ln();
    if finite(value) {
        value
    } else {
        f64::NAN
    }
}

/// Normalized entropy of a categorical transition path.
///
/// `transition_valid[i]` declares whether `states[i] -> states[i + 1]` is a
/// usable adjacent transition.  This lets callers preserve source-calendar
/// gaps exactly instead of treating two non-adjacent rows as neighbours.
/// A non-negative pseudocount supports both the 27x27 Jeffreys-smoothed
/// liquidity-state entropy and the unsmoothed 9x9 topology entropy.
pub(crate) fn normalized_transition_entropy(
    states: &[i16],
    transition_valid: &[bool],
    state_count: usize,
    min_transitions: usize,
    pseudocount: f64,
    require_terminal_state: bool,
    require_terminal_transition: bool,
) -> f64 {
    if states.len() < 2
        || transition_valid.len() != states.len() - 1
        || state_count < 2
        || !finite(pseudocount)
        || pseudocount < 0.0
    {
        return f64::NAN;
    }
    let terminal_valid = |state: i16| state >= 0 && (state as usize) < state_count;
    if require_terminal_state && !terminal_valid(*states.last().expect("non-empty states")) {
        return f64::NAN;
    }
    if require_terminal_transition && !transition_valid.last().copied().unwrap_or(false) {
        return f64::NAN;
    }
    let Some(matrix_len) = state_count.checked_mul(state_count) else {
        return f64::NAN;
    };
    let mut counts = vec![pseudocount; matrix_len];
    let mut valid_count = 0usize;
    for index in 0..transition_valid.len() {
        if !transition_valid[index] {
            continue;
        }
        let from = states[index];
        let to = states[index + 1];
        if !terminal_valid(from) || !terminal_valid(to) {
            return f64::NAN;
        }
        let cell = (from as usize) * state_count + (to as usize);
        counts[cell] += 1.0;
        valid_count += 1;
    }
    if valid_count < min_transitions {
        return f64::NAN;
    }
    let total = counts.iter().sum::<f64>();
    if !finite(total) || total <= 0.0 {
        return f64::NAN;
    }
    let normalizer = (matrix_len as f64).ln();
    if !finite(normalizer) || normalizer <= 0.0 {
        return f64::NAN;
    }
    let mut entropy = 0.0;
    for count in counts {
        if count <= 0.0 {
            continue;
        }
        let probability = count / total;
        entropy -= probability * probability.ln();
    }
    let value = entropy / normalizer;
    if finite(value) {
        value
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }

    #[test]
    fn finite_and_pair_drop_are_strict() {
        assert!(finite(1.0));
        assert!(!finite(f64::NAN));
        assert_eq!(
            finite_values(&[1.0, f64::NAN, f64::INFINITY, -2.0]),
            vec![1.0, -2.0]
        );

        let (left, right) = pair_drop_finite(
            &[1.0, f64::NAN, 3.0, f64::INFINITY, 5.0],
            &[10.0, 20.0, f64::NAN, 40.0, 50.0],
        )
        .expect("equal-length inputs");
        assert_eq!(left, vec![1.0, 5.0]);
        assert_eq!(right, vec![10.0, 50.0]);
        assert!(pair_drop_finite(&[1.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn sample_std_and_pearson_follow_fail_closed_contract() {
        assert_close(sample_std_ddof1(&[1.0, 2.0, 3.0]), 1.0, 1e-15);
        assert!(sample_std_ddof1(&[1.0]).is_nan());
        assert!(sample_std_ddof1(&[1.0, f64::NAN]).is_nan());

        assert_close(
            pearson_pair_drop(&[1.0, 2.0, f64::NAN, 4.0], &[3.0, 5.0, 7.0, 9.0], 3),
            1.0,
            1e-15,
        );
        assert!(pearson_pair_drop(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0], 3).is_nan());
        assert!(pearson_pair_drop(&[1.0], &[1.0, 2.0], 1).is_nan());
    }

    #[test]
    fn average_pct_rank_matches_average_tie_semantics() {
        let ranks = average_pct_rank(&[3.0, f64::NAN, 1.0, 1.0, 2.0]);
        assert_close(ranks[0], 1.0, 1e-15);
        assert!(ranks[1].is_nan());
        assert_close(ranks[2], 0.375, 1e-15);
        assert_close(ranks[3], 0.375, 1e-15);
        assert_close(ranks[4], 0.75, 1e-15);

        let infinity_ranks = average_pct_rank(&[f64::INFINITY, -f64::INFINITY, 0.0]);
        assert_close(infinity_ranks[0], 1.0, 1e-15);
        assert_close(infinity_ranks[1], 1.0 / 3.0, 1e-15);
        assert_close(infinity_ranks[2], 2.0 / 3.0, 1e-15);
    }

    #[test]
    fn linear_quantile_uses_numpy_linear_interpolation() {
        assert_close(linear_quantile(&[0.0, 10.0, 20.0, 30.0], 0.25), 7.5, 1e-15);
        assert_close(linear_quantile(&[5.0, 1.0, 3.0], 0.5), 3.0, 1e-15);
        assert!(linear_quantile(&[1.0, f64::NAN], 0.5).is_nan());
        assert!(linear_quantile(&[1.0], 1.1).is_nan());
    }

    #[test]
    fn sign3_and_nmi_respect_terminal_and_missing_guards() {
        assert_eq!(sign3(-1e-4, 1e-12), 0);
        assert_eq!(sign3(0.0, 1e-12), 1);
        assert_eq!(sign3(1e-4, 1e-12), 2);
        assert_eq!(sign3(f64::NAN, 1e-12), -1);

        let left = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
        let right = [-2.0, 0.0, 3.0, -4.0, 0.0, 5.0];
        let nmi = normalized_mutual_information_3state(&left, &right, 1e-12, 6, 0.5);
        assert!(nmi.is_finite() && nmi > 0.1 && nmi < 1.0);
        assert_close(
            nmi,
            normalized_mutual_information_3state(&right, &left, 1e-12, 6, 0.5),
            1e-15,
        );
        assert!(
            normalized_mutual_information_3state(&[1.0, f64::NAN], &[1.0, 2.0], 1e-12, 1, 0.5,)
                .is_nan()
        );
    }

    #[test]
    fn nmi_matches_numpy_pairwise_reduction_bits() {
        // This 3x3 table occurs in the real live50 LCC score-day input.  The
        // expected bits are the Python/NumPy result, not merely a tolerance
        // comparison: a serial accumulator is one ULP lower.
        let expected_counts = [[19usize, 0, 9], [0, 0, 0], [8, 0, 23]];
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (left_state, row) in expected_counts.iter().enumerate() {
            for (right_state, count) in row.iter().enumerate() {
                for _ in 0..*count {
                    left.push(left_state as f64 - 1.0);
                    right.push(right_state as f64 - 1.0);
                }
            }
        }
        // Keep the terminal observation valid, as the production kernel
        // requires; the final cell already encodes the positive/positive
        // state from the persisted Python reference.
        let value = normalized_mutual_information_3state(&left, &right, 1e-12, 45, 0.5);
        assert_eq!(value.to_bits(), 0x3fb6_e925_c10c_8af9);
    }

    #[test]
    fn transition_entropy_honours_calendar_and_terminal_masks() {
        let states = [0_i16, 1, 0, 1, 0];
        let valid = [true, true, true, true];
        assert_close(
            normalized_transition_entropy(&states, &valid, 2, 4, 0.0, true, true),
            0.5,
            1e-15,
        );

        let terminal_gap = [true, true, true, false];
        assert!(
            normalized_transition_entropy(&states, &terminal_gap, 2, 3, 0.0, true, true,).is_nan()
        );
        assert!(
            normalized_transition_entropy(&[0_i16, -1], &[true], 2, 1, 0.5, true, false,).is_nan()
        );
    }
}
