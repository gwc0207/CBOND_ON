#[allow(dead_code)]
#[path = "../src/typed_factor_daily_paths.rs"]
mod typed_factor_daily_paths;

use chrono::{Duration, NaiveDate};
use typed_factor_daily_paths::{
    bstk_tail_cocrash_residual20, dohw_intraday_sign_range_asymmetry60, dohw_mean_wick_asymmetry60,
    dret_drawup_drawdown_asym, TypedFactorDailyPathContext, TypedFactorDailyPathPriceRow,
    TypedFactorDailyPathsError, TypedFactorDailyPriceAnchorRow, TypedFactorDailyTrackingBaseRow,
    TypedFactorDailyTrackingContext,
};

const CODE: &str = "110001.SH";
const OTHER: &str = "110002.SH";

fn day(offset: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap() + Duration::days(offset)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(actual.is_finite(), "expected finite value, got {actual:?}");
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17}, expected={expected:.17}, tolerance={tolerance:.3e}"
    );
}

fn path_row(offset: i64, code: &str, positive: bool) -> TypedFactorDailyPathPriceRow {
    let (close, high, low) = if positive {
        (102.0, 108.0, 96.0)
    } else {
        (98.0, 104.0, 92.0)
    };
    TypedFactorDailyPathPriceRow {
        trade_date: day(offset),
        code: code.to_string(),
        prev_close_price: 100.0,
        act_prev_close_price: 100.0,
        open_price: 100.0,
        high_price: high,
        low_price: low,
        close_price: close,
    }
}

fn ohlc_context() -> TypedFactorDailyPathContext {
    let mut ctx = TypedFactorDailyPathContext::new(day(60));
    let mut main = Vec::new();
    let mut other = Vec::new();
    for offset in 0..60_i64 {
        main.push(path_row(offset, CODE, offset % 2 == 0));
        other.push(path_row(offset, OTHER, offset % 2 != 0));
    }
    // A score-day outlier must be ignored by the strict-prior filter.
    let mut score_day = path_row(60, CODE, true);
    score_day.high_price = 1.0e12;
    score_day.low_price = 1.0;
    score_day.close_price = 1.0e12;
    main.push(score_day);
    ctx.price_by_code.insert(CODE.to_string(), main);
    ctx.price_by_code.insert(OTHER.to_string(), other);
    ctx
}

#[test]
fn ohlc_signals_use_a_global_strict_prior_calendar_and_ignore_score_day_rows() {
    let ctx = ohlc_context();
    // Positive rows have wick +1/6 and negative rows have -1/6, so the
    // evenly split 60-session mean is exactly zero.  Directional ranges are
    // deliberately different and exercise the sign-conditioned mean.
    assert_close(dohw_mean_wick_asymmetry60(&ctx, CODE).unwrap(), 0.0, 1e-15);
    assert_close(
        dohw_intraday_sign_range_asymmetry60(&ctx, CODE).unwrap(),
        (108.0_f64 / 96.0).ln() - (104.0_f64 / 92.0).ln(),
        1e-15,
    );
}

#[test]
fn ohlc_terminal_and_duplicate_contracts_fail_closed() {
    let mut stale = ohlc_context();
    stale
        .price_by_code
        .get_mut(CODE)
        .unwrap()
        .retain(|row| row.trade_date != day(59));
    assert!(dohw_mean_wick_asymmetry60(&stale, CODE).unwrap().is_nan());
    assert!(dohw_intraday_sign_range_asymmetry60(&stale, CODE)
        .unwrap()
        .is_nan());

    let mut invalid_terminal = ohlc_context();
    let row = invalid_terminal
        .price_by_code
        .get_mut(CODE)
        .unwrap()
        .iter_mut()
        .find(|row| row.trade_date == day(59))
        .unwrap();
    row.high_price = 80.0;
    row.low_price = 90.0;
    assert!(dohw_mean_wick_asymmetry60(&invalid_terminal, CODE)
        .unwrap()
        .is_nan());

    let mut duplicate = ohlc_context();
    let duplicated = duplicate.price_by_code[CODE][0].clone();
    duplicate
        .price_by_code
        .get_mut(CODE)
        .unwrap()
        .push(duplicated);
    assert!(matches!(
        dohw_mean_wick_asymmetry60(&duplicate, CODE),
        Err(TypedFactorDailyPathsError::DuplicateStrictPriorDate { .. })
    ));
}

fn drawup_drawdown_reference(returns: &[f64]) -> f64 {
    let wealth: Vec<f64> = returns
        .iter()
        .scan(1.0_f64, |state, return_value| {
            *state *= 1.0 + *return_value;
            Some(*state)
        })
        .collect();
    let mut high = wealth[0];
    let mut low = wealth[0];
    let mut drawdown = f64::INFINITY;
    let mut drawup = f64::NEG_INFINITY;
    for value in wealth {
        high = high.max(value);
        low = low.min(value);
        drawdown = drawdown.min(value / high - 1.0);
        drawup = drawup.max(value / low - 1.0);
    }
    drawup / drawdown.abs()
}

#[test]
fn drawup_drawdown_uses_act_prev_close_and_a_finite_only_tail_without_terminal_gate() {
    let returns = [
        -0.10, 0.05, -0.02, 0.06, -0.03, 0.04, -0.01, 0.03, -0.02, 0.05, -0.04, 0.02, -0.01, 0.03,
        -0.02, 0.04, -0.03, 0.02, -0.01, 0.01,
    ];
    let mut ctx = TypedFactorDailyPathContext::new(day(21));
    let mut rows = Vec::new();
    for (index, return_value) in returns.iter().enumerate() {
        rows.push(TypedFactorDailyPathPriceRow {
            trade_date: day(index as i64),
            code: CODE.to_string(),
            // A wrong fallback would turn every row into an enormous return.
            prev_close_price: 1.0,
            act_prev_close_price: 100.0,
            open_price: f64::NAN,
            high_price: f64::NAN,
            low_price: f64::NAN,
            close_price: 100.0 * (1.0 + *return_value),
        });
    }
    // Both an invalid latest strict-prior price and a score-day outlier are
    // intentionally omitted by the finite-tail/strict-prior contracts.
    rows[19].close_price = f64::NAN;
    rows.push(TypedFactorDailyPathPriceRow {
        trade_date: day(21),
        code: CODE.to_string(),
        prev_close_price: 1.0,
        act_prev_close_price: 1.0,
        open_price: f64::NAN,
        high_price: f64::NAN,
        low_price: f64::NAN,
        close_price: 1.0e12,
    });
    ctx.price_by_code.insert(CODE.to_string(), rows);

    assert_close(
        dret_drawup_drawdown_asym(&ctx, CODE).unwrap(),
        drawup_drawdown_reference(&returns[..19]),
        1e-14,
    );
}

fn tracking_context(session_count: usize) -> TypedFactorDailyTrackingContext {
    let mut ctx = TypedFactorDailyTrackingContext::new(day(session_count as i64));
    let mut anchors = Vec::new();
    let mut base = Vec::new();
    let mut cb_previous = 100.0;
    let mut stock_previous = 10.0;
    for index in 0..session_count {
        let stock_return = if index + 1 == session_count {
            -0.10
        } else {
            0.01 + index as f64 * 0.001
        };
        let residual = if index + 1 == session_count {
            0.04
        } else {
            0.0
        };
        let cb_return = 2.0 * stock_return + residual;
        let cb_close = cb_previous * (1.0 + cb_return);
        let stock_close = stock_previous * (1.0 + stock_return);
        anchors.push(TypedFactorDailyPriceAnchorRow {
            trade_date: day(index as i64),
            code: CODE.to_string(),
        });
        base.push(TypedFactorDailyTrackingBaseRow {
            trade_date: day(index as i64),
            code: CODE.to_string(),
            cb_prev_close_price: cb_previous,
            cb_close_price: cb_close,
            stk_prev_close_price: stock_previous,
            stk_close_price: stock_close,
        });
        cb_previous = cb_close;
        stock_previous = stock_close;
    }
    ctx.price_anchor_by_code.insert(CODE.to_string(), anchors);
    ctx.base_by_code.insert(CODE.to_string(), base);
    ctx
}

#[test]
fn tail_cocrash_matches_linear_quantile_beta_residual_formula() {
    let ctx = tracking_context(41);
    // The beta window is exact 2x stock returns.  The final tail point is the
    // stock's unique crash and carries a +4% residual.  NumPy linear q=.2 on
    // the 20-point tail includes four lower-tail sessions, so mean=4%/4=1%.
    assert_close(
        bstk_tail_cocrash_residual20(&ctx, CODE).unwrap(),
        0.01,
        1e-12,
    );
}

#[test]
fn tail_cocrash_requires_current_anchor_and_adjacent_calendar_sessions() {
    let mut stale = tracking_context(41);
    stale
        .base_by_code
        .get_mut(CODE)
        .unwrap()
        .retain(|row| row.trade_date != day(40));
    assert!(bstk_tail_cocrash_residual20(&stale, CODE).unwrap().is_nan());

    let mut gapped = tracking_context(42);
    // Still 41 base rows after removal, so this specifically checks the
    // source-calendar ordinal rather than a simple row-count condition.
    gapped
        .base_by_code
        .get_mut(CODE)
        .unwrap()
        .retain(|row| row.trade_date != day(30));
    assert!(bstk_tail_cocrash_residual20(&gapped, CODE)
        .unwrap()
        .is_nan());

    let mut duplicate = tracking_context(41);
    let duplicated = duplicate.base_by_code[CODE][0].clone();
    duplicate
        .base_by_code
        .get_mut(CODE)
        .unwrap()
        .push(duplicated);
    assert!(matches!(
        bstk_tail_cocrash_residual20(&duplicate, CODE),
        Err(TypedFactorDailyPathsError::DuplicateStrictPriorDate { .. })
    ));
}
