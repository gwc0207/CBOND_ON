#[path = "../src/typed_factor_daily_rank_state.rs"]
mod typed_factor_daily_rank_state;

use chrono::{Duration, NaiveDate};
use typed_factor_daily_rank_state::{
    prcn_return_capacity_rank_corr60, ydpt_yield_fall_return_beta60, TypedFactorRankStateBaseRow,
    TypedFactorRankStateContext, TypedFactorRankStateError, TypedFactorRankStatePriceRow,
};

fn day(offset: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 2)
        .expect("date")
        .checked_add_signed(Duration::days(offset))
        .expect("offset")
}

fn push_rank_rows(ctx: &mut TypedFactorRankStateContext, offset: i64) {
    // A, B and C reverse their market ordering every day.  A's return and
    // capacity ranks move together, so the terminal 60-session correlation is
    // exactly one after pandas-style daily percentile ranking.
    let reversed = offset % 2 != 0;
    for (code, return_multiplier, capacity_multiplier) in [
        (
            "A.SH",
            if reversed { 1.10 } else { 0.90 },
            if reversed { 30.0 } else { 10.0 },
        ),
        ("B.SH", 1.00, 20.0),
        (
            "C.SH",
            if reversed { 0.90 } else { 1.10 },
            if reversed { 10.0 } else { 30.0 },
        ),
    ] {
        ctx.price_rows.push(TypedFactorRankStatePriceRow {
            trade_date: day(offset),
            code: code.to_string(),
            prev_close_price: 100.0,
            act_prev_close_price: 100.0,
            close_price: 100.0 * return_multiplier,
            amount: capacity_multiplier,
        });
        ctx.base_rows.push(TypedFactorRankStateBaseRow {
            trade_date: day(offset),
            code: code.to_string(),
            remain_size: 1.0,
            current_yield: 5.0,
        });
    }
}

#[test]
fn prcn_uses_global_calendar_and_daily_average_percentile_ranks() {
    let mut ctx = TypedFactorRankStateContext::new(day(62));
    for offset in 0..62 {
        push_rank_rows(&mut ctx, offset);
    }
    let value = prcn_return_capacity_rank_corr60(&ctx, "A.SH").expect("valid context");
    assert!((value - 1.0).abs() <= 1e-15, "value={value:?}");

    // A stale per-bond path is fail-closed even though B/C establish the
    // source anchor, matching the source kernel's terminal-anchor guard.
    ctx.price_rows
        .retain(|row| !(row.code == "A.SH" && row.trade_date == day(61)));
    ctx.base_rows
        .retain(|row| !(row.code == "A.SH" && row.trade_date == day(61)));
    assert!(prcn_return_capacity_rank_corr60(&ctx, "A.SH")
        .expect("valid context")
        .is_nan());
}

#[test]
fn ydpt_requires_complete_terminal_61_rows_and_uses_negative_yield_changes() {
    let mut ctx = TypedFactorRankStateContext::new(day(61));
    let mut yield_value = 5.0;
    for offset in 0..61 {
        if offset > 0 {
            yield_value -= 0.001 * (1.0 + (offset % 5) as f64);
        }
        let change = if offset == 0 {
            0.0
        } else {
            -0.001 * (1.0 + (offset % 5) as f64)
        };
        ctx.price_rows.push(TypedFactorRankStatePriceRow {
            trade_date: day(offset),
            code: "A.SH".to_string(),
            prev_close_price: 100.0,
            act_prev_close_price: 100.0,
            close_price: 100.0 * (2.5 * change).exp(),
            amount: 10.0,
        });
        ctx.base_rows.push(TypedFactorRankStateBaseRow {
            trade_date: day(offset),
            code: "A.SH".to_string(),
            remain_size: 1.0,
            current_yield: yield_value,
        });
    }
    let value = ydpt_yield_fall_return_beta60(&ctx, "A.SH").expect("valid context");
    // The fixture constructs returns through exp/log, so this unit-level
    // mathematical assertion permits normal IEEE rounding.  The later PyO3
    // kernel bridge will use the same NumPy reduction primitive as Python
    // before it is eligible for exact parity promotion.
    assert!((value - 2.5).abs() <= 1e-10, "value={value:?}");

    ctx.price_rows.last_mut().expect("terminal row").close_price = f64::NAN;
    assert!(ydpt_yield_fall_return_beta60(&ctx, "A.SH")
        .expect("valid context")
        .is_nan());
}

#[test]
fn duplicate_any_strict_prior_source_key_fails_the_entire_context_closed() {
    let mut ctx = TypedFactorRankStateContext::new(day(2));
    push_rank_rows(&mut ctx, 0);
    ctx.price_rows.push(ctx.price_rows[0].clone());
    let error = prcn_return_capacity_rank_corr60(&ctx, "A.SH").expect_err("duplicate must fail");
    assert!(matches!(
        error,
        TypedFactorRankStateError::DuplicateStrictPriorDate { .. }
    ));
}
