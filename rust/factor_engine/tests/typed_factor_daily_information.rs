#[allow(dead_code)]
#[path = "../src/typed_factor_daily.rs"]
mod typed_factor_daily;
#[path = "../src/typed_factor_daily_information.rs"]
mod typed_factor_daily_information;
#[allow(dead_code)]
#[path = "../src/typed_factor_math.rs"]
mod typed_factor_math;

use chrono::{Duration, NaiveDate};
use typed_factor_daily::{TypedFactorDailyContext, TypedFactorDailyPriceRow};
use typed_factor_daily_information::{
    compute_daily_information_signal, lcc_amount_trade_size_information60,
    lcc_volume_deal_information60, rjst_amount_joint_transition_entropy60,
    rlmi_return_deal_sign_mutual_information60, TypedFactorDailyInformationError,
};

const CODE: &str = "110001.SH";
const OTHER_CODE: &str = "110002.SH";

fn day(offset: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid date") + Duration::days(offset)
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "expected finite value, got {actual:?}");
    assert!(
        (actual - expected).abs() <= 1e-14,
        "actual={actual:.17}, expected={expected:.17}"
    );
}

fn synthetic_row(offset: i64, code: &str, code_phase: f64) -> TypedFactorDailyPriceRow {
    let i = offset as f64;
    let mut price = 100.0 + 7.0 * code_phase;
    // Rebuild the compounded close exactly as the Python golden generator.
    for position in 0..=offset {
        let p = position as f64;
        let daily_return = 0.012 * (0.73 * p + 0.31 * code_phase).sin();
        if position == offset {
            let previous = price;
            price *= daily_return.exp();
            return TypedFactorDailyPriceRow {
                trade_date: day(offset),
                code: code.to_string(),
                prev_close_price: previous,
                close_price: price,
                high_price: price + 2.0,
                low_price: price - 2.0,
                volume: 9_000.0 * (0.22 * (0.41 * i + 0.17 * code_phase).sin()).exp(),
                amount: 1_000_000.0
                    * (0.35 * (0.51 * i + 0.19 * code_phase).cos() + 0.09 * (0.91 * i).sin()).exp(),
                deal: 7_000.0
                    * (0.27 * (0.59 * i + 0.11 * code_phase).sin() + 0.07 * (0.37 * i).cos()).exp(),
            };
        }
        price *= daily_return.exp();
    }
    unreachable!("the inclusive loop returns at offset")
}

fn synthetic_history(
    code: &str,
    code_phase: f64,
    end_offset: i64,
) -> Vec<TypedFactorDailyPriceRow> {
    (0..=end_offset)
        .map(|offset| synthetic_row(offset, code, code_phase))
        .collect()
}

fn baseline_context() -> TypedFactorDailyContext {
    let mut ctx = TypedFactorDailyContext::new(day(65));
    ctx.price_by_code
        .insert(CODE.to_string(), synthetic_history(CODE, 1.0, 64));
    ctx.price_by_code.insert(
        OTHER_CODE.to_string(),
        synthetic_history(OTHER_CODE, 2.0, 64),
    );
    ctx
}

#[test]
fn four_outputs_match_python_golden_values() {
    let ctx = baseline_context();

    // Generated once from the unmodified Python reference kernels on the same
    // 65 strict-prior rows.  The values only depend on categorical state
    // counts, so this is also sensitive to the 60-session, pseudocount,
    // terminal, and exact-zero contracts.
    assert_close(
        lcc_amount_trade_size_information60(&ctx, CODE).expect("valid input"),
        0.09187866553035927,
    );
    assert_close(
        lcc_volume_deal_information60(&ctx, CODE).expect("valid input"),
        0.017626897157766767,
    );
    assert_close(
        rlmi_return_deal_sign_mutual_information60(&ctx, CODE).expect("valid input"),
        0.0206045005922979,
    );
    assert_close(
        rjst_amount_joint_transition_entropy60(&ctx, CODE).expect("valid input"),
        0.5318027980642955,
    );
}

#[test]
fn score_day_rows_are_excluded_from_all_four_signals() {
    let baseline = baseline_context();
    let mut contaminated = baseline.clone();
    for code in [CODE, OTHER_CODE] {
        contaminated
            .price_by_code
            .get_mut(code)
            .expect("seeded code")
            .push(TypedFactorDailyPriceRow {
                trade_date: day(65),
                code: code.to_string(),
                prev_close_price: 1.0,
                close_price: 9_999_999.0,
                high_price: 9_999_999.0,
                low_price: 1.0,
                volume: 9_999_999.0,
                amount: 9_999_999_999.0,
                deal: 9_999_999.0,
            });
    }
    let signals = [
        "lcc_amount_trade_size_information60",
        "lcc_volume_deal_information60",
        "rlmi_return_deal_sign_mutual_information60",
        "rjst_amount_joint_transition_entropy60",
    ];
    for signal in signals {
        let expected = compute_daily_information_signal(&baseline, CODE, signal).unwrap();
        let actual = compute_daily_information_signal(&contaminated, CODE, signal).unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits(), "signal={signal}");
    }
}

#[test]
fn stale_terminal_code_and_terminal_calendar_gap_fail_closed() {
    let mut stale = baseline_context();
    stale
        .price_by_code
        .get_mut(CODE)
        .expect("seeded code")
        .pop();
    for signal in [
        "lcc_amount_trade_size_information60",
        "lcc_volume_deal_information60",
        "rlmi_return_deal_sign_mutual_information60",
        "rjst_amount_joint_transition_entropy60",
    ] {
        assert!(
            compute_daily_information_signal(&stale, CODE, signal)
                .unwrap()
                .is_nan(),
            "stale signal={signal}"
        );
    }

    // The code still reaches the global anchor, but it has no immediately
    // preceding source session.  The terminal state/transition cannot bridge
    // that gap and must fail closed.
    let mut gapped = baseline_context();
    gapped
        .price_by_code
        .get_mut(CODE)
        .expect("seeded code")
        .remove(63);
    for signal in [
        "lcc_amount_trade_size_information60",
        "lcc_volume_deal_information60",
        "rlmi_return_deal_sign_mutual_information60",
        "rjst_amount_joint_transition_entropy60",
    ] {
        assert!(
            compute_daily_information_signal(&gapped, CODE, signal)
                .unwrap()
                .is_nan(),
            "gapped signal={signal}"
        );
    }
}

#[test]
fn terminal_deal_only_invalidates_deal_dependent_signals() {
    let mut ctx = baseline_context();
    ctx.price_by_code.get_mut(CODE).expect("seeded code")[64].deal = 0.0;

    assert!(lcc_amount_trade_size_information60(&ctx, CODE)
        .unwrap()
        .is_nan());
    assert!(lcc_volume_deal_information60(&ctx, CODE).unwrap().is_nan());
    assert!(rlmi_return_deal_sign_mutual_information60(&ctx, CODE)
        .unwrap()
        .is_nan());
    assert!(rjst_amount_joint_transition_entropy60(&ctx, CODE)
        .unwrap()
        .is_finite());
}

#[test]
fn duplicate_strict_prior_key_is_global_failure_and_dispatch_is_explicit() {
    let mut ctx = baseline_context();
    let duplicate = ctx.price_by_code[CODE][0].clone();
    ctx.price_by_code
        .get_mut(CODE)
        .expect("seeded code")
        .push(duplicate);
    assert!(matches!(
        lcc_amount_trade_size_information60(&ctx, OTHER_CODE),
        Err(TypedFactorDailyInformationError::DuplicateStrictPriorDate { .. })
    ));
    assert!(matches!(
        compute_daily_information_signal(&ctx, CODE, "not_a_typed_factor_signal"),
        Err(TypedFactorDailyInformationError::UnknownSignal(_))
    ));
}
