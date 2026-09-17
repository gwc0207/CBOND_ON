#[allow(dead_code)]
#[path = "../src/typed_factor_daily.rs"]
mod typed_factor_daily;

use chrono::{Duration, NaiveDate};
use typed_factor_daily::{
    base_debt_premium_floor_gap, base_duration_stockvol_interaction, base_stockvol_per_moneyness,
    base_trigger_progress_ratio, base_trigger_revision_gap, compute_p1_daily_signal,
    dliq_volume_return_corr20, dredemption_bondpremium_interaction, dredemption_premium_z20,
    dret_momentum_5, dret_volatility_20, drt_rebound_from_low20, dtwap_morning_slope20,
    rating_current_ordinal, TypedFactorDailyBaseRow, TypedFactorDailyContext,
    TypedFactorDailyError, TypedFactorDailyPriceRow, TypedFactorDailyTwapRow,
};

const CODE: &str = "110001.SH";

fn day(offset: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap() + Duration::days(offset)
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "expected finite value, got {actual:?}");
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual={actual:.17}, expected={expected:.17}"
    );
}

fn base_row(
    offset: i64,
    debt: f64,
    pure: f64,
    bond: f64,
    redemption: f64,
) -> TypedFactorDailyBaseRow {
    TypedFactorDailyBaseRow {
        trade_date: day(offset),
        code: CODE.to_string(),
        debt_puredebt_ratio: debt,
        puredebt_prem_ratio: pure,
        bond_prem_ratio: bond,
        redemption_prem_ratio: redemption,
        duration: f64::NAN,
        stock_volatility: f64::NAN,
        cb_close_price: f64::NAN,
        conv_value: f64::NAN,
        trigger_cum_days: f64::NAN,
        trigger_reach_days: f64::NAN,
        trigger_cum_days_revise: f64::NAN,
        rating: String::new(),
    }
}

fn price_row(offset: i64, close: f64, low: f64, volume: f64) -> TypedFactorDailyPriceRow {
    TypedFactorDailyPriceRow {
        trade_date: day(offset),
        code: CODE.to_string(),
        prev_close_price: 100.0,
        close_price: close,
        high_price: close + 2.0,
        low_price: low,
        volume,
        amount: 1_000.0,
        deal: 10.0,
    }
}

#[test]
fn p1_daily_metrics_match_closed_form_references() {
    let mut ctx = TypedFactorDailyContext::new(day(20));
    let mut base = Vec::new();
    let mut price = Vec::new();
    let mut twap = Vec::new();
    for index in 0..20_i64 {
        let value = (index + 1) as f64;
        base.push(base_row(
            index,
            0.30,
            0.10,
            0.10 + value * 0.01,
            value * 0.01,
        ));
        price.push(price_row(
            index,
            100.0 + value,
            if index == 3 { 80.0 } else { 95.0 + value },
            value.exp(),
        ));
        twap.push(TypedFactorDailyTwapRow {
            trade_date: day(index),
            code: CODE.to_string(),
            twap_0930_0935: 100.0,
            twap_0935_1000: 101.0,
        });
    }
    ctx.base_by_code.insert(CODE.to_string(), base);
    ctx.price_by_code.insert(CODE.to_string(), price);
    ctx.twap_by_code.insert(CODE.to_string(), twap);

    assert_close(base_debt_premium_floor_gap(&ctx, CODE).unwrap(), 0.20);
    assert_close(
        dredemption_bondpremium_interaction(&ctx, CODE).unwrap(),
        0.30 * 0.20,
    );
    assert_close(
        dredemption_premium_z20(&ctx, CODE).unwrap(),
        9.5 / 35.0_f64.sqrt(),
    );
    assert_close(
        dret_volatility_20(&ctx, CODE).unwrap(),
        0.01 * 35.0_f64.sqrt(),
    );
    assert_close(dliq_volume_return_corr20(&ctx, CODE).unwrap(), 1.0);
    assert_close(dtwap_morning_slope20(&ctx, CODE).unwrap(), 0.01);
    assert_close(
        drt_rebound_from_low20(&ctx, CODE).unwrap(),
        (120.0 - 80.0) / 80.0,
    );
}

#[test]
fn score_day_rows_are_excluded_and_missing_terminal_values_fail_closed() {
    let mut ctx = TypedFactorDailyContext::new(day(12));
    let mut base = Vec::new();
    let mut price = Vec::new();
    let mut twap = Vec::new();
    for index in 0..12_i64 {
        base.push(base_row(index, 0.30, 0.10, 0.20, 0.05));
        price.push(price_row(index, 101.0, 90.0, 100.0 + index as f64));
        twap.push(TypedFactorDailyTwapRow {
            trade_date: day(index),
            code: CODE.to_string(),
            twap_0930_0935: 100.0,
            twap_0935_1000: 101.0,
        });
    }
    // These score-day values must not leak into strict-prior calculations.
    base.push(base_row(12, 999.0, 0.0, 999.0, 999.0));
    price.push(price_row(12, 1_000.0, 1.0, 1e12));
    twap.push(TypedFactorDailyTwapRow {
        trade_date: day(12),
        code: CODE.to_string(),
        twap_0930_0935: 1.0,
        twap_0935_1000: 1_000.0,
    });
    ctx.base_by_code.insert(CODE.to_string(), base);
    ctx.price_by_code.insert(CODE.to_string(), price);
    ctx.twap_by_code.insert(CODE.to_string(), twap);

    assert_close(base_debt_premium_floor_gap(&ctx, CODE).unwrap(), 0.20);
    assert_close(dtwap_morning_slope20(&ctx, CODE).unwrap(), 0.01);

    // `dredemption_premium_z20` requires a finite terminal state, unlike the
    // catalogue gap's separate last-non-null semantics.
    let mut terminal_nan = ctx.clone();
    terminal_nan.base_by_code.get_mut(CODE).unwrap()[11].redemption_prem_ratio = f64::NAN;
    assert!(dredemption_premium_z20(&terminal_nan, CODE)
        .unwrap()
        .is_nan());
}

#[test]
fn rebound_requires_the_common_latest_price_anchor() {
    let mut ctx = TypedFactorDailyContext::new(day(14));
    let mut stale = Vec::new();
    for index in 0..12_i64 {
        stale.push(price_row(index, 100.0 + index as f64, 80.0, 100.0));
    }
    ctx.price_by_code.insert(CODE.to_string(), stale);

    let fresh_code = "110002.SH".to_string();
    let mut fresh = Vec::new();
    for index in 0..13_i64 {
        let mut row = price_row(index, 100.0 + index as f64, 80.0, 100.0);
        row.code = fresh_code.clone();
        fresh.push(row);
    }
    ctx.price_by_code.insert(fresh_code, fresh);
    assert!(drt_rebound_from_low20(&ctx, CODE).unwrap().is_nan());
}

#[test]
fn rebound_rejects_an_incomplete_terminal_strict_prior_row() {
    let mut ctx = TypedFactorDailyContext::new(day(13));
    let mut rows = Vec::new();
    for index in 0..13_i64 {
        rows.push(price_row(
            index,
            100.0 + index as f64,
            if index == 2 { 80.0 } else { 90.0 },
            100.0,
        ));
    }
    // Python `_complete_tail` fails closed when the raw terminal row is
    // incomplete, even though there are enough complete older observations.
    rows[12].amount = f64::NAN;
    ctx.price_by_code.insert(CODE.to_string(), rows);
    assert!(drt_rebound_from_low20(&ctx, CODE).unwrap().is_nan());
}

#[test]
fn duplicate_strict_prior_rows_fail_closed_and_dispatch_is_explicit() {
    let mut ctx = TypedFactorDailyContext::new(day(3));
    ctx.base_by_code.insert(
        CODE.to_string(),
        vec![
            base_row(0, 0.3, 0.1, 0.2, 0.1),
            base_row(0, 0.3, 0.1, 0.2, 0.1),
        ],
    );
    assert!(matches!(
        base_debt_premium_floor_gap(&ctx, CODE),
        Err(TypedFactorDailyError::DuplicateStrictPriorDate { .. })
    ));
    assert!(matches!(
        compute_p1_daily_signal(&ctx, CODE, "not_a_typed_factor_signal"),
        Err(TypedFactorDailyError::UnknownP1Signal(_))
    ));
}

#[test]
fn research_catalog_and_contract_extensions_keep_strict_prior_semantics() {
    let mut ctx = TypedFactorDailyContext::new(day(7));
    let mut base = Vec::new();
    let mut price = Vec::new();
    for index in 0..7_i64 {
        let mut row = base_row(index, 0.3, 0.1, 0.2, 0.05);
        row.duration = 4.0 + index as f64;
        row.stock_volatility = 0.20 + index as f64 * 0.01;
        row.cb_close_price = 100.0;
        row.conv_value = 125.0;
        row.trigger_cum_days = 3.0 + index as f64;
        row.trigger_reach_days = 20.0;
        row.trigger_cum_days_revise = 1.0 + index as f64 * 0.5;
        row.rating = if index == 6 { "AA+" } else { "AA" }.to_string();
        base.push(row);
        price.push(price_row(index, 100.0 + index as f64, 90.0, 100.0));
    }
    ctx.base_by_code.insert(CODE.to_string(), base);
    ctx.price_by_code.insert(CODE.to_string(), price);

    assert_close(
        base_duration_stockvol_interaction(&ctx, CODE).unwrap(),
        10.0 * 0.26,
    );
    assert_close(base_trigger_progress_ratio(&ctx, CODE).unwrap(), 9.0 / 20.0);
    assert_close(base_trigger_revision_gap(&ctx, CODE).unwrap(), 9.0 - 4.0);
    assert_close(
        base_stockvol_per_moneyness(&ctx, CODE).unwrap(),
        0.26 / 1.25,
    );
    assert_close(
        dret_momentum_5(&ctx, CODE).unwrap(),
        (106.0 - 101.0) / 101.0,
    );
    assert_close(rating_current_ordinal(&ctx, CODE).unwrap(), 17.0);

    // Score-day values are forbidden from all six typed extensions.
    let mut contaminated = ctx.clone();
    let mut score_base = base_row(7, 0.0, 0.0, 0.0, 0.0);
    score_base.duration = 9_999.0;
    score_base.stock_volatility = 9_999.0;
    score_base.cb_close_price = 1.0;
    score_base.conv_value = 9_999.0;
    score_base.trigger_cum_days = 9_999.0;
    score_base.trigger_reach_days = 1.0;
    score_base.trigger_cum_days_revise = -9_999.0;
    score_base.rating = "CCC".to_string();
    contaminated
        .base_by_code
        .get_mut(CODE)
        .unwrap()
        .push(score_base);
    contaminated
        .price_by_code
        .get_mut(CODE)
        .unwrap()
        .push(price_row(7, 9_999.0, 1.0, 1.0));
    for signal in [
        "base_duration_stockvol_interaction",
        "base_trigger_progress_ratio",
        "base_trigger_revision_gap",
        "base_stockvol_per_moneyness",
        "dret_momentum_5",
        "rating_current_ordinal",
    ] {
        assert_eq!(
            compute_p1_daily_signal(&ctx, CODE, signal)
                .unwrap()
                .to_bits(),
            compute_p1_daily_signal(&contaminated, CODE, signal)
                .unwrap()
                .to_bits(),
            "signal={signal}"
        );
    }
}
